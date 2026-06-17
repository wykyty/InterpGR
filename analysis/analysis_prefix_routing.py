"""
DocID 层次结构在模型内部如何存储和建模？

核心问题：是否存在一组 SAE 特征，在模型准备生成特定前缀时一致地、排他地激活？
如果存在，这些就是"前缀路由特征"——直接证明模型内部存储了树结构的离散路由信号。

分析方法：
1. 将样本按 DocID 第一层前缀（Semantic ID 第一个 token, 0-29）分组
2. 对每个前缀组，计算每个 SAE 特征的平均激活值
3. 与全局基线比较，筛选出对特定前缀显著激活的特征
4. 区分"前缀路由特征"（排他性高）和"通用语义特征"（多个前缀共享）

用法:
    cd /home/zyq/wyk/InterpGR
    python sae/analysis_prefix_routing.py 2>&1 | tee sae/prefix_routing.log
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_sae(sae_dir, device="cuda"):
    from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
    from safetensors.torch import load_file
    cfg = json.load(open(f"{sae_dir}/sae_config.json"))
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=int(cfg["k"]),
        aux_loss_coefficient=1.0, rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.1, apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in", decoder_init_norm=0.01,
        device=str(device), dtype="float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg).to(device)
    weights = load_file(f"{sae_dir}/sae_weights.safetensors")
    sae.W_enc.data = weights["W_enc"].to(device)
    sae.W_dec.data = weights["W_dec"].to(device)
    sae.b_enc.data = weights["b_enc"].to(device)
    sae.b_dec.data = weights["b_dec"].to(device)
    sae.topk_threshold = weights["topk_threshold"].to(device)
    sae.eval()
    return sae


def get_docid_str(doc_id, semantic_ids):
    if isinstance(doc_id, list): doc_id = doc_id[0]
    sid = semantic_ids[doc_id]
    return "[" + ", ".join(str(x) for x in sid) + "]"


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"

    sae_dir = "out/sae_train_8x/layer_12"
    cache_path = "data/activation_cache_dev/layer_12.npz"
    dev_data_path = "dataset/nq320k/dev.json"
    semantic_id_path = "dataset/nq320k_id/id.semantic.bert.json"
    output_path = "analysis/prefix_routing_report.md"
    device = "cuda"

    # ── 1. 加载数据 ──
    print("Loading data...")
    dev_data = json.load(open(dev_data_path))
    semantic_ids = json.load(open(semantic_id_path))

    # 建立 prefix -> sample_indices 映射
    prefix_to_samples = defaultdict(list)
    sample_to_prefix = {}
    for i, (q, d) in enumerate(dev_data):
        if isinstance(d, list): d = d[0]
        sid = semantic_ids[d]
        p = sid[0]  # 第一层前缀
        prefix_to_samples[p].append(i)
        sample_to_prefix[i] = p

    n_prefixes = len(prefix_to_samples)
    print(f"  Found {n_prefixes} unique first-level prefixes")
    for p in sorted(prefix_to_samples.keys()):
        print(f"    prefix {p:2d}: {len(prefix_to_samples[p])} samples")

    # ── 2. 加载 SAE 并编码 ──
    print("Loading SAE and encoding...")
    sae = load_sae(sae_dir, device)

    data = np.load(cache_path)
    all_acts = torch.from_numpy(data["activations"])
    sample_indices = torch.from_numpy(data["sample_indices"])

    batch_size = 4096
    all_features = []
    for i in tqdm(range(0, len(all_acts), batch_size), desc="SAE encode"):
        batch = all_acts[i:i+batch_size].to(device)
        with torch.no_grad():
            features, _ = sae.encode_with_hidden_pre(batch)
        all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0).numpy()

    # ── 3. 按样本聚合 ──
    n_samples = len(dev_data)
    d_sae = all_features.shape[1]
    print(f"Aggregating per-sample ({n_samples} samples, d_sae={d_sae})...")
    sample_features = np.zeros((n_samples, d_sae))
    counts = np.zeros(n_samples)
    for i in range(len(all_features)):
        sid = sample_indices[i].item()
        sample_features[sid] += all_features[i]
        counts[sid] += 1
    counts = np.maximum(counts, 1)
    sample_features = sample_features / counts[:, None]

    # ── 4. 计算每个前缀组的特征均值 ──
    print("Computing per-prefix feature means...")
    # 全局基线
    global_mean = sample_features.mean(axis=0)
    global_std = sample_features.std(axis=0)

    # 每个前缀的均值
    prefix_means = {}
    prefix_sizes = {}
    for p in sorted(prefix_to_samples.keys()):
        indices = prefix_to_samples[p]
        prefix_means[p] = sample_features[indices].mean(axis=0)
        prefix_sizes[p] = len(indices)

    # ── 5. Z-score: 每个特征对每个前缀的偏离程度 ──
    print("Computing z-scores...")
    # z[p][f] = (prefix_mean[p][f] - global_mean[f]) / global_std[f]
    prefix_zscores = {}
    for p in sorted(prefix_means.keys()):
        prefix_zscores[p] = (prefix_means[p] - global_mean) / (global_std + 1e-8)

    # ── 6. 筛选前缀特异特征 ──
    # 对每个前缀，找 z-score 最高的特征
    z_threshold = 2.0  # z > 2 表示显著高于全局均值

    prefix_specific_features = {}  # prefix -> [(feat_id, z_score)]
    for p in sorted(prefix_zscores.keys()):
        z = prefix_zscores[p]
        sig_idx = np.where(z > z_threshold)[0]
        sig_idx = sig_idx[np.argsort(-z[sig_idx])]
        prefix_specific_features[p] = [(fid, z[fid]) for fid in sig_idx]

    # ── 7. 筛选前缀路由特征 ──
    # 路由特征定义：对某个前缀 z 很高，对其他前缀 z 很低
    # 排他性 = max_z / second_max_z（越大越排他）
    print("Identifying prefix routing features...")

    # 对每个特征，计算它对哪个前缀最特异
    feature_primary_prefix = {}  # feat_id -> (best_prefix, best_z, second_best_z, exclusivity)
    for fid in range(d_sae):
        z_per_prefix = [prefix_zscores[p][fid] for p in sorted(prefix_zscores.keys())]
        z_arr = np.array(z_per_prefix)
        sorted_z = np.sort(z_arr)[::-1]
        best_p = sorted(prefix_zscores.keys())[np.argmax(z_arr)]
        best_z = sorted_z[0]
        second_z = sorted_z[1] if len(sorted_z) > 1 else 0
        # 排他性：best_z 比 second_z 大多少
        if second_z > 0:
            exclusivity = best_z / second_z
        else:
            exclusivity = best_z / 0.01  # 避免除零
        feature_primary_prefix[fid] = (best_p, best_z, second_z, exclusivity)

    # 前缀路由特征：z > 2 且排他性 > 1.5（best 比 second 高 50% 以上）
    routing_features = []
    for fid in range(d_sae):
        best_p, best_z, second_z, exclusivity = feature_primary_prefix[fid]
        if best_z > z_threshold and exclusivity > 1.5:
            routing_features.append((fid, best_p, best_z, second_z, exclusivity))
    routing_features.sort(key=lambda x: -x[2])  # 按 best_z 降序

    # ── 8. 生成报告 ──
    print("Generating report...")
    lines = []
    lines.append("# DocID 层次结构在模型内部的存储分析\n")
    lines.append("## 核心问题\n")
    lines.append("是否存在一组 SAE 特征，在模型准备生成特定前缀时一致地、排他地激活？")
    lines.append("如果存在，这些就是 **前缀路由特征**——直接证明模型内部存储了树结构的离散路由信号。\n")

    lines.append("## 1. 数据概览\n")
    lines.append(f"- SAE 特征维度: {d_sae}")
    lines.append(f"- 样本数: {n_samples}")
    lines.append(f"- 第一层前缀数: {n_prefixes}（Semantic ID 第一个 token, 取值 0-29）\n")

    lines.append("| 前缀 | 样本数 | 占比 |")
    lines.append("|------|--------|------|")
    for p in sorted(prefix_to_samples.keys()):
        n = prefix_sizes[p]
        lines.append("| {} | {} | {:.1f}% |".format(p, n, n/n_samples*100))

    lines.append("\n## 2. 前缀路由特征\n")
    lines.append(f"**筛选标准**: 对某前缀 z-score > {z_threshold}，且排他性 > 1.5（最高 z 比第二高 z 高 50% 以上）\n")
    lines.append(f"共找到 **{len(routing_features)}** 个前缀路由特征。\n")

    if routing_features:
        lines.append("### 完整列表\n")
        lines.append("| 排名 | 特征ID | 目标前缀 | z-score | 第二高z | 排他性 | 目标前缀样本数 |")
        lines.append("|------|--------|---------|---------|---------|--------|-------------|")
        for rank, (fid, best_p, best_z, second_z, exc) in enumerate(routing_features[:50]):
            lines.append("| {} | {} | {} | {:.2f} | {:.2f} | {:.2f}x | {} |".format(
                rank+1, fid, best_p, best_z, second_z, exc, prefix_sizes[best_p]
            ))

        # 按前缀分组展示
        lines.append("\n### 按前缀分组\n")
        routing_by_prefix = defaultdict(list)
        for fid, best_p, best_z, second_z, exc in routing_features:
            routing_by_prefix[best_p].append((fid, best_z, exc))

        for p in sorted(routing_by_prefix.keys()):
            feats = routing_by_prefix[p]
            lines.append(f"#### 前缀 {p}（{prefix_sizes[p]} 个样本，{len(feats)} 个路由特征）\n")
            lines.append("| 特征ID | z-score | 排他性 |")
            lines.append("|--------|---------|--------|")
            for fid, z, exc in feats[:10]:
                lines.append("| {} | {:.2f} | {:.2f}x |".format(fid, z, exc))

            # 展示 top-3 路由特征的 top 激活样本
            lines.append("")
            for fid, z, exc in feats[:3]:
                feat_vals = sample_features[:, fid]
                top_idx = np.argsort(-feat_vals)[:5]
                lines.append("**Feature {}** (z={:.2f}, 排他性={:.2f}x) 的 top-5 激活样本：\n".format(fid, z, exc))
                lines.append("| 激活值 | Query | DocID | 第一前缀 |")
                lines.append("|--------|-------|-------|---------|")
                for sid in top_idx:
                    q = dev_data[sid][0]
                    d = dev_data[sid][1]
                    if isinstance(q, list): q = q[0]
                    if isinstance(d, list): d = d[0]
                    docid_str = get_docid_str(d, semantic_ids)
                    p_val = sample_to_prefix[sid]
                    if len(q) > 60: q = q[:57] + "..."
                    q = q.replace("|", "/")
                    lines.append("| {:.2f} | {} | {} | {} |".format(feat_vals[sid], q, docid_str, p_val))
                lines.append("")

    # ── 9. 每个前缀的 top-10 特异特征 ──
    lines.append("\n## 3. 每个前缀的 Top-10 特异特征\n")
    lines.append("按 z-score 降序排列，展示每个前缀最特异的特征。\n")

    for p in sorted(prefix_specific_features.keys()):
        feats = prefix_specific_features[p]
        if len(feats) == 0:
            continue
        lines.append(f"### 前缀 {p}（{prefix_sizes[p]} 个样本）\n")
        lines.append("| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |")
        lines.append("|------|--------|---------|--------------|------------|")
        for rank, (fid, z) in enumerate(feats[:10]):
            p_rate = (sample_features[prefix_to_samples[p]] > 0)[:, fid].mean() * 100
            g_rate = (sample_features > 0)[:, fid].mean() * 100
            lines.append("| {} | {} | {:.2f} | {:.1f}% | {:.1f}% |".format(
                rank+1, fid, z, p_rate, g_rate
            ))
        lines.append("")

    # ── 10. 共享特征分析 ──
    lines.append("\n## 4. 前缀间共享特征分析\n")
    lines.append("有些特征在多个前缀上都有高 z-score，可能是更通用的语义特征。\n")

    # 对每个特征，统计它在多少个前缀上 z > 2
    feature_prefix_count = np.zeros(d_sae)
    for fid in range(d_sae):
        for p in prefix_zscores:
            if prefix_zscores[p][fid] > z_threshold:
                feature_prefix_count[fid] += 1

    # 独占特征（仅 1 个前缀 z > 2）
    exclusive_features = np.where(feature_prefix_count == 1)[0]
    # 共享特征（>= 3 个前缀 z > 2）
    shared_features = np.where(feature_prefix_count >= 3)[0]

    lines.append(f"- 仅在 1 个前缀上显著的特征（独占特征）: **{len(exclusive_features)}** 个")
    lines.append(f"- 在 >= 3 个前缀上显著的特征（共享特征）: **{len(shared_features)}** 个")
    lines.append(f"- 在 >= 5 个前缀上显著的特征（高度共享）: **{(feature_prefix_count >= 5).sum()}** 个\n")

    # 独占特征的前缀分布
    lines.append("### 独占特征的前缀分布\n")
    lines.append("| 前缀 | 独占特征数 |")
    lines.append("|------|-----------|")
    for p in sorted(prefix_to_samples.keys()):
        n_excl = sum(1 for fid in exclusive_features if feature_primary_prefix[fid][0] == p)
        if n_excl > 0:
            lines.append("| {} | {} |".format(p, n_excl))

    # ── 11. 总结 ──
    lines.append("\n## 5. 总结\n")
    lines.append("### 关键发现\n")

    if routing_features:
        # 按前缀统计路由特征数
        prefix_routing_count = defaultdict(int)
        for fid, best_p, best_z, second_z, exc in routing_features:
            prefix_routing_count[best_p] += 1
        max_routing_prefix = max(prefix_routing_count, key=prefix_routing_count.get)

        lines.append(f"1. **前缀路由特征存在**。共找到 {len(routing_features)} 个特征，它们对特定前缀高度特异（z > {z_threshold}）且排他性强（排他性 > 1.5x）。")
        lines.append(f"2. 路由特征最多的是前缀 {max_routing_prefix}（{prefix_routing_count[max_routing_prefix]} 个），说明该前缀在 SAE 特征空间中有最清晰的 \"签名\"。")
        lines.append(f"3. 独占特征（仅 1 个前缀显著）有 {len(exclusive_features)} 个，说明模型内部确实存在针对特定前缀的离散路由信号。")
        lines.append(f"4. 共享特征（>= 3 个前缀显著）有 {len(shared_features)} 个，这些可能是更通用的语义编码（如 \"is a question about X\" 的特征）。")
        lines.append("")
        lines.append("### 含义\n")
        lines.append("- 模型内部确实存储了 DocID 层次结构的离散路由信号，表现为 SAE 特征空间中的前缀特异激活模式。")
        lines.append("- 当模型准备生成某个前缀时，对应的路由特征会被激活，引导解码器走向正确的分支。")
        lines.append("- 路由特征的存在支持了 \"生成式检索模型内部维护了一棵隐式决策树\" 的假设。")
    else:
        lines.append(f"1. 在当前阈值（z > {z_threshold}，排他性 > 1.5x）下，**未找到前缀路由特征**。")
        lines.append("2. 这可能意味着：")
        lines.append("   - 前缀路由信号分散在多个特征的组合中，而非集中在单个特征上")
        lines.append("   - 阈值设置过严，需要降低 z-score 或排他性阈值")
        lines.append("   - Layer 12 的 SAE 主要编码语义信息，路由信息可能在更深层")
        lines.append("3. 独占特征有 {} 个，共享特征有 {} 个，说明特征空间中确实存在前缀特异性，但排他性不够强。".format(len(exclusive_features), len(shared_features)))

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")

    # 保存原始数据
    np.savez(
        "analysis/prefix_routing_data.npz",
        global_mean=global_mean,
        global_std=global_std,
        **{f"prefix_mean_{p}": prefix_means[p] for p in prefix_means},
        **{f"prefix_zscore_{p}": prefix_zscores[p] for p in prefix_zscores},
    )
    print("Raw data saved to analysis/prefix_routing_data.npz")


if __name__ == "__main__":
    main()
