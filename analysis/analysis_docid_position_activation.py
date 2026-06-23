"""
DocID 分位条件概率分析

对于每个 SAE latent，当它激活时，统计 DocID 各位置取不同值的条件概率分布。
与全局基线对比，找出对特定 DocID 值有偏好的 latent。

条件概率: P(DocID[pos]=v | latent active) = #激活且该位置为v / #激活总数
全局基线: P(DocID[pos]=v) = #该位置为v的样本 / #总样本

区分度: KL( P(DocID|active) || P(DocID) ) — 越大说明该 latent 的激活越偏离随机

用法:
    cd /home/zyq/wyk/InterpGR
    uv run python analysis/analysis_docid_position_activation.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def kl_divergence(p, q, eps=1e-10):
    """KL(p || q)，p 和 q 是概率分布向量"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 配置路径
    sae_dir = "out/sae_train_8x/layer_12"
    cache_path = "data/activation_cache_dev/layer_12.npz"
    dev_data_path = "dataset/nq320k/dev.json"
    semantic_id_path = "dataset/nq320k_id/id.semantic.bert.json"
    output_dir = "analysis/docid_position_charts"
    report_path = "analysis/docid_position_activation_report.md"
    device = "cuda"

    positions = [0, 1, 2, 3]
    n_values = 30  # 每个位置取值 0-29

    # ── 1. 加载数据 ──
    print("Loading data...")
    dev_data = json.load(open(dev_data_path))
    semantic_ids = json.load(open(semantic_id_path))
    n_samples = len(dev_data)

    # per-sample 的 DocID 各位置值
    sample_pos_values = []  # sample_pos_values[i] = {pos: value}
    for i, (q, d) in enumerate(dev_data):
        if isinstance(d, list):
            d = d[0]
        sid = semantic_ids[d]
        pos_dict = {}
        for p in positions:
            if p < len(sid):
                pos_dict[p] = sid[p]
        sample_pos_values.append(pos_dict)

    print(f"  Samples: {n_samples}")
    for p in positions:
        n_with_p = sum(1 for pd in sample_pos_values if p in pd)
        print(f"  Position {p}: {n_with_p} samples")

    # ── 2. 全局基线分布 P(DocID[pos]=v) ──
    print("Computing baseline distributions...")
    baseline = {}  # baseline[pos] = array(30,) — 每个值的概率
    for p in positions:
        counts = np.zeros(n_values)
        for pd in sample_pos_values:
            if p in pd:
                counts[pd[p]] += 1
        total = counts.sum()
        baseline[p] = counts / total if total > 0 else np.zeros(n_values)

    # ── 3. 加载 SAE 并编码 ──
    print("Loading SAE and encoding...")
    sae = load_sae(sae_dir, device)

    data = np.load(cache_path)
    all_acts = torch.from_numpy(data["activations"])
    sample_indices = torch.from_numpy(data["sample_indices"])

    batch_size = 4096
    all_features = []
    for i in tqdm(range(0, len(all_acts), batch_size), desc="SAE encode"):
        batch = all_acts[i:i + batch_size].to(device)
        with torch.no_grad():
            features, _ = sae.encode_with_hidden_pre(batch)
        all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0).numpy()

    # ── 4. 按样本聚合（mean pooling）──
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

    # active_mask[i, fid] = True 表示样本 i 的 latent fid 激活
    active_mask = sample_features > 0  # (n_samples, d_sae)

    # ── 5. 计算条件概率 P(DocID[pos]=v | latent fid active) ──
    print("Computing conditional probabilities...")

    # 对每个 latent，统计激活样本在各 (pos, value) 上的计数
    # cond_counts[fid][pos][v] = # samples where latent fid active AND DocID[pos]=v
    # 用矩阵运算: 对每个 pos, 构建 (n_samples, n_values) 的 one-hot，乘以 active_mask

    # 先构建 per-position 的 value 数组
    pos_value_arrays = {}
    for p in positions:
        arr = np.zeros(n_samples, dtype=np.int32)
        has_pos = np.zeros(n_samples, dtype=bool)
        for i, pd in enumerate(sample_pos_values):
            if p in pd:
                arr[i] = pd[p]
                has_pos[i] = True
        pos_value_arrays[p] = (arr, has_pos)

    # cond_probs[fid][pos] = array(30,) — 条件概率分布
    # 用矩阵运算一次性算所有 latent
    cond_probs = {}
    for p in positions:
        print(f"  Position {p}...")
        val_arr, has_pos = pos_value_arrays[p]  # (n_samples,)
        valid = has_pos  # 样本是否在这个 position 有值

        # 对每个 value v，统计激活样本中该 value 的数量
        # cond_counts[:, v] = active_mask[valid, :].T @ one_hot(valid_val == v)
        # 更高效: 直接用 bincount
        p_cond_counts = np.zeros((d_sae, n_values))  # (d_sae, 30)
        for v in range(n_values):
            mask_v = valid & (val_arr == v)  # 样本在这个 position 值为 v
            if mask_v.any():
                p_cond_counts[:, v] = active_mask[mask_v].sum(axis=0)  # (d_sae,)

        # 归一化为条件概率
        total_active_per_fid = p_cond_counts.sum(axis=1, keepdims=True)  # (d_sae, 1)
        total_active_per_fid = np.maximum(total_active_per_fid, 1)
        p_cond_probs = p_cond_counts / total_active_per_fid  # (d_sae, 30)

        for fid in range(d_sae):
            if fid not in cond_probs:
                cond_probs[fid] = {}
            cond_probs[fid][p] = p_cond_probs[fid]

    # ── 6. 计算区分度: KL( P(DocID|active) || P(DocID) ) ──
    print("Computing KL divergence scores...")
    min_active = 50  # 最小激活样本数阈值

    # 每个 latent 的激活样本数
    n_active_per_fid = active_mask.sum(axis=0)  # (d_sae,)

    kl_scores = np.zeros((len(positions), d_sae))
    for pi, p in enumerate(positions):
        for fid in range(d_sae):
            if n_active_per_fid[fid] >= min_active:
                kl_scores[pi, fid] = kl_divergence(cond_probs[fid][p], baseline[p])
            else:
                kl_scores[pi, fid] = 0.0  # 激活太少，不参与排名

    max_kl = kl_scores.max(axis=0)  # (d_sae,)
    best_pos = kl_scores.argmax(axis=0)  # (d_sae,)

    # 统计过滤情况
    n_qualified = (n_active_per_fid >= min_active).sum()
    print(f"  激活样本 >= {min_active} 的 latent: {n_qualified}/{d_sae}")

    # ── 7. 画图（top-500，只画合格的 latent）──
    os.makedirs(output_dir, exist_ok=True)
    top_k_charts = 500
    # 只从合格的 latent 中选 top
    qualified_mask = n_active_per_fid >= min_active
    qualified_indices = np.where(qualified_mask)[0]
    qualified_kl = max_kl[qualified_indices]
    top_local = np.argsort(-qualified_kl)[:top_k_charts]
    top_indices = qualified_indices[top_local]

    print(f"Generating top-{top_k_charts} bar charts...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for rank, fid in enumerate(tqdm(top_indices, desc="Plotting")):
        fig.suptitle(f"Latent {fid} (rank {rank+1}, max KL={max_kl[fid]:.4f})",
                     fontsize=14, fontweight="bold")

        for pi, p in enumerate(positions):
            ax = axes[pi // 2][pi % 2]
            ax.cla()

            cond = cond_probs[fid][p]  # (30,)
            base = baseline[p]  # (30,)
            x = np.arange(n_values)
            width = 0.35

            ax.bar(x - width / 2, base, width, label="Baseline P(v)", color="gray", alpha=0.5)
            ax.bar(x + width / 2, cond, width, label=f"P(v|latent {fid})", color="steelblue", alpha=0.8)

            ax.set_xlabel(f"DocID Position {p} Value", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(f"Position {p} (KL={kl_scores[pi][fid]:.4f})", fontsize=11)
            ax.set_xlim(-1, n_values)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(0, n_values, 2))
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"latent_{fid}.png"), dpi=72, bbox_inches="tight")

    plt.close(fig)
    print(f"Charts saved to {output_dir}/")

    # ── 8. 生成汇总报告 ──
    print("Generating report...")
    lines = []
    lines.append("# DocID 分位条件概率分析报告\n")
    lines.append("## 核心思路\n")
    lines.append("对于每个 SAE latent，当它激活时，统计 DocID 各位置取不同值的 **条件概率** P(DocID[pos]=v | latent active)。")
    lines.append("与全局基线 P(DocID[pos]=v) 对比，用 KL 散度衡量偏离程度。")
    lines.append("KL 散度越大，说明该 latent 的激活越偏好某些特定的 DocID 值。\n")

    lines.append("## 方法\n")
    lines.append("- **条件概率**: P(DocID[pos]=v | latent active) = #激活样本中该位置为v / #该latent激活总数")
    lines.append("- **全局基线**: P(DocID[pos]=v) = #该位置为v的样本 / #有该位置的样本总数")
    lines.append("- **区分度**: KL( P(DocID[pos]=v | active) || P(DocID[pos]=v) ) — 越大越偏离随机")
    lines.append(f"- **过滤**: 仅统计激活样本数 >= {min_active} 的 latent（共 {n_qualified}/{d_sae} 个合格）\n")

    lines.append("## 数据概览\n")
    lines.append(f"- SAE 特征维度: {d_sae}")
    lines.append(f"- 样本数: {n_samples}")
    lines.append(f"- 分析位置: {positions}")
    lines.append(f"- 生成柱状图数: top-{top_k_charts}\n")

    for p in positions:
        n_with_p = sum(1 for pd in sample_pos_values if p in pd)
        lines.append(f"- Position {p}: {n_with_p} 个样本")

    # 全局基线展示
    lines.append("\n### 全局基线分布\n")
    for p in positions:
        top3 = np.argsort(-baseline[p])[:3]
        top3_str = ", ".join(f"v{v}={baseline[p][v]:.3f}" for v in top3)
        lines.append(f"- Position {p}: top3 [{top3_str}]")

    # Top-500 表格
    lines.append(f"\n## Top-{top_k_charts} 最有区分度的 Latent\n")
    lines.append("| 排名 | Latent | 最佳位置 | KL散度 | Pos 0 KL | Pos 1 KL | Pos 2 KL | Pos 3 KL | 激活样本数 |")
    lines.append("|------|--------|---------|--------|----------|----------|----------|----------|-----------|")
    for rank, fid in enumerate(top_indices):
        kl_str = " | ".join(f"{kl_scores[pi][fid]:.4f}" for pi in range(len(positions)))
        n_active = active_mask[:, fid].sum()
        lines.append(f"| {rank + 1} | {fid} | {positions[best_pos[fid]]} | {max_kl[fid]:.4f} | {kl_str} | {n_active} |")

    # Top-20 详细分析
    lines.append("\n## Top-20 Latent 详细分析\n")
    for rank, fid in enumerate(top_indices[:20]):
        bp = positions[best_pos[fid]]
        n_active = active_mask[:, fid].sum()
        lines.append(f"### Latent {fid}（最佳位置: {bp}, KL={max_kl[fid]:.4f}, 激活样本: {n_active}）\n")
        lines.append(f"![Latent {fid}](docid_position_charts/latent_{fid}.png)\n")

        for p in positions:
            cond = cond_probs[fid][p]
            base = baseline[p]
            # top-3 条件概率值
            top3_idx = np.argsort(-cond)[:3]
            bot3_idx = np.argsort(cond)[:3]
            top3_str = ", ".join(f"v{v}={cond[v]:.3f}(基线{base[v]:.3f})" for v in top3_idx)
            bot3_str = ", ".join(f"v{v}={cond[v]:.3f}(基线{base[v]:.3f})" for v in bot3_idx)
            lines.append(f"- **Position {p}** (KL={kl_scores[positions.index(p)][fid]:.4f}):")
            lines.append(f"  - top3: {top3_str}")
            lines.append(f"  - bottom3: {bot3_str}")
        lines.append("")

    # 已知 prefix routing features 验证
    lines.append("\n## 已知 Prefix Routing Features 验证\n")
    known_routing = [2125, 7760, 2236, 6972]
    for fid in known_routing:
        if fid >= d_sae:
            continue
        n_active = active_mask[:, fid].sum()
        lines.append(f"### Feature {fid}（激活样本: {n_active}）\n")
        lines.append(f"![Feature {fid}](docid_position_charts/latent_{fid}.png)\n")

        p = 0  # 主要看 position 0
        cond = cond_probs[fid][p]
        base = baseline[p]
        top5_idx = np.argsort(-cond)[:5]
        lines.append(f"**Position 0 条件概率** (KL={kl_scores[0][fid]:.4f}):\n")
        lines.append("| Value | P(v|active) | P(v) 基线 | 倍率 |")
        lines.append("|-------|------------|----------|------|")
        for v in top5_idx:
            ratio = cond[v] / max(base[v], 1e-10)
            lines.append(f"| {v} | {cond[v]:.3f} | {base[v]:.3f} | {ratio:.1f}x |")
        lines.append("")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # 保存原始数据
    np.savez(
        "analysis/docid_position_activation_data.npz",
        kl_scores=kl_scores,
        max_kl=max_kl,
        best_pos=best_pos,
        **{f"baseline_p{p}": baseline[p] for p in positions},
        **{f"cond_probs_fid{fid}_p{p}": cond_probs[fid][p]
           for fid in tqdm(top_indices[:200], desc="Saving data")
           for p in positions},
    )
    print("Raw data saved to analysis/docid_position_activation_data.npz")
    print("Done!")


if __name__ == "__main__":
    main()
