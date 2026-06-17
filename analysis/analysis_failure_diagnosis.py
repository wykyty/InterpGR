"""
检索失败诊断分析：当模型检索失败时，SAE 特征层面出了什么问题？

核心问题：
1. 哪些特征在成功/失败时有显著差异？
2. 失败时是"路由特征"失活（走错分支），还是"细粒度区分特征"失活（前缀对了但最后错了）？

用法:
    cd /home/zyq/wyk/InterpGR
    python sae/analysis_failure_diagnosis.py 2>&1 | tee sae/failure_diagnosis.log
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
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


def analyze_failure_diagnosis():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"

    sae_dir = "out/sae_train_8x/layer_12"
    cache_path = "data/activation_cache_dev/layer_12.npz"
    comparison_dir = "dataset/nq320k/comparison"
    dev_data_path = "dataset/nq320k/dev.json"
    semantic_id_path = "dataset/nq320k_id/id.semantic.bert.json"
    output_path = "analysis/failure_diagnosis_report.md"
    device = "cuda"

    # ── 1. 加载数据 ──
    print("Loading data...")
    dev_data = json.load(open(dev_data_path))
    semantic_ids = json.load(open(semantic_id_path))
    top1_correct = json.load(open(f"{comparison_dir}/top1_correct.json"))
    top1_wrong = json.load(open(f"{comparison_dir}/top1_wrong.json"))

    dev_index_map = {}
    for i, (q, d) in enumerate(dev_data):
        if isinstance(q, list): q = q[0]
        if isinstance(d, list): d = d[0]
        dev_index_map[(q, d)] = i

    correct_idx = sorted(dev_index_map[(item['query'], item['doc_id'])] for item in top1_correct)
    wrong_idx = sorted(dev_index_map[(item['query'], item['doc_id'])] for item in top1_wrong)

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
    sample_features = np.zeros((n_samples, d_sae))
    counts = np.zeros(n_samples)
    for i in range(len(all_features)):
        sid = sample_indices[i].item()
        sample_features[sid] += all_features[i]
        counts[sid] += 1
    counts = np.maximum(counts, 1)
    sample_features = sample_features / counts[:, None]

    # ── 4. 分组 ──
    feats_success = sample_features[correct_idx]  # [n_success, d_sae]
    feats_failure = sample_features[wrong_idx]     # [n_failure, d_sae]
    n_success = len(correct_idx)
    n_failure = len(wrong_idx)
    print(f"  Success samples: {n_success}, Failure samples: {n_failure}")

    # ── 5. 逐特征差异分析 ──
    print("Computing per-feature statistics...")

    # 均值
    mean_success = feats_success.mean(axis=0)
    mean_failure = feats_failure.mean(axis=0)
    mean_diff = mean_success - mean_failure

    # 激活率 (值 > 0 的比例)
    rate_success = (feats_success > 0).mean(axis=0)
    rate_failure = (feats_failure > 0).mean(axis=0)
    rate_diff = rate_success - rate_failure

    # Cohen's d
    std_success = feats_success.std(axis=0, ddof=1)
    std_failure = feats_failure.std(axis=0, ddof=1)
    pooled_std = np.sqrt(((n_success - 1) * std_success**2 + (n_failure - 1) * std_failure**2) / (n_success + n_failure - 2))
    cohens_d = mean_diff / (pooled_std + 1e-8)

    # Top-k 选中率 (每个样本取 top-10 特征)
    k = 10
    topk_success = np.argsort(-feats_success, axis=1)[:, :k]
    topk_failure = np.argsort(-feats_failure, axis=1)[:, :k]

    topk_freq_success = np.zeros(d_sae)
    topk_freq_failure = np.zeros(d_sae)
    for j in range(k):
        idx, cnt = np.unique(topk_success[:, j], return_counts=True)
        topk_freq_success[idx] += cnt
        idx, cnt = np.unique(topk_failure[:, j], return_counts=True)
        topk_freq_failure[idx] += cnt
    topk_freq_success /= n_success
    topk_freq_failure /= n_failure
    topk_freq_diff = topk_freq_success - topk_freq_failure

    # ── 6. 筛选显著差异特征 ──
    print("Filtering significant features...")

    # 成功特征: Cohen's d > 0.3 且激活率差异 > 5%
    success_features = np.where((cohens_d > 0.3) & (rate_diff > 0.05))[0]
    success_features = success_features[np.argsort(-cohens_d[success_features])]

    # 失败特征: Cohen's d < -0.3 且激活率差异 < -5%
    failure_features = np.where((cohens_d < -0.3) & (rate_diff < -0.05))[0]
    failure_features = failure_features[np.argsort(cohens_d[failure_features])]

    # 更严格的：中等效应量
    success_medium = np.where((cohens_d > 0.5) & (rate_diff > 0.1))[0]
    success_medium = success_medium[np.argsort(-cohens_d[success_medium])]

    failure_medium = np.where((cohens_d < -0.5) & (rate_diff < -0.1))[0]
    failure_medium = failure_medium[np.argsort(cohens_d[failure_medium])]

    print(f"  Success features (d>0.3, rate_diff>5%): {len(success_features)}")
    print(f"  Failure features (d<-0.3, rate_diff<-5%): {len(failure_features)}")
    print(f"  Success features (d>0.5, rate_diff>10%): {len(success_medium)}")
    print(f"  Failure features (d<-0.5, rate_diff<-10%): {len(failure_medium)}")

    # ── 7. 分析特征类型: 路由 vs 细粒度 ──
    # 路由特征: top 激活样本的 DocID 前缀一致
    # 细粒度特征: top 激活样本的 DocID 前缀不同但语义相关

    def classify_feature_type(feat_id, sample_features, dev_data, semantic_ids, top_n=15):
        """分析一个特征的 top 激活样本，判断是路由特征还是语义特征。"""
        feat_vals = sample_features[:, feat_id]
        top_idx = np.argsort(-feat_vals)[:top_n]

        # 收集 top 样本的 DocID
        doc_ids = []
        queries = []
        for sid in top_idx:
            q = dev_data[sid][0]
            d = dev_data[sid][1]
            if isinstance(q, list): q = q[0]
            if isinstance(d, list): d = d[0]
            doc_ids.append(d)
            queries.append(q)

        # DocID 语义 ID
        sem_ids = [semantic_ids[d] for d in doc_ids]

        # 检查 DocID 前缀一致性 (取前 2 位)
        prefixes = [tuple(s[:2]) for s in sem_ids]
        from collections import Counter
        prefix_counts = Counter(prefixes)
        most_common_prefix, most_common_count = prefix_counts.most_common(1)[0]
        prefix_consistency = most_common_count / top_n

        # 检查主题一致性 (用 query 中的关键词)
        # 简单方法: 看 query 是否都包含某个关键词
        query_texts = [q.lower() for q in queries]

        return {
            "feat_id": feat_id,
            "top_queries": queries,
            "top_doc_ids": doc_ids,
            "top_sem_ids": sem_ids,
            "prefix_consistency": prefix_consistency,
            "most_common_prefix": most_common_prefix,
            "most_common_count": most_common_count,
        }

    # ── 8. 生成报告 ──
    print("Generating report...")
    lines = []
    lines.append("# 检索失败诊断报告：SAE 特征层面的分析\n")
    lines.append("## 核心问题\n")
    lines.append("检索失败时，模型的内部表征出了什么问题？")
    lines.append("是路由特征失活（走错分支），还是细粒度区分特征失活（前缀对了但最后错了）？\n")

    lines.append("## 1. 数据概览\n")
    lines.append(f"- 成功样本（top-1 检索正确）: {n_success} 条")
    lines.append(f"- 失败样本（top-1 检索错误）: {n_failure} 条")
    lines.append(f"- SAE 特征维度: {d_sae}")
    lines.append(f"- 每个 token 激活的特征数: ~100 (k=100)\n")

    # ── 成功特征 ──
    lines.append("## 2. 成功特征：成功时更活跃，失败时 \"熄火\"\n")
    lines.append("筛选标准: Cohen's d > 0.3（成功组均值更高）且激活率差异 > 5%\n")
    lines.append(f"共找到 **{len(success_features)}** 个成功特征。\n")

    # 展示 top 30
    lines.append("### Top-30 成功特征\n")
    lines.append("| 排名 | 特征ID | Cohen's d | 成功组均值 | 失败组均值 | 成功激活率 | 失败激活率 | 激活率差 | top-k选中率差 |")
    lines.append("|------|--------|-----------|-----------|-----------|-----------|-----------|---------|-------------|")
    for rank, fid in enumerate(success_features[:30]):
        lines.append("| {} | {} | {:.3f} | {:.2f} | {:.2f} | {:.1f}% | {:.1f}% | {:+.1f}% | {:+.1f}% |".format(
            rank+1, fid, cohens_d[fid], mean_success[fid], mean_failure[fid],
            rate_success[fid]*100, rate_failure[fid]*100, rate_diff[fid]*100, topk_freq_diff[fid]*100
        ))

    # 逐个分析 top-10 成功特征
    lines.append("\n### Top-10 成功特征详细分析\n")
    for rank, fid in enumerate(success_features[:10]):
        info = classify_feature_type(fid, sample_features, dev_data, semantic_ids)
        lines.append("#### Feature {} (Cohen's d = {:.3f})\n".format(fid, cohens_d[fid]))
        lines.append(f"**激活率**: 成功 {rate_success[fid]*100:.1f}% vs 失败 {rate_failure[fid]*100:.1f}% (差 {rate_diff[fid]*100:+.1f}%)")
        lines.append(f"**均值**: 成功 {mean_success[fid]:.2f} vs 失败 {mean_failure[fid]:.2f}")
        lines.append(f"**DocID 前缀一致性**: {info['prefix_consistency']*100:.0f}% (最常见前缀 {info['most_common_prefix']}, {info['most_common_count']}/{15})\n")

        # 判断类型
        if info['prefix_consistency'] > 0.5:
            feat_type = "**路由特征** — top 样本的 DocID 前缀高度一致"
        else:
            feat_type = "**语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散"
        lines.append(f"**类型判断**: {feat_type}\n")

        lines.append("| 排名 | 激活值 | Query | DocID (语义ID) |")
        lines.append("|------|--------|-------|---------------|")
        feat_vals = sample_features[:, fid]
        top_idx = np.argsort(-feat_vals)[:10]
        for r, sid in enumerate(top_idx):
            q = dev_data[sid][0]
            d = dev_data[sid][1]
            if isinstance(q, list): q = q[0]
            if isinstance(d, list): d = d[0]
            docid_str = get_docid_str(d, semantic_ids)
            if len(q) > 70: q = q[:67] + "..."
            q = q.replace("|", "/")
            lines.append("| {} | {:.2f} | {} | {} |".format(r+1, feat_vals[sid], q, docid_str))
        lines.append("")

    # ── 失败特征 ──
    lines.append("\n## 3. 失败特征：失败时更活跃，可能是 \"错误信号\"\n")
    lines.append("筛选标准: Cohen's d < -0.3（失败组均值更高）且激活率差异 < -5%\n")
    lines.append(f"共找到 **{len(failure_features)}** 个失败特征。\n")

    lines.append("### Top-30 失败特征\n")
    lines.append("| 排名 | 特征ID | Cohen's d | 成功组均值 | 失败组均值 | 成功激活率 | 失败激活率 | 激活率差 | top-k选中率差 |")
    lines.append("|------|--------|-----------|-----------|-----------|-----------|-----------|---------|-------------|")
    for rank, fid in enumerate(failure_features[:30]):
        lines.append("| {} | {} | {:.3f} | {:.2f} | {:.2f} | {:.1f}% | {:.1f}% | {:+.1f}% | {:+.1f}% |".format(
            rank+1, fid, cohens_d[fid], mean_success[fid], mean_failure[fid],
            rate_success[fid]*100, rate_failure[fid]*100, rate_diff[fid]*100, topk_freq_diff[fid]*100
        ))

    # 逐个分析 top-10 失败特征
    lines.append("\n### Top-10 失败特征详细分析\n")
    for rank, fid in enumerate(failure_features[:10]):
        info = classify_feature_type(fid, sample_features, dev_data, semantic_ids)
        lines.append("#### Feature {} (Cohen's d = {:.3f})\n".format(fid, cohens_d[fid]))
        lines.append(f"**激活率**: 成功 {rate_success[fid]*100:.1f}% vs 失败 {rate_failure[fid]*100:.1f}% (差 {rate_diff[fid]*100:+.1f}%)")
        lines.append(f"**均值**: 成功 {mean_success[fid]:.2f} vs 失败 {mean_failure[fid]:.2f}")
        lines.append(f"**DocID 前缀一致性**: {info['prefix_consistency']*100:.0f}% (最常见前缀 {info['most_common_prefix']}, {info['most_common_count']}/{15})\n")

        if info['prefix_consistency'] > 0.5:
            feat_type = "**路由特征** — top 样本的 DocID 前缀高度一致"
        else:
            feat_type = "**语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散"
        lines.append(f"**类型判断**: {feat_type}\n")

        lines.append("| 排名 | 激活值 | Query | DocID (语义ID) |")
        lines.append("|------|--------|-------|---------------|")
        feat_vals = sample_features[:, fid]
        top_idx = np.argsort(-feat_vals)[:10]
        for r, sid in enumerate(top_idx):
            q = dev_data[sid][0]
            d = dev_data[sid][1]
            if isinstance(q, list): q = q[0]
            if isinstance(d, list): d = d[0]
            docid_str = get_docid_str(d, semantic_ids)
            if len(q) > 70: q = q[:67] + "..."
            q = q.replace("|", "/")
            lines.append("| {} | {:.2f} | {} | {} |".format(r+1, feat_vals[sid], q, docid_str))
        lines.append("")

    # ── 中等效应量 ──
    lines.append("\n## 4. 中等效应量特征 (|d| > 0.5)\n")
    lines.append(f"成功特征 (d > 0.5, 激活率差 > 10%): **{len(success_medium)}** 个")
    lines.append(f"失败特征 (d < -0.5, 激活率差 < -10%): **{len(failure_medium)}** 个\n")

    if len(success_medium) > 0:
        lines.append("### 成功特征 (中等效应量)\n")
        lines.append("| 特征ID | Cohen's d | 成功激活率 | 失败激活率 | 激活率差 |")
        lines.append("|--------|-----------|-----------|-----------|---------|")
        for fid in success_medium[:20]:
            lines.append("| {} | {:.3f} | {:.1f}% | {:.1f}% | {:+.1f}% |".format(
                fid, cohens_d[fid], rate_success[fid]*100, rate_failure[fid]*100, rate_diff[fid]*100
            ))

    if len(failure_medium) > 0:
        lines.append("\n### 失败特征 (中等效应量)\n")
        lines.append("| 特征ID | Cohen's d | 成功激活率 | 失败激活率 | 激活率差 |")
        lines.append("|--------|-----------|-----------|-----------|---------|")
        for fid in failure_medium[:20]:
            lines.append("| {} | {:.3f} | {:.1f}% | {:.1f}% | {:+.1f}% |".format(
                fid, cohens_d[fid], rate_success[fid]*100, rate_failure[fid]*100, rate_diff[fid]*100
            ))

    # ── 总结 ──
    lines.append("\n## 5. 总结：检索失败时，模型内部出了什么问题？\n")

    # 统计成功/失败特征的类型分布
    success_routing = 0
    success_semantic = 0
    for fid in success_features[:20]:
        info = classify_feature_type(fid, sample_features, dev_data, semantic_ids)
        if info['prefix_consistency'] > 0.5:
            success_routing += 1
        else:
            success_semantic += 1

    failure_routing = 0
    failure_semantic = 0
    for fid in failure_features[:20]:
        info = classify_feature_type(fid, sample_features, dev_data, semantic_ids)
        if info['prefix_consistency'] > 0.5:
            failure_routing += 1
        else:
            failure_semantic += 1

    lines.append("### 成功特征的类型分布（top-20）\n")
    lines.append(f"- 路由特征（DocID 前缀一致 > 50%）: {success_routing} 个")
    lines.append(f"- 语义/细粒度特征（DocID 前缀分散）: {success_semantic} 个\n")

    lines.append("### 失败特征的类型分布（top-20）\n")
    lines.append(f"- 路由特征: {failure_routing} 个")
    lines.append(f"- 语义/细粒度特征: {failure_semantic} 个\n")

    lines.append("### 关键发现\n")
    lines.append("1. **成功特征**在成功样本中显著更活跃。这些特征可能编码了正确检索所需的关键信息——当它们 \"熄火\" 时，模型就会犯错。")
    lines.append("2. **失败特征**在失败样本中更活跃。这些可能是 \"干扰信号\"——模型被错误的模式吸引，导致走错分支。")
    lines.append("3. 通过观察特征的 top 激活样本，可以判断特征是编码 DocID 路由（前缀一致）还是编码细粒度语义（主题相关但前缀分散）。")
    lines.append("4. 如果成功特征主要是路由特征，说明失败是因为走错了大分支；如果成功特征主要是细粒度特征，说明失败是因为在最后一步区分能力不足。")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")

    # 保存原始数据
    np.savez(
        "analysis/failure_diagnosis_data.npz",
        cohens_d=cohens_d,
        rate_diff=rate_diff,
        mean_diff=mean_diff,
        topk_freq_diff=topk_freq_diff,
        success_features=success_features,
        failure_features=failure_features,
        success_medium=success_medium,
        failure_medium=failure_medium,
    )
    print("Raw data saved to analysis/failure_diagnosis_data.npz")


if __name__ == "__main__":
    analyze_failure_diagnosis()
