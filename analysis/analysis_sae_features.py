"""
用训练好的 SAE 放大第12层，分析 SAE 特征在好组/坏组之间的差异。

SAE 将 1024 维的 MLP 输出映射到 8192 维稀疏空间（每次只有 ~100 个特征激活），

用法:
    cd /home/zyq/wyk/InterpGR
    python sae/analysis_sae_features.py 2>&1 | tee sae/sae_feature_analysis.log
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    below = sorted_pvals <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool), pvals
    max_idx = np.max(np.where(below))
    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx[:max_idx + 1]] = True
    adjusted = np.minimum(1.0, sorted_pvals * n / np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1] if i + 1 < n else 1.0)
    pvals_adj = np.empty(n)
    pvals_adj[sorted_idx] = adjusted
    return reject, pvals_adj


def load_sae(sae_dir, device="cuda"):
    """加载训练好的 BatchTopK SAE（权重已 fold）。"""
    from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

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


def aggregate_per_sample(features, sample_indices, n_samples):
    """每个样本的多 token 特征取 mean。"""
    d_sae = features.shape[1]
    result = torch.zeros(n_samples, d_sae)
    counts = torch.zeros(n_samples)
    for i in range(len(features)):
        sid = sample_indices[i].item()
        result[sid] += features[i]
        counts[sid] += 1
    counts = counts.clamp(min=1)
    return result / counts.unsqueeze(1)


def analyze_sae_features(feats_good, feats_bad, k=10):
    """分析 SAE 特征在两组间的差异。"""
    n_good, d_sae = feats_good.shape
    n_bad = feats_bad.shape[0]

    # 1. 激活频率（特征值 > 0 的比例）
    freq_good = (feats_good > 0).float().mean(dim=0).numpy()
    freq_bad = (feats_bad > 0).float().mean(dim=0).numpy()
    freq_diff = freq_good - freq_bad

    # 2. 平均激活值（只算被激活的 token）
    mean_good = feats_good.mean(dim=0).numpy()
    mean_bad = feats_bad.mean(dim=0).numpy()

    # 3. 每个样本的 top-k 特征
    topk_good = torch.topk(feats_good, k, dim=1).indices.numpy()
    topk_bad = torch.topk(feats_bad, k, dim=1).indices.numpy()

    topk_freq_good = np.zeros(d_sae)
    topk_freq_bad = np.zeros(d_sae)
    for j in range(k):
        idx, cnt = np.unique(topk_good[:, j], return_counts=True)
        topk_freq_good[idx] += cnt
        idx, cnt = np.unique(topk_bad[:, j], return_counts=True)
        topk_freq_bad[idx] += cnt
    topk_freq_good /= n_good
    topk_freq_bad /= n_bad
    topk_freq_diff = topk_freq_good - topk_freq_bad

    # 4. 统计检验（只对有差异的候选做）
    cohens_d = np.zeros(d_sae)
    t_pvals = np.ones(d_sae)

    # 只对激活频率差异 > 0.5% 的特征做检验
    candidates = np.where(np.abs(freq_diff) > 0.005)[0]
    print(f"  Running t-test on {len(candidates)} candidate features (|freq_diff|>0.005)...")
    for j in tqdm(candidates, desc="t-test"):
        g = feats_good[:, j].numpy()
        b = feats_bad[:, j].numpy()
        t_stat, t_p = stats.ttest_ind(g, b, equal_var=False)
        t_pvals[j] = t_p
        pooled_std = np.sqrt((g.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
        cohens_d[j] = (g.mean() - b.mean()) / (pooled_std + 1e-8)

    reject, pvals_fdr = fdr_bh(t_pvals)

    return {
        "freq_good": freq_good,
        "freq_bad": freq_bad,
        "freq_diff": freq_diff,
        "mean_good": mean_good,
        "mean_bad": mean_bad,
        "topk_freq_good": topk_freq_good,
        "topk_freq_bad": topk_freq_bad,
        "topk_freq_diff": topk_freq_diff,
        "cohens_d": cohens_d,
        "t_pvals_fdr": pvals_fdr,
        "t_significant": reject,
        "n_good": n_good,
        "n_bad": n_bad,
        "d_sae": d_sae,
        "k": k,
    }


def generate_report(stats, output_path, top_n=30):
    freq_good = stats["freq_good"]
    freq_bad = stats["freq_bad"]
    freq_diff = stats["freq_diff"]
    topk_freq_good = stats["topk_freq_good"]
    topk_freq_bad = stats["topk_freq_bad"]
    topk_freq_diff = stats["topk_freq_diff"]
    mean_good = stats["mean_good"]
    mean_bad = stats["mean_bad"]
    cohens_d = stats["cohens_d"]
    t_sig = stats["t_significant"]
    d_sae = stats["d_sae"]
    k = stats["k"]

    lines = []
    lines.append("# Layer 12 SAE 特征分析报告\n")
    lines.append(f"**SAE 维度**: {d_sae}（原始 MLP 维度 1024 扩展到 {d_sae}，每次约 {k} 个特征激活）\n")
    lines.append(f"**好组 (top1 正确)**: {stats['n_good']} 个样本")
    lines.append(f"**坏组 (top1 错误)**: {stats['n_bad']} 个样本\n")

    # ── 1. 全局统计 ──
    lines.append("## 1. 全局统计\n")
    n_ever_active = ((freq_good > 0.001) | (freq_bad > 0.001)).sum()
    n_always_good = ((freq_good > 0.01) & (freq_bad < 0.001)).sum()
    n_always_bad = ((freq_bad > 0.01) & (freq_good < 0.001)).sum()
    n_sig = t_sig.sum()

    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| SAE 总特征数 | {d_sae} |")
    lines.append(f"| 至少在一组中激活过的特征 | {n_ever_active} ({n_ever_active/d_sae*100:.1f}%) |")
    lines.append(f"| 仅在好组出现的特征（好组频率>1%，坏组<0.1%） | {n_always_good} |")
    lines.append(f"| 仅在坏组出现的特征（坏组频率>1%，好组<0.1%） | {n_always_bad} |")
    lines.append(f"| 统计检验显著的特征（FDR<0.05） | {n_sig} |")
    lines.append(f"| |Cohen's d| > 0.1 的特征 | {(np.abs(cohens_d) > 0.1).sum()} |")
    lines.append(f"| |Cohen's d| > 0.2 的特征 | {(np.abs(cohens_d) > 0.2).sum()} |")

    # ── 2. 好组 top-k 频率更高的特征 ──
    lines.append(f"\n## 2. 好组更常被选为 Top-{k} 激活的特征\n")
    lines.append(f"每个样本取激活值最高的 {k} 个特征。以下特征在好组中更常被选入 top-{k}。\n")

    # 只选至少在一组中频率 > 0.5% 的
    mask = (topk_freq_good > 0.005) | (topk_freq_bad > 0.005)
    good_idx = np.where(mask & (topk_freq_diff > 0))[0]
    good_idx = good_idx[np.argsort(-topk_freq_diff[good_idx])][:top_n]

    lines.append(f"| 排名 | 特征ID | 好组选中率 | 坏组选中率 | 差距 | 激活频率(好) | 激活频率(坏) |")
    lines.append(f"|------|--------|-----------|-----------|------|------------|------------|")
    for rank, fid in enumerate(good_idx):
        lines.append(
            f"| {rank+1} | {fid} | {topk_freq_good[fid]*100:.2f}% | {topk_freq_bad[fid]*100:.2f}% | "
            f"{topk_freq_diff[fid]*100:+.2f}% | {freq_good[fid]*100:.2f}% | {freq_bad[fid]*100:.2f}% |"
        )

    # ── 3. 坏组 top-k 频率更高的特征 ──
    lines.append(f"\n## 3. 坏组更常被选为 Top-{k} 激活的特征\n")
    lines.append(f"以下特征在坏组中更常被选入 top-{k}。\n")

    bad_idx = np.where(mask & (topk_freq_diff < 0))[0]
    bad_idx = bad_idx[np.argsort(topk_freq_diff[bad_idx])][:top_n]

    lines.append(f"| 排名 | 特征ID | 好组选中率 | 坏组选中率 | 差距 | 激活频率(好) | 激活频率(坏) |")
    lines.append(f"|------|--------|-----------|-----------|------|------------|------------|")
    for rank, fid in enumerate(bad_idx):
        lines.append(
            f"| {rank+1} | {fid} | {topk_freq_good[fid]*100:.2f}% | {topk_freq_bad[fid]*100:.2f}% | "
            f"{topk_freq_diff[fid]*100:+.2f}% | {freq_good[fid]*100:.2f}% | {freq_bad[fid]*100:.2f}% |"
        )

    # ── 4. 激活频率差异最大的特征 ──
    lines.append(f"\n## 4. 激活频率差异最大的特征\n")
    lines.append(f"激活频率 = 该特征在所有 token 中被激活（值>0）的比例。\n")

    lines.append("### 好组激活频率更高的特征\n")
    active_mask = (freq_good > 0.005) | (freq_bad > 0.005)
    good_freq_idx = np.where(active_mask)[0]
    good_freq_idx = good_freq_idx[np.argsort(-freq_diff[good_freq_idx])][:top_n]

    lines.append(f"| 排名 | 特征ID | 好组频率 | 坏组频率 | 差距 |")
    lines.append(f"|------|--------|---------|---------|------|")
    for rank, fid in enumerate(good_freq_idx):
        lines.append(
            f"| {rank+1} | {fid} | {freq_good[fid]*100:.2f}% | {freq_bad[fid]*100:.2f}% | "
            f"{freq_diff[fid]*100:+.2f}% |"
        )

    lines.append(f"\n### 坏组激活频率更高的特征\n")
    bad_freq_idx = np.where(active_mask)[0]
    bad_freq_idx = bad_freq_idx[np.argsort(freq_diff[bad_freq_idx])][:top_n]

    lines.append(f"| 排名 | 特征ID | 好组频率 | 坏组频率 | 差距 |")
    lines.append(f"|------|--------|---------|---------|------|")
    for rank, fid in enumerate(bad_freq_idx):
        lines.append(
            f"| {rank+1} | {fid} | {freq_good[fid]*100:.2f}% | {freq_bad[fid]*100:.2f}% | "
            f"{freq_diff[fid]*100:+.2f}% |"
        )

    # ── 5. 仅在某一组出现的特征 ──
    lines.append(f"\n## 5. 仅在某一组出现的特征\n")

    only_good = np.where((freq_good > 0.01) & (freq_bad < 0.001))[0]
    only_bad = np.where((freq_bad > 0.01) & (freq_good < 0.001))[0]

    lines.append(f"以下特征只在好组中被激活（好组频率>1%，坏组<0.1%），共 {len(only_good)} 个：\n")
    if len(only_good) > 0:
        only_good_sorted = only_good[np.argsort(-freq_good[only_good])]
        lines.append(f"| 特征ID | 好组激活频率 | 好组平均激活值 |")
        lines.append(f"|--------|------------|--------------|")
        for fid in only_good_sorted[:50]:
            lines.append(f"| {fid} | {freq_good[fid]*100:.2f}% | {mean_good[fid]:.4f} |")

    lines.append(f"\n以下特征只在坏组中被激活（坏组频率>1%，好组<0.1%），共 {len(only_bad)} 个：\n")
    if len(only_bad) > 0:
        only_bad_sorted = only_bad[np.argsort(-freq_bad[only_bad])]
        lines.append(f"| 特征ID | 坏组激活频率 | 坏组平均激活值 |")
        lines.append(f"|--------|------------|--------------|")
        for fid in only_bad_sorted[:50]:
            lines.append(f"| {fid} | {freq_bad[fid]*100:.2f}% | {mean_bad[fid]:.4f} |")

    # ── 6. 总结 ──
    lines.append(f"\n## 6. 总结\n")
    lines.append(f"- SAE 将 1024 维 MLP 输出扩展到 {d_sae} 维稀疏空间，每次只有约 {k} 个特征激活。")
    lines.append(f"- 共 {n_ever_active} 个特征（{n_ever_active/d_sae*100:.1f}%）至少在一组中被使用过。")
    lines.append(f"- {n_always_good} 个特征仅在好组出现，{n_always_bad} 个仅在坏组出现——这些是最有区分度的特征。")
    lines.append(f"- 统计检验显著的特征有 {n_sig} 个，说明好/坏组在 SAE 特征空间中确实有系统性差异。")
    lines.append(f"- 与原始神经元分析（72% 显著但效应量小）相比，SAE 特征的稀疏性使得差异更容易定位到具体的特征上。")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"

    sae_dir = "out/sae_train_8x/layer_12"
    cache_path = "data/activation_cache_dev/layer_12.npz"
    comparison_dir = "dataset/nq320k/comparison"
    dev_data_path = "dataset/nq320k/dev.json"
    device = "cuda"

    # ── 1. 加载缓存的激活值 ──
    print("Loading cached activations...")
    data = np.load(cache_path)
    all_acts = torch.from_numpy(data["activations"])
    sample_indices = torch.from_numpy(data["sample_indices"])
    print(f"  Shape: {all_acts.shape}")

    # ── 2. 加载 SAE ──
    print(f"Loading SAE from {sae_dir}...")
    sae = load_sae(sae_dir, device)
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, k={sae.cfg.k}")

    # ── 3. 编码为 SAE 特征 ──
    print("Encoding activations through SAE...")
    batch_size = 4096
    all_features = []
    for i in tqdm(range(0, len(all_acts), batch_size), desc="SAE encode"):
        batch = all_acts[i:i+batch_size].to(device)
        with torch.no_grad():
            features, _ = sae.encode_with_hidden_pre(batch)
        all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0)
    print(f"  SAE features shape: {all_features.shape}")
    print(f"  Avg active features per token: {(all_features > 0).float().sum(dim=1).mean():.1f}")

    # ── 4. 按样本聚合 ──
    dev_data = json.load(open(dev_data_path))
    n_samples = len(dev_data)
    print(f"Aggregating per-sample ({n_samples} samples)...")
    sample_features = aggregate_per_sample(all_features, sample_indices, n_samples)

    # ── 5. 分组 ──
    top1_correct = json.load(open(f"{comparison_dir}/top1_correct.json"))
    top1_wrong = json.load(open(f"{comparison_dir}/top1_wrong.json"))

    dev_index_map = {}
    for i, (q, d) in enumerate(dev_data):
        if isinstance(q, list): q = q[0]
        if isinstance(d, list): d = d[0]
        dev_index_map[(q, d)] = i

    correct_idx = sorted(dev_index_map[(item['query'], item['doc_id'])] for item in top1_correct)
    wrong_idx = sorted(dev_index_map[(item['query'], item['doc_id'])] for item in top1_wrong)

    feats_good = sample_features[correct_idx]
    feats_bad = sample_features[wrong_idx]
    print(f"  Good: {feats_good.shape}, Bad: {feats_bad.shape}")

    # ── 6. 分析 ──
    print("\nAnalyzing SAE features...")
    stats = analyze_sae_features(feats_good, feats_bad, k=10)

    # ── 7. 打印摘要 ──
    topk_diff = stats["topk_freq_diff"]
    freq_diff = stats["freq_diff"]

    print(f"\n{'='*70}")
    # top-k 频率差异最大的
    mask = (stats["topk_freq_good"] > 0.005) | (stats["topk_freq_bad"] > 0.005)
    good_topk = np.where(mask & (topk_diff > 0))[0]
    good_topk = good_topk[np.argsort(-topk_diff[good_topk])][:10]
    print("好组更常选为 top-10 的特征:")
    for fid in good_topk:
        print(f"  feature {fid:5d}: good={stats['topk_freq_good'][fid]*100:.2f}%, bad={stats['topk_freq_bad'][fid]*100:.2f}%, diff={topk_diff[fid]*100:+.2f}%")

    bad_topk = np.where(mask & (topk_diff < 0))[0]
    bad_topk = bad_topk[np.argsort(topk_diff[bad_topk])][:10]
    print("坏组更常选为 top-10 的特征:")
    for fid in bad_topk:
        print(f"  feature {fid:5d}: good={stats['topk_freq_good'][fid]*100:.2f}%, bad={stats['topk_freq_bad'][fid]*100:.2f}%, diff={topk_diff[fid]*100:+.2f}%")

    # 仅在一组出现的
    only_good = ((stats["freq_good"] > 0.01) & (stats["freq_bad"] < 0.001)).sum()
    only_bad = ((stats["freq_bad"] > 0.01) & (stats["freq_good"] < 0.001)).sum()
    print(f"\n仅好组出现: {only_good} 个特征, 仅坏组出现: {only_bad} 个特征")
    print(f"统计检验显著: {stats['t_significant'].sum()} 个特征")
    print(f"{'='*70}")

    # ── 8. 生成报告 ──
    report_path = "sae/sae_feature_analysis_report.md"
    generate_report(stats, report_path)

    # ── 9. 保存原始数据 ──
    np.savez(
        "sae/sae_feature_analysis_data.npz",
        freq_good=stats["freq_good"],
        freq_bad=stats["freq_bad"],
        topk_freq_good=stats["topk_freq_good"],
        topk_freq_bad=stats["topk_freq_bad"],
        cohens_d=stats["cohens_d"],
        t_significant=stats["t_significant"],
    )
    print("Raw data saved to sae/sae_feature_analysis_data.npz")


if __name__ == "__main__":
    main()
