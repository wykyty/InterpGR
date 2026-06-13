"""
分析 t5-large 生成式检索模型第12层 decoder MLP 的神经元激活分布差异。
对比 top1 正确组 vs 错误组。

分析方法:
  1. 每个神经元做 t-test + Mann-Whitney U 检验
  2. Cohen's d 效应量
  3. Top-k 激活神经元排名差异

用法:
    cd /home/zyq/wyk/InterpGR
    python sae/analysis_neuron.py 2>&1 | tee sae/neuron_analysis.log
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction (no statsmodels dependency)."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    # Find the largest i where p(i) <= threshold(i)
    below = sorted_pvals <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool), pvals
    max_idx = np.max(np.where(below))
    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx[:max_idx + 1]] = True
    # Adjusted p-values
    adjusted = np.minimum(1.0, sorted_pvals * n / np.arange(1, n + 1))
    # Make monotonic
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1] if i + 1 < n else 1.0)
    pvals_adj = np.empty(n)
    pvals_adj[sorted_idx] = adjusted
    return reject, pvals_adj
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# 1. 收集 dev 集激活值
# ──────────────────────────────────────────────
def collect_dev_activations(
    checkpoint_path, dev_data, semantic_ids, cache_path,
    layer=12, context_size=32, device="cuda",
):
    if os.path.exists(cache_path):
        print(f"Loading cached activations from {cache_path}")
        data = np.load(cache_path)
        return (
            torch.from_numpy(data["activations"]),
            torch.from_numpy(data["sample_indices"]),
        )

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
    tokenizer.add_tokens([f'${i}$' for i in range(30)])

    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-large")
    model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    hook_target = model.decoder.block[layer].layer[2].DenseReluDense
    activations_cache = []

    def hook_fn(module, input, output):
        activations_cache.append(output.detach().cpu())

    hook = hook_target.register_forward_hook(hook_fn)

    all_acts = []
    sample_indices = []

    for idx, (query, doc_id) in enumerate(tqdm(dev_data, desc="Caching dev activations")):
        if isinstance(query, list):
            query = query[0]
        if isinstance(doc_id, list):
            doc_id = doc_id[0]

        sem_id = semantic_ids[doc_id]
        enc_tokens = tokenizer(
            query, return_tensors="pt", truncation=True,
            max_length=context_size, padding=False,
        )
        target_str = ''.join([f'${i}$' for i in sem_id])
        target_tokens = tokenizer(target_str, return_tensors="pt", add_special_tokens=False).input_ids
        pad_token = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long)
        dec_input = torch.cat([pad_token, target_tokens[:, :-1]], dim=1).to(device)

        activations_cache.clear()
        with torch.no_grad():
            model(
                input_ids=enc_tokens.input_ids.to(device),
                attention_mask=enc_tokens.attention_mask.to(device),
                decoder_input_ids=dec_input,
            )

        if activations_cache:
            acts = activations_cache[0].squeeze(0).float()
            all_acts.append(acts)
            sample_indices.extend([idx] * acts.shape[0])

    hook.remove()

    all_acts = torch.cat(all_acts, dim=0)
    sample_indices = torch.tensor(sample_indices, dtype=torch.long)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, activations=all_acts.numpy(), sample_indices=sample_indices.numpy())
    print(f"Saved activations: {all_acts.shape}, indices: {sample_indices.shape}")
    return all_acts, sample_indices


# ──────────────────────────────────────────────
# 2. 分析: 每个样本聚合成一个向量，再做统计检验
# ──────────────────────────────────────────────
def aggregate_per_sample(acts, sample_indices, n_samples):
    """每个样本的多 token 激活值取 mean，聚合成 [n_samples, d_model]。"""
    d_model = acts.shape[1]
    result = torch.zeros(n_samples, d_model)
    counts = torch.zeros(n_samples)
    for i in range(len(acts)):
        sid = sample_indices[i].item()
        result[sid] += acts[i]
        counts[sid] += 1
    counts = counts.clamp(min=1)
    result = result / counts.unsqueeze(1)
    return result


def analyze_statistical(acts_good_np, acts_bad_np, alpha=0.05):
    """对每个神经元做统计检验 + 效应量。

    返回每个神经元的:
      - mean_good, mean_bad
      - t_stat, t_pvalue (独立 t 检验)
      - u_stat, u_pvalue (Mann-Whitney U 检验)
      - cohens_d (效应量)
    """
    n_neurons = acts_good_np.shape[1]
    n_good = acts_good_np.shape[0]
    n_bad = acts_bad_np.shape[0]

    mean_good = acts_good_np.mean(axis=0)
    mean_bad = acts_bad_np.mean(axis=0)
    std_good = acts_good_np.std(axis=0, ddof=1)
    std_bad = acts_bad_np.std(axis=0, ddof=1)

    # Cohen's d
    pooled_std = np.sqrt(((n_good - 1) * std_good**2 + (n_bad - 1) * std_bad**2) / (n_good + n_bad - 2))
    cohens_d = (mean_good - mean_bad) / (pooled_std + 1e-8)

    # t-test (向量化)
    t_stats = np.zeros(n_neurons)
    t_pvals = np.zeros(n_neurons)
    for j in range(n_neurons):
        t_stat, t_p = stats.ttest_ind(acts_good_np[:, j], acts_bad_np[:, j], equal_var=False)
        t_stats[j] = t_stat
        t_pvals[j] = t_p

    # Mann-Whitney U (向量化，只对 top 候选做)
    u_pvals = np.ones(n_neurons)
    # 只对 |cohens_d| > 0.05 的神经元做 U 检验（节省时间）
    candidates = np.where(np.abs(cohens_d) > 0.05)[0]
    print(f"  Running Mann-Whitney U on {len(candidates)} candidate neurons (|d|>0.05)...")
    for j in tqdm(candidates, desc="Mann-Whitney U"):
        _, u_p = stats.mannwhitneyu(acts_good_np[:, j], acts_bad_np[:, j], alternative='two-sided')
        u_pvals[j] = u_p

    # 多重比较校正 (Benjamini-Hochberg FDR)
    reject_t, pvals_t_fdr = fdr_bh(t_pvals, alpha=alpha)
    reject_u, pvals_u_fdr = fdr_bh(u_pvals, alpha=alpha)

    return {
        "mean_good": mean_good,
        "mean_bad": mean_bad,
        "std_good": std_good,
        "std_bad": std_bad,
        "cohens_d": cohens_d,
        "t_stats": t_stats,
        "t_pvals": t_pvals,
        "t_pvals_fdr": pvals_t_fdr,
        "t_significant": reject_t,
        "u_pvals": u_pvals,
        "u_pvals_fdr": pvals_u_fdr,
        "u_significant": reject_u,
        "n_good": n_good,
        "n_bad": n_bad,
    }


def analyze_topk_pattern(acts_good_np, acts_bad_np, k=10):
    """分析 top-k 激活神经元的排名分布差异。

    对每个样本，找出激活值最高的 k 个神经元，统计每个神经元
    出现在 top-k 中的频率。
    """
    n_good, d_model = acts_good_np.shape
    n_bad = acts_bad_np.shape[0]

    # 每个样本的 top-k 神经元
    topk_good = np.argsort(-acts_good_np, axis=1)[:, :k]
    topk_bad = np.argsort(-acts_bad_np, axis=1)[:, :k]

    # 统计每个神经元出现在 top-k 中的频率
    freq_good = np.zeros(d_model)
    freq_bad = np.zeros(d_model)
    for j in range(k):
        idx, counts = np.unique(topk_good[:, j], return_counts=True)
        freq_good[idx] += counts
        idx, counts = np.unique(topk_bad[:, j], return_counts=True)
        freq_bad[idx] += counts

    freq_good /= n_good
    freq_bad /= n_bad

    return {
        "freq_good": freq_good,
        "freq_bad": freq_bad,
        "freq_diff": freq_good - freq_bad,
    }


# ──────────────────────────────────────────────
# 3. 生成 Markdown 报告
# ──────────────────────────────────────────────
def generate_report(stats_result, topk_result, output_path, top_n=30, k=10):
    cohens_d = stats_result["cohens_d"]
    mean_good = stats_result["mean_good"]
    mean_bad = stats_result["mean_bad"]
    t_significant = stats_result["t_significant"]
    u_significant = stats_result["u_significant"]
    t_pvals_fdr = stats_result["t_pvals_fdr"]
    cohens_d_abs = np.abs(cohens_d)

    freq_good = topk_result["freq_good"]
    freq_bad = topk_result["freq_bad"]
    freq_diff = topk_result["freq_diff"]

    lines = []
    lines.append("# Layer 12 神经元激活分布分析报告\n")
    lines.append(f"**好组 (top1 正确) 样本数**: {stats_result['n_good']}")
    lines.append(f"**坏组 (top1 错误) 样本数**: {stats_result['n_bad']}\n")

    # ── 1. 全局统计 ──
    lines.append("## 1. 全局统计\n")
    n_t_sig = t_significant.sum()
    n_u_sig = u_significant.sum()
    n_both = (t_significant & u_significant).sum()
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 总神经元数 | {len(cohens_d)} |")
    lines.append(f"| t-test 显著 (FDR<0.05) | {n_t_sig} ({n_t_sig/len(cohens_d)*100:.1f}%) |")
    lines.append(f"| Mann-Whitney U 显著 (FDR<0.05) | {n_u_sig} ({n_u_sig/len(cohens_d)*100:.1f}%) |")
    lines.append(f"| 两者均显著 | {n_both} ({n_both/len(cohens_d)*100:.1f}%) |")
    lines.append(f"| |Cohen's d| > 0.1 (小效应) | {(cohens_d_abs > 0.1).sum()} |")
    lines.append(f"| |Cohen's d| > 0.2 (小-中效应) | {(cohens_d_abs > 0.2).sum()} |")
    lines.append(f"| |Cohen's d| > 0.5 (中效应) | {(cohens_d_abs > 0.5).sum()} |")
    lines.append(f"| |Cohen's d| > 0.8 (大效应) | {(cohens_d_abs > 0.8).sum()} |")

    # ── 2. 按 Cohen's d 排序: 好组更高 ──
    lines.append(f"\n## 2. 好组激活值更高的 Top-{top_n} 神经元 (按 Cohen's d)\n")
    lines.append("按效应量降序排列，仅展示 t-test FDR 校正后显著的。\n")

    # 只选显著的
    good_mask = (cohens_d > 0) & t_significant
    good_idx = np.where(good_mask)[0]
    good_idx = good_idx[np.argsort(-cohens_d[good_idx])][:top_n]

    lines.append("| 排名 | 神经元ID | Cohen's d | 好组均值 | 坏组均值 | 好组top-k频率 | 坏组top-k频率 | t-test p(FDR) |")
    lines.append("|------|---------|-----------|---------|---------|-------------|-------------|--------------|")
    for rank, nid in enumerate(good_idx):
        lines.append(
            f"| {rank+1} | {nid} | {cohens_d[nid]:.4f} | {mean_good[nid]:.4f} | {mean_bad[nid]:.4f} | "
            f"{freq_good[nid]:.4f} | {freq_bad[nid]:.4f} | {t_pvals_fdr[nid]:.2e} |"
        )

    # ── 3. 按 Cohen's d 排序: 坏组更高 ──
    lines.append(f"\n## 3. 坏组激活值更高的 Top-{top_n} 神经元 (按 Cohen's d)\n")
    lines.append("按效应量升序排列（坏组更高 = 负 Cohen's d），仅展示 t-test FDR 校正后显著的。\n")

    bad_mask = (cohens_d < 0) & t_significant
    bad_idx = np.where(bad_mask)[0]
    bad_idx = bad_idx[np.argsort(cohens_d[bad_idx])][:top_n]

    lines.append("| 排名 | 神经元ID | Cohen's d | 好组均值 | 坏组均值 | 好组top-k频率 | 坏组top-k频率 | t-test p(FDR) |")
    lines.append("|------|---------|-----------|---------|---------|-------------|-------------|--------------|")
    for rank, nid in enumerate(bad_idx):
        lines.append(
            f"| {rank+1} | {nid} | {cohens_d[nid]:.4f} | {mean_good[nid]:.4f} | {mean_bad[nid]:.4f} | "
            f"{freq_good[nid]:.4f} | {freq_bad[nid]:.4f} | {t_pvals_fdr[nid]:.2e} |"
        )

    # ── 4. Top-k 频率差异最大的神经元 ──
    lines.append(f"\n## 4. Top-{k} 激活频率差异最大的神经元\n")
    lines.append(f"对每个样本取激活值最高的 {k} 个神经元，统计每个神经元出现在 top-{k} 中的频率。\n")

    lines.append("### 好组 top-k 频率更高的神经元\n")
    lines.append("| 排名 | 神经元ID | 好组频率 | 坏组频率 | 频率差 | Cohen's d |")
    lines.append("|------|---------|---------|---------|--------|-----------|")
    topk_good_idx = np.argsort(-freq_diff)[:top_n]
    for rank, nid in enumerate(topk_good_idx):
        lines.append(
            f"| {rank+1} | {nid} | {freq_good[nid]:.4f} | {freq_bad[nid]:.4f} | "
            f"{freq_diff[nid]:+.4f} | {cohens_d[nid]:.4f} |"
        )

    lines.append(f"\n### 坏组 top-k 频率更高的神经元\n")
    lines.append("| 排名 | 神经元ID | 好组频率 | 坏组频率 | 频率差 | Cohen's d |")
    lines.append("|------|---------|---------|---------|--------|-----------|")
    topk_bad_idx = np.argsort(freq_diff)[:top_n]
    for rank, nid in enumerate(topk_bad_idx):
        lines.append(
            f"| {rank+1} | {nid} | {freq_good[nid]:.4f} | {freq_bad[nid]:.4f} | "
            f"{freq_diff[nid]:+.4f} | {cohens_d[nid]:.4f} |"
        )

    # ── 5. Cohen's d 分布 ──
    lines.append(f"\n## 5. Cohen's d 分布\n")
    lines.append(f"| 区间 | 神经元数 | 占比 |")
    lines.append(f"|------|---------|------|")
    bins = [(-999, -0.8), (-0.8, -0.5), (-0.5, -0.2), (-0.2, -0.1), (-0.1, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 999)]
    labels = ["d < -0.8 (大)", "-0.8 ~ -0.5 (中)", "-0.5 ~ -0.2 (小-中)", "-0.2 ~ -0.1 (小)",
              "-0.1 ~ 0.1 (无)", "0.1 ~ 0.2 (小)", "0.2 ~ 0.5 (小-中)", "0.5 ~ 0.8 (中)", "d > 0.8 (大)"]
    for (lo, hi), label in zip(bins, labels):
        count = ((cohens_d >= lo) & (cohens_d < hi)).sum()
        lines.append(f"| {label} | {count} | {count/len(cohens_d)*100:.1f}% |")

    # ── 6. 结论 ──
    lines.append(f"\n## 6. 结论\n")
    lines.append(f"- 共 {len(cohens_d)} 个神经元，t-test FDR 校正后 {n_t_sig} 个显著 (p<0.05)。")
    lines.append(f"- |Cohen's d| > 0.1 的神经元有 {(cohens_d_abs > 0.1).sum()} 个，> 0.2 的有 {(cohens_d_abs > 0.2).sum()} 个。")
    lines.append(f"- 大多数效应量很小 (|d|<0.1)，说明两组激活值差异在单神经元层面较微弱。")
    if n_t_sig > 0:
        lines.append(f"- 好组更高 (d>0) 的显著神经元: {(cohens_d > 0).sum()} 个。")
        lines.append(f"- 坏组更高 (d<0) 的显著神经元: {(cohens_d < 0).sum()} 个。")
    lines.append(f"- Top-k 频率分析可能比均值比较更有区分度，因为它关注的是 '哪些神经元被选中' 而非 '激活值多高'。")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")
    return report


# ──────────────────────────────────────────────
# 4. 主函数
# ──────────────────────────────────────────────
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"

    checkpoint_path = "out/dsi-semantic-bert/99.pt"
    dev_data_path = "dataset/nq320k/dev.json"
    semantic_id_path = "dataset/nq320k_id/id.semantic.bert.json"
    comparison_dir = "dataset/nq320k/comparison"
    cache_path = "data/activation_cache_dev/layer_12.npz"
    layer = 12
    device = "cuda"

    # ── 加载数据 ──
    print("Loading data...")
    dev_data = json.load(open(dev_data_path))
    semantic_ids = json.load(open(semantic_id_path))

    top1_correct = json.load(open(f"{comparison_dir}/top1_correct.json"))
    top1_wrong = json.load(open(f"{comparison_dir}/top1_wrong.json"))
    print(f"  top1_correct: {len(top1_correct)}, top1_wrong: {len(top1_wrong)}")

    dev_index_map = {}
    for i, (q, d) in enumerate(dev_data):
        if isinstance(q, list): q = q[0]
        if isinstance(d, list): d = d[0]
        dev_index_map[(q, d)] = i

    correct_indices = set(dev_index_map[(item['query'], item['doc_id'])] for item in top1_correct)
    wrong_indices = set(dev_index_map[(item['query'], item['doc_id'])] for item in top1_wrong)
    print(f"  Mapped: correct={len(correct_indices)}, wrong={len(wrong_indices)}")

    # ── 收集激活值 ──
    all_acts, sample_indices = collect_dev_activations(
        checkpoint_path, dev_data, semantic_ids, cache_path,
        layer=layer, device=device,
    )
    print(f"  Total activations: {all_acts.shape}")

    # ── 按样本聚合 ──
    n_samples = len(dev_data)
    print("Aggregating per-sample activations (mean over tokens)...")
    sample_acts = aggregate_per_sample(all_acts, sample_indices, n_samples)

    # 分组
    correct_idx_list = sorted(correct_indices)
    wrong_idx_list = sorted(wrong_indices)
    acts_good = sample_acts[correct_idx_list].numpy()
    acts_bad = sample_acts[wrong_idx_list].numpy()
    print(f"  Good samples: {acts_good.shape}, Bad samples: {acts_bad.shape}")

    # ── 统计检验 ──
    print("\nRunning statistical tests per neuron...")
    stats_result = analyze_statistical(acts_good, acts_bad)

    n_t_sig = stats_result["t_significant"].sum()
    n_u_sig = stats_result["u_significant"].sum()
    print(f"  t-test significant (FDR<0.05): {n_t_sig}/{len(stats_result['cohens_d'])}")
    print(f"  MW-U significant (FDR<0.05): {n_u_sig}/{len(stats_result['cohens_d'])}")

    # ── Top-k 频率分析 ──
    print("\nRunning top-k activation frequency analysis...")
    topk_result = analyze_topk_pattern(acts_good, acts_bad, k=10)

    # ── 打印摘要 ──
    cohens_d = stats_result["cohens_d"]
    t_sig = stats_result["t_significant"]

    good_sig_idx = np.where((cohens_d > 0) & t_sig)[0]
    good_sig_idx = good_sig_idx[np.argsort(-cohens_d[good_sig_idx])][:10]
    bad_sig_idx = np.where((cohens_d < 0) & t_sig)[0]
    bad_sig_idx = bad_sig_idx[np.argsort(cohens_d[bad_sig_idx])][:10]

    print(f"\n{'='*70}")
    print(f"好组更高的 Top-10 (t-test 显著):")
    for nid in good_sig_idx:
        print(f"  neuron {nid:4d}: d={cohens_d[nid]:.4f}, mean_good={stats_result['mean_good'][nid]:.4f}, mean_bad={stats_result['mean_bad'][nid]:.4f}")
    print(f"坏组更高的 Top-10 (t-test 显著):")
    for nid in bad_sig_idx:
        print(f"  neuron {nid:4d}: d={cohens_d[nid]:.4f}, mean_good={stats_result['mean_good'][nid]:.4f}, mean_bad={stats_result['mean_bad'][nid]:.4f}")
    print(f"{'='*70}")

    # ── 生成报告 ──
    report_path = "sae/neuron_analysis_report.md"
    generate_report(stats_result, topk_result, report_path)

    # ── 保存原始数据 ──
    np.savez(
        "sae/neuron_analysis_data.npz",
        mean_good=stats_result["mean_good"],
        mean_bad=stats_result["mean_bad"],
        cohens_d=stats_result["cohens_d"],
        t_pvals_fdr=stats_result["t_pvals_fdr"],
        t_significant=stats_result["t_significant"],
        freq_good=topk_result["freq_good"],
        freq_bad=topk_result["freq_bad"],
    )
    print("Raw data saved to sae/neuron_analysis_data.npz")


if __name__ == "__main__":
    main()
