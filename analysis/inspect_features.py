"""
从 SAE 高频特征中随机抽取 20-30 个，展示每个特征的 top-10 激活样本，
方便人工标注特征类型（语义/路由/其他）。

用法:
    cd /home/zyq/wyk/InterpGR
    python sae/inspect_features.py 2>&1 | tee sae/inspect_features.log
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"

    sae_dir = "out/sae_train_8x/layer_12"
    cache_path = "data/activation_cache_dev/layer_12.npz"
    dev_data_path = "dataset/nq320k/dev.json"
    semantic_id_path = "dataset/nq320k_id/id.semantic.bert.json"
    n_features = 25  # 抽取特征数
    top_n_samples = 10  # 每个特征展示的样本数
    seed = 42

    # ── 1. 加载数据 ──
    print("Loading data...")
    dev_data = json.load(open(dev_data_path))
    semantic_ids = json.load(open(semantic_id_path))

    # 加载缓存激活值
    data = np.load(cache_path)
    all_acts = torch.from_numpy(data["activations"])
    sample_indices = torch.from_numpy(data["sample_indices"])
    print(f"  Activations: {all_acts.shape}")

    # ── 2. 加载 SAE ──
    print(f"Loading SAE from {sae_dir}...")
    from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
    from safetensors.torch import load_file

    cfg = json.load(open(f"{sae_dir}/sae_config.json"))
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=int(cfg["k"]),
        aux_loss_coefficient=1.0, rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.1, apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in", decoder_init_norm=0.01,
        device="cuda", dtype="float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg).cuda()
    weights = load_file(f"{sae_dir}/sae_weights.safetensors")
    sae.W_enc.data = weights["W_enc"].cuda()
    sae.W_dec.data = weights["W_dec"].cuda()
    sae.b_enc.data = weights["b_enc"].cuda()
    sae.b_dec.data = weights["b_dec"].cuda()
    sae.topk_threshold = weights["topk_threshold"].cuda()
    sae.eval()
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, k={sae.cfg.k}")

    # ── 3. SAE 编码 ──
    print("Encoding through SAE...")
    batch_size = 4096
    all_features = []
    for i in tqdm(range(0, len(all_acts), batch_size), desc="SAE encode"):
        batch = all_acts[i:i+batch_size].cuda()
        with torch.no_grad():
            features, _ = sae.encode_with_hidden_pre(batch)
        all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0).numpy()
    print(f"  Features shape: {all_features.shape}")

    # ── 4. 聚合: 每个样本的特征取 mean ──
    n_samples = len(dev_data)
    d_sae = all_features.shape[1]
    print(f"Aggregating per-sample ({n_samples} samples)...")
    sample_features = np.zeros((n_samples, d_sae))
    sample_counts = np.zeros(n_samples)
    for i in range(len(all_features)):
        sid = sample_indices[i].item()
        sample_features[sid] += all_features[i]
        sample_counts[sid] += 1
    sample_counts = np.maximum(sample_counts, 1)
    sample_features = sample_features / sample_counts[:, None]

    # ── 5. 找高频特征 ──
    # 高频 = 在所有样本中被激活（>0）的比例最高的特征
    activation_rate = (sample_features > 0).mean(axis=0)
    # 排除太低频的（<5%），从剩下的中随机抽
    high_freq_mask = activation_rate > 0.05
    high_freq_indices = np.where(high_freq_mask)[0]
    high_freq_sorted = high_freq_indices[np.argsort(-activation_rate[high_freq_indices])]
    print(f"  Features with >5% activation rate: {len(high_freq_indices)}")

    np.random.seed(seed)
    selected = np.random.choice(high_freq_sorted, size=min(n_features, len(high_freq_sorted)), replace=False)
    selected = sorted(selected)
    print(f"  Selected {len(selected)} features for inspection")

    # ── 6. 准备 DocID 信息 ──
    # 语义 ID: doc_id -> [token_ids]
    # 我们需要知道每个样本模型实际生成的 top-1 DocID
    # 这里用 semantic_ids[doc_id] 来展示真实 DocID 路径

    def get_docid_str(doc_id):
        """获取 doc_id 的语义 ID 字符串表示。"""
        if isinstance(doc_id, list):
            doc_id = doc_id[0]
        sid = semantic_ids[doc_id]
        return f"[{', '.join(str(x) for x in sid)}]"

    # ── 7. 生成 Markdown ──
    lines = []
    lines.append("# SAE 特征人工检查表\n")
    lines.append(f"**SAE**: {sae_dir} (d_sae={d_sae}, k={sae.cfg.k})")
    lines.append(f"**数据**: {dev_data_path} ({n_samples} 样本)")
    lines.append(f"**抽取方式**: 从激活频率 >5% 的特征中随机抽 {len(selected)} 个 (seed={seed})\n")
    lines.append("**标注指南**: 对每个特征，查看 top-10 激活样本的 query 和 DocID，判断特征类型：\n")
    lines.append("- **语义特征**: top 样本的主题/领域一致（如都关于体育、地理、电影等）")
    lines.append("- **路由特征**: top 样本的 DocID 前缀/路径一致（如都映射到某个 doc_id range）")
    lines.append("- **其他**: 无明显规律\n")
    lines.append("---\n")

    for feat_id in selected:
        feat_rate = activation_rate[feat_id]
        feat_vals = sample_features[:, feat_id]

        # 找 top-N 激活的样本
        top_indices = np.argsort(-feat_vals)[:top_n_samples]

        lines.append(f"## Feature {feat_id}\n")
        lines.append(f"**激活频率**: {feat_rate*100:.1f}%（{int(feat_rate*n_samples)}/{n_samples} 个样本激活）")
        lines.append(f"**最大激活值**: {feat_vals.max():.4f}, **平均激活值(激活时)**: {feat_vals[feat_vals>0].mean():.4f}\n")
        lines.append(f"**标注**: 语义特征 / 路由特征 / 其他 （请标注）\n")

        lines.append(f"| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |")
        lines.append(f"|------|--------|--------|-------|---------------|")
        for rank, sid in enumerate(top_indices):
            query = dev_data[sid][0]
            if isinstance(query, list):
                query = query[0]
            doc_id = dev_data[sid][1]
            if isinstance(doc_id, list):
                doc_id = doc_id[0]
            docid_str = get_docid_str(doc_id)
            val = feat_vals[sid]
            # 截断过长的 query
            if len(query) > 80:
                query = query[:77] + "..."
            lines.append(f"| {rank+1} | {sid} | {val:.4f} | {query} | {docid_str} |")

        lines.append("")

    # 统计摘要
    lines.append("---\n")
    lines.append("## 汇总\n")
    lines.append(f"| 特征ID | 激活频率 | 最大激活值 | 特征类型(请标注) |")
    lines.append(f"|--------|---------|-----------|----------------|")
    for feat_id in selected:
        feat_rate = activation_rate[feat_id]
        feat_max = sample_features[:, feat_id].max()
        lines.append(f"| {feat_id} | {feat_rate*100:.1f}% | {feat_max:.4f} | |")

    report = "\n".join(lines)
    output_path = "sae/feature_inspection.md"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
