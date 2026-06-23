"""
Latent Top-10 激活 Query-Doc 对的 LLM 解释分析

对于 docid_position_activation_report 中 top-20 高 KL 散度的 latent，
找出每个 latent 激活值最高的 top-10 个 query-doc 对，
用大模型分析这些对是否有共同点，解释该 latent 编码的语义概念。

用法:
    cd /home/zyq/wyk/InterpGR
    uv run python analysis/analysis_latent_topk_explanation.py
"""

import json
import os
import sys
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


def call_llm(prompt: str, model: str = "mimo-v2.5-pro") -> str:
    """调用 Anthropic 兼容 API"""
    import anthropic
    client = anthropic.Anthropic(
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
    )
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    # 兼容 ThinkingBlock：优先取 TextBlock，否则取 ThinkingBlock
    for block in resp.content:
        if hasattr(block, "text"):
            return block.text
    for block in resp.content:
        if hasattr(block, "thinking"):
            return block.thinking
    return str(resp.content[0])


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # ── 配置 ──
    sae_dir = "out/sae_train_8x/layer_12"
    cache_path = "data/activation_cache_dev/layer_12.npz"
    dev_data_path = "dataset/nq320k/dev.json"
    semantic_id_path = "dataset/nq320k_id/id.semantic.bert.json"
    corpus_path = "dataset/nq320k/corpus_lite.json"
    npz_data_path = "analysis/docid_position_activation_data.npz"
    report_path = "analysis/latent_topk_explanation_report.md"
    device = "cuda"

    positions = [0, 1, 2, 3]
    top_k_latents = 20
    top_k_samples = 10
    doc_preview_len = 300  # document 预览截断字符数

    # ── 1. 加载数据 ──
    print("Loading data...")
    dev_data = json.load(open(dev_data_path))
    semantic_ids = json.load(open(semantic_id_path))
    corpus = json.load(open(corpus_path))
    n_samples = len(dev_data)
    print(f"  Samples: {n_samples}, Corpus size: {len(corpus)}")

    # ── 2. 从 npz 读取 top-20 latent ID ──
    print("Loading KL divergence data...")
    npz = np.load(npz_data_path)
    max_kl = npz["max_kl"]          # (8192,)
    best_pos = npz["best_pos"]      # (8192,)
    kl_scores = npz["kl_scores"]    # (4, 8192)

    # 按 max_kl 降序取 top-20
    top_latent_ids = np.argsort(-max_kl)[:top_k_latents]
    print(f"  Top-{top_k_latents} latent IDs: {top_latent_ids.tolist()}")
    print(f"  KL divergences: {[f'{max_kl[fid]:.4f}' for fid in top_latent_ids]}")

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

    # ── 5. 对每个 latent，找 top-10 激活样本并调用 LLM ──
    print(f"\nAnalyzing top-{top_k_latents} latents, top-{top_k_samples} samples each...")

    results = []  # (latent_id, kl, best_pos, samples, explanation)

    for rank, fid in enumerate(tqdm(top_latent_ids, desc="Latents")):
        fid = int(fid)
        acts = sample_features[:, fid]  # (n_samples,)
        top_indices = np.argsort(-acts)[:top_k_samples]

        # 收集 query-doc 对
        samples = []
        for idx in top_indices:
            idx = int(idx)
            query, doc_id = dev_data[idx]
            if isinstance(doc_id, list):
                doc_id = doc_id[0]
            doc_text = corpus[doc_id] if doc_id < len(corpus) else "[N/A]"
            sid = semantic_ids[doc_id] if doc_id < len(semantic_ids) else []
            samples.append({
                "sample_idx": idx,
                "query": query,
                "doc_id": doc_id,
                "doc_text": doc_text,
                "semantic_id": sid,
                "activation": float(acts[idx]),
            })

        # 构建 prompt
        table_rows = []
        for i, s in enumerate(samples):
            sid_str = str(s["semantic_id"])
            table_rows.append(
                f"| {i+1} | {s['query'][:80]} | {s['doc_id']} | {sid_str[:30]} | {s['activation']:.4f} |"
            )
        table_str = "\n".join(table_rows)

        doc_previews = []
        for i, s in enumerate(samples):
            preview = s["doc_text"][:doc_preview_len].replace("\n", " ")
            doc_previews.append(f"[{i+1}] (doc_id={s['doc_id']}) {preview}")
        docs_str = "\n".join(doc_previews)

        bp = int(best_pos[fid])
        kl_val = float(max_kl[fid])
        kl_detail = ", ".join(f"pos{p}={kl_scores[p][fid]:.4f}" for p in positions)

        prompt = f"""以下是 SAE (Sparse Autoencoder) latent {fid} 激活值最高的 {top_k_samples} 个 query-document 对。
该 latent 的 KL 散度为 {kl_val:.4f}，最佳 DocID position 为 {bp}。各位置 KL: {kl_detail}

| # | Query | DocID | Semantic ID | 激活值 |
|---|-------|-------|-------------|--------|
{table_str}

Document 内容摘要：
{docs_str}

请分析这 {top_k_samples} 个 query-document 对的共同点：
1. 这些 query 有什么共同的语义模式或主题？
2. 这些 document 有什么共同特征？
3. 你认为这个 latent 可能在编码什么语义概念或特征？
4. 这能否解释为什么该 latent 对 DocID 的特定 position 值有偏好？

请用中文回答，简洁但具体。"""

        try:
            explanation = call_llm(prompt)
        except Exception as e:
            explanation = f"[LLM 调用失败: {e}]"
            print(f"  Warning: LLM call failed for latent {fid}: {e}")

        results.append((fid, kl_val, bp, kl_scores[:, fid].tolist(), samples, explanation))
        print(f"  Latent {fid} done (KL={kl_val:.4f})")

    # ── 6. 生成报告 ──
    print("\nGenerating report...")
    lines = []
    lines.append("# Latent Top-10 激活 Query-Doc 对 LLM 解释报告\n")
    lines.append("## 方法\n")
    lines.append("对于 `docid_position_activation_report` 中 top-20 高 KL 散度的 latent，")
    lines.append(f"找出每个 latent 激活值最高的 top-{top_k_samples} 个 query-doc 对，")
    lines.append("用大模型分析这些对的共同点，解释该 latent 编码的语义概念。\n")
    lines.append(f"- SAE 特征维度: {d_sae}")
    lines.append(f"- 样本数: {n_samples}")
    lines.append(f"- 分析 latent 数: {top_k_latents}")
    lines.append(f"- 每个 latent 取 top-{top_k_samples} 激活样本\n")

    lines.append("---\n")

    for rank, (fid, kl_val, bp, kl_list, samples, explanation) in enumerate(results):
        lines.append(f"## {rank+1}. Latent {fid}\n")
        lines.append(f"- **KL 散度**: {kl_val:.4f}")
        lines.append(f"- **最佳 DocID Position**: {bp}")
        kl_str = ", ".join(f"pos{i}={v:.4f}" for i, v in enumerate(kl_list))
        lines.append(f"- **各位置 KL**: {kl_str}\n")

        lines.append("### Top-10 激活 Query-Doc 对\n")
        lines.append("| # | Query | DocID | Semantic ID | 激活值 |")
        lines.append("|---|-------|-------|-------------|--------|")
        for i, s in enumerate(samples):
            sid_str = str(s["semantic_id"])
            # 截断 query 避免表格过宽
            q_short = s["query"][:60].replace("|", "/")
            lines.append(f"| {i+1} | {q_short} | {s['doc_id']} | {sid_str[:25]} | {s['activation']:.4f} |")

        lines.append("\n<details><summary>Document 内容摘要</summary>\n")
        for i, s in enumerate(samples):
            preview = s["doc_text"][:doc_preview_len].replace("\n", " ")
            lines.append(f"**[{i+1}] doc_id={s['doc_id']}** (query: {s['query'][:80]})")
            lines.append(f"> {preview}\n")
        lines.append("</details>\n")

        lines.append("### LLM 解释\n")
        lines.append(explanation)
        lines.append("\n---\n")

    report = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
