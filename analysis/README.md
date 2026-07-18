# InterpGR Layer Analysis

The three experiments share SAE loading, layer discovery, latent sampling, and
trend helpers from `analysis/common.py`. Run every command from the repository
root.

## Experiment 1: Layer Activation Statistics

```bash
uv run python analysis/experiment1_layer_activation.py \
  --cache-dir data/activation_cache \
  --checkpoint-root out/sae_train_8x \
  --output-dir results/layer_activation_statistics
```

Reports average active latents, average active magnitude, activation sparsity,
and activation variance for every decoder layer.

## Experiment 2: Layer Document Purity

```bash
uv run python analysis/experiment2_document_purity.py \
  --cache-dir data/activation_cache \
  --checkpoint-root out/sae_train_8x \
  --output-dir results/layer_semantic_purity
```

Aggregates every DocID-position activation by document and reports document
entropy, BERT semantic-DocID cluster purity, and independent topic purity.

## Experiment 3: Layer Query Purity

```bash
uv run python analysis/experiment3_query_purity.py \
  --cache-dir data/activation_cache \
  --checkpoint-root out/sae_train_8x \
  --output-dir results/layer_query_purity
```

Aggregates every DocID-position activation by query. It records top activated
queries and computes activation-weighted semantic similarity in the fine-tuned
DSI T5 encoder space, plus independent query topic purity. Query embeddings are
cached under `data/analysis_cache/`.

Experiments 2 and 3 default to a fixed-seed sample of 64 latents from the 512
latents with the largest total activation mass in each layer. Each sampled
latent uses its top 20 documents or queries for purity measurements.

## Experiment 4: Layer-wise DocID Information

```bash
uv run python analysis/experiment4_docid_information.py \
  --cache-dir data/activation_cache \
  --checkpoint-root out/sae_train_8x \
  --output-dir results/layer_docid_information
```

Computes binary-latent MI with current DocID tokens, prefixes, semantic clusters,
eligible full DocIDs, and query-topic controls at every layer and generation
position. It also fits a query-grouped L1 linear probe for current-token
prediction and writes MI distributions plus the four layer/position figures.
