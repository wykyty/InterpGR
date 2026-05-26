"""
Generate hierarchical semantic DocIDs (DSI algorithm).

Algorithm (k=10, c=100):
  1. Encode all docs with all-MiniLM-L6-v2 on GPU
  2. 10-way k-means clustering
  3. Recurse on clusters with >100 docs
  4. Sequential IDs for clusters <=100 docs
"""

import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

# ── Config ──
CORPUS_PATH = "dataset/nq320k/corpus_lite.json"
OUTPUT_PATH = "dataset/nq320k_id/id.semantic.json"
K = 10
C = 100


def encode(corpus):
    """Encode with all-MiniLM-L6-v2 on GPU."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    return model.encode(corpus, batch_size=512,
                        show_progress_bar=True, normalize_embeddings=True).astype(np.float32)


def generate_ids(embeddings, k=K, c=C):
    """
    Recursively generate hierarchical semantic IDs.

    Returns: List[List[int]], variable-length per doc.
    """
    n = len(embeddings)
    if n <= c:
        return [[i] for i in range(n)]

    k_eff = min(k, n)
    kmeans = MiniBatchKMeans(n_clusters=k_eff, batch_size=min(4096, n),
                             n_init=3, max_iter=100, random_state=None)
    labels = kmeans.fit_predict(embeddings)

    clusters = [[] for _ in range(k_eff)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    ids = [None] * n
    for cluster_id, indices in enumerate(clusters):
        if not indices:
            continue
        sub_emb = embeddings[indices]
        if len(indices) <= c:
            for seq, global_idx in enumerate(indices):
                ids[global_idx] = [cluster_id, seq]
        else:
            sub_ids = generate_ids(sub_emb, k, c)
            for i, global_idx in enumerate(indices):
                ids[global_idx] = [cluster_id] + sub_ids[i]
    return ids


def main():
    corpus = json.load(open(CORPUS_PATH))
    print(f"Corpus: {len(corpus)} docs")

    embeddings = encode(corpus)
    print(f"Embeddings: {embeddings.shape}  ({embeddings.nbytes / 1024**2:.0f} MB)")

    ids = generate_ids(embeddings)

    depths = [len(x) for x in ids]
    unique = len(set(tuple(x) for x in ids))
    print(f"IDs: {len(ids)} docs, {unique} unique "
          f"(collision {1 - unique / len(ids):.4%})")
    print(f"Depth: {min(depths)} ~ {max(depths)}, "
          f"dist={dict(sorted({d: depths.count(d) for d in set(depths)}.items()))}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(ids, f)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
