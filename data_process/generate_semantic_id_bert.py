"""
Generate hierarchical semantic DocIDs using BERT (bert-base-uncased).
Reproduces NCI's bert+kmeans pipeline with k=30, c=30.
"""

import json
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from transformers import BertTokenizer, BertModel

# ── Config ──
CORPUS_PATH = "../dataset/nq320k/corpus_lite.json"
OUTPUT_PATH = "../dataset/nq320k_id/id.semantic.bert.json"
K = 30
C = 30
BATCH_SIZE = 64
MAX_LEN = 512

# KMeans: use KMeans for small clusters, MiniBatchKMeans for large (>=1000)
kmeans = KMeans(n_clusters=K, max_iter=300, n_init=100,
                init='k-means++', random_state=7, tol=1e-7)
mini_kmeans = MiniBatchKMeans(n_clusters=K, max_iter=300, n_init=100,
                              init='k-means++', random_state=3,
                              batch_size=1000, reassignment_ratio=0.01,
                              max_no_improvement=20, tol=1e-7)


def encode(corpus):
    """Encode with BERT CLS token on GPU."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').eval().cuda()

    all_embs = []
    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="BERT encoding"):
        batch = corpus[i:i + BATCH_SIZE]
        encoded = tokenizer(batch, max_length=MAX_LEN, padding=True,
                            truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            cls_emb = model(**encoded).last_hidden_state[:, 0, :]  # CLS token
        all_embs.append(cls_emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0).astype(np.float32)


def classify_recursion(x_data_pos, X, new_id_list):
    """
    Recursively cluster and append sub-cluster IDs to each doc's ID list.
    Matches NCI's classify_recursion logic.
    """
    if x_data_pos.shape[0] <= C:
        if x_data_pos.shape[0] == 1:
            return
        for idx, pos in enumerate(x_data_pos):
            new_id_list[pos].append(idx)
        return

    # Extract embeddings for this cluster
    temp_data = X[x_data_pos]

    # Use KMeans for small, MiniBatchKMeans for large (>=1000)
    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(temp_data)
    else:
        pred = kmeans.fit_predict(temp_data)

    for i in range(K):
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(x_data_pos[id_])
                new_id_list[x_data_pos[id_]].append(i)
        classify_recursion(np.array(pos_lists), X, new_id_list)


def generate_ids(X):
    """
    Generate hierarchical semantic IDs following NCI's logic.
    Returns: List[List[int]], variable-length per doc.
    """
    n = len(X)
    new_id_list = [[] for _ in range(n)]

    # First level clustering
    print('Start First Clustering')
    pred = mini_kmeans.fit_predict(X)
    print(f"First level: {pred.shape}, iter={mini_kmeans.n_iter_}")

    # NCI: for class_ in pred: new_id_list.append([class_])
    for doc_idx, class_ in enumerate(pred):
        new_id_list[doc_idx].append(class_)

    # Recursively refine each top-level cluster
    print('Start Recursively Clustering...')
    for i in range(K):
        print(f"  Cluster {i}/{K}")
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(id_)
        classify_recursion(np.array(pos_lists), X, new_id_list)

    return new_id_list


def main():
    corpus = json.load(open(CORPUS_PATH))
    print(f"Corpus: {len(corpus)} docs")

    embeddings = encode(corpus)
    print(f"Embeddings: {embeddings.shape}  ({embeddings.nbytes / 1024**2:.0f} MB)")

    ids = [[int(x) for x in doc_id] for doc_id in generate_ids(embeddings)]

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
