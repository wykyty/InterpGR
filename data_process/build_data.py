"""
基于 test_semantic3 的逻辑，用 t5-large 生成式检索模型对 dev 数据集进行推理，
收集每个样本的 Hit@1/10/100 和 MRR@100，然后构造对比数据。

两种分割方式：
  方式1: 按 top-1 是否命中正确 docid，分为正确 / 错误两类
  方式2: 按 MRR@100 的值，找一个阈值分为高 / 低两类
目标：两类数据量尽量平衡。
"""

import json
import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline import Tree, NewNQDataset


def build_comparison_data():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    batch_size = 2
    save_path = 'out/dsi-semantic-bert'
    num_of_new_tokens = 30
    top_k = 100

    # ── 1. 加载模型 ──
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')
    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    model.eval()

    # ── 2. 加载数据 ──
    raw_dev_data = json.load(open('dataset/nq320k/dev.json'))

    semantic_ids = json.load(open('dataset/nq320k_id/id.semantic.bert.json'))
    corpus_strs = [''.join([f'${i}$' for i in z]) for z in semantic_ids]

    # Token ID tuple → doc_idx 的映射
    tuple_to_docs = defaultdict(list)
    doc_to_tuple = {}
    for doc_idx, z in enumerate(semantic_ids):
        token_ids = []
        for token_val in z:
            t_id = tokenizer.convert_tokens_to_ids(f'${token_val}$')
            token_ids.append(t_id)
        token_tuple = tuple(token_ids)
        tuple_to_docs[token_tuple].append(doc_idx)
        doc_to_tuple[doc_idx] = token_tuple

    # 构建受限解码树
    corpus_token_ids = [[0] + list(doc_to_tuple[idx]) + [1] for idx in range(len(corpus_strs))]
    tree = Tree()
    tree.set_all(corpus_token_ids)

    # ── 3. DataLoader ──
    dataset = NewNQDataset(data=raw_dev_data, corpus=corpus_strs, tokenizer=tokenizer, max_len=32)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
        shuffle=False, num_workers=4,
    )

    # ── 4. 加载 checkpoint ──
    ckpt_path = f'{save_path}/99.pt'
    print(f'Loading checkpoint {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))

    # ── 5. 推理 + 收集指标 ──
    # 语义 ID tuple → 字符串 的缓存
    tuple_to_str_cache = {}
    def tuple_to_id_str(t):
        if t not in tuple_to_str_cache:
            tuple_to_str_cache[t] = ''.join(
                [tokenizer.decode([tid], skip_special_tokens=False).replace('$', '').strip() for tid in t]
            )
        return tuple_to_str_cache[t]

    results = []
    data_ptr = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            batch_size_actual = batch['input_ids'].size(0)
            batch = {k: v.cuda() for k, v in batch.items() if v is not None}

            output = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=15,
                num_beams=top_k,
                num_return_sequences=top_k,
                prefix_allowed_tokens_fn=tree,
            )

            # 解码并分组
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            decoded = [x.replace('$', '').strip() for x in decoded]
            preds_str = [decoded[i:i + top_k] for i in range(0, len(decoded), top_k)]

            # 提取 token tuple
            output_list = output.cpu().tolist()
            cleaned_preds = []
            for seq in output_list:
                valid_tokens = [t for t in seq if t not in [0, 1, 2]]
                cleaned_preds.append(tuple(valid_tokens))
            preds_tuple = [cleaned_preds[i:i + top_k] for i in range(0, len(cleaned_preds), top_k)]

            for b_idx in range(batch_size_actual):
                pred_tuples = preds_tuple[b_idx]
                pred_strs = preds_str[b_idx]

                _, true_doc_id = raw_dev_data[data_ptr]
                while isinstance(true_doc_id, list):
                    true_doc_id = true_doc_id[0]

                true_token_tuple = doc_to_tuple.get(true_doc_id)

                if true_token_tuple is None:
                    results.append({
                        'query': raw_dev_data[data_ptr][0],
                        'doc_id': true_doc_id,
                        'top1_pred': pred_strs[0],
                        'top10_preds': pred_strs[:10],
                        'hit1': 0, 'hit10': 0, 'hit100': 0, 'mrr100': 0.0,
                    })
                    data_ptr += 1
                    continue

                # Hit@K
                h1 = int(pred_tuples[0] == true_token_tuple)
                h10 = int(true_token_tuple in pred_tuples[:10])
                h100 = int(true_token_tuple in pred_tuples[:100])

                # MRR@100
                rr = 0.0
                if true_token_tuple in pred_tuples:
                    rank = pred_tuples.index(true_token_tuple) + 1
                    rr = 1.0 / rank

                results.append({
                    'query': raw_dev_data[data_ptr][0],
                    'doc_id': true_doc_id,
                    'top1_pred': pred_strs[0],
                    'top10_preds': pred_strs[:10],
                    'hit1': h1, 'hit10': h10, 'hit100': h100, 'mrr100': rr,
                })
                data_ptr += 1

    # ── 6. 打印总体指标 ──
    n = len(results)
    hit1 = sum(r['hit1'] for r in results) / n
    hit10 = sum(r['hit10'] for r in results) / n
    hit100 = sum(r['hit100'] for r in results) / n
    mrr = sum(r['mrr100'] for r in results) / n

    print(f"\n{'='*50}")
    print(f"Total samples: {n}")
    print(f"Hits@1   : {hit1:.4f}")
    print(f"Hits@10  : {hit10:.4f}")
    print(f"Hits@100 : {hit100:.4f}")
    print(f"MRR@100  : {mrr:.4f}")
    print(f"{'='*50}\n")

    # ── 7. 方式1: 按 top-1 是否正确分割 ──
    top1_correct = [r for r in results if r['hit1'] == 1]
    top1_wrong = [r for r in results if r['hit1'] == 0]

    min_count = min(len(top1_correct), len(top1_wrong))
    print(f"[方式1] top-1 正确: {len(top1_correct)}, top-1 错误: {len(top1_wrong)}")
    print(f"  各取 {min_count} 条以保证平衡")

    top1_correct_data = top1_correct[:min_count]
    top1_wrong_data = top1_wrong[:min_count]

    os.makedirs('dataset/nq320k/comparison', exist_ok=True)
    json.dump(top1_correct_data, open('dataset/nq320k/comparison/top1_correct.json', 'w'), ensure_ascii=False)
    json.dump(top1_wrong_data, open('dataset/nq320k/comparison/top1_wrong.json', 'w'), ensure_ascii=False)
    print(f"  -> 已保存 dataset/nq320k/comparison/top1_correct.json ({len(top1_correct_data)} 条)")
    print(f"  -> 已保存 dataset/nq320k/comparison/top1_wrong.json   ({len(top1_wrong_data)} 条)")

    # ── 8. 方式2: 按 MRR@100 阈值分割 ──
    mrr_values = sorted([r['mrr100'] for r in results])
    median_mrr = mrr_values[len(mrr_values) // 2]

    # 在 median 附近搜索最平衡的阈值
    unique_mrrs = sorted(set(mrr_values))
    best_threshold = median_mrr
    best_diff = float('inf')
    for t in unique_mrrs:
        high = sum(1 for v in mrr_values if v >= t)
        low = sum(1 for v in mrr_values if v < t)
        diff = abs(high - low)
        if diff < best_diff:
            best_diff = diff
            best_threshold = t

    mrr_high = [r for r in results if r['mrr100'] >= best_threshold]
    mrr_low = [r for r in results if r['mrr100'] < best_threshold]

    min_count2 = min(len(mrr_high), len(mrr_low))
    print(f"\n[方式2] MRR@100 阈值: {best_threshold:.4f}")
    print(f"  高MRR(≥阈值): {len(mrr_high)}, 低MRR(<阈值): {len(mrr_low)}")
    print(f"  各取 {min_count2} 条以保证平衡")

    mrr_high_data = mrr_high[:min_count2]
    mrr_low_data = mrr_low[:min_count2]

    json.dump(mrr_high_data, open('dataset/nq320k/comparison/mrr_high.json', 'w'), ensure_ascii=False)
    json.dump(mrr_low_data, open('dataset/nq320k/comparison/mrr_low.json', 'w'), ensure_ascii=False)
    print(f"  -> 已保存 dataset/nq320k/comparison/mrr_high.json ({len(mrr_high_data)} 条)")
    print(f"  -> 已保存 dataset/nq320k/comparison/mrr_low.json  ({len(mrr_low_data)} 条)")

    # ── 9. 汇总 ──
    print(f"\n{'='*50}")
    print("对比数据构造完成！")
    print(f"方式1: top1_correct={len(top1_correct_data)}, top1_wrong={len(top1_wrong_data)}")
    print(f"方式2: mrr_high={len(mrr_high_data)}, mrr_low={len(mrr_low_data)} (阈值={best_threshold:.4f})")
    print(f"{'='*50}")


if __name__ == '__main__':
    build_comparison_data()
