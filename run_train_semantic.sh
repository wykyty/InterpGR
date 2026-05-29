#!/bin/bash
cd "$(dirname "$0")"
mkdir -p log
uv run torchrun --nproc_per_node=8 baseline.py > log/train_semantic_bert_2.log 2>&1 &
echo "PID: $!, log: log/train_semantic_bert_2.log"
