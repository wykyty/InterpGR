#!/bin/bash
cd "$(dirname "$0")"
mkdir -p log
python -u generate_semantic_id_bert.py > log/nq320k_bert_k30_c30.log 2>&1 &
echo "PID: $!, log: log/nq320k_bert_k30_c30.log"
