#!/bin/bash

set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Error: python/python3 not found in PATH"
  exit 127
fi

# EMAformer (ETTh1)
# 预测窗口: 24, 48, 72, 96
# 使用方法:
#   bash scripts/multivariate_forecasting/ETT/EMAformer_ETTh1_step24.sh

model_name=EMAformer
seq_len=96
start_pred=24
end_pred=96
step_pred=24

for pred_len in $(seq "$start_pred" "$step_pred" "$end_pred"); do
  "$PYTHON_BIN" -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_${pred_len} \
    --model "$model_name" \
    --data ETTh1 \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp_step24' \
    --d_model 256 \
    --n_heads 4 \
    --d_ff 256 \
    --output_proj_dropout 0.3 \
    --itr 1 \
    --cycle 24
done
