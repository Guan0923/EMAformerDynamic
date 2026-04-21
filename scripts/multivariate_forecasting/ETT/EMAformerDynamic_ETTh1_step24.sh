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

# EMAformerDynamic (ETTh1)
# 预测窗口从 24 开始，按 24 递增。
# 使用方法:
#   bash scripts/multivariate_forecasting/EMAformerDynamic_ETTh1_step24.sh

model_name=EMAformerDynamic
auto_cycle=true
seq_len=96
start_pred=24
end_pred=96
step_pred=24

for pred_len in $(seq "$start_pred" "$step_pred" "$end_pred"); do
  e_layers=3
  output_proj_dropout=0.3
  patience_args=()

  # 与现有 ETTh1 720 配置对齐
  if [[ "$pred_len" -eq 720 ]]; then
    e_layers=2
    output_proj_dropout=0.5
    patience_args=(--patience 5)
  fi

  "$PYTHON_BIN" -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_Dynamic_${seq_len}_${pred_len} \
    --model "$model_name" \
    --data ETTh1 \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp_Dynamic_step24' \
    --d_model 256 \
    --n_heads 4 \
    --d_ff 256 \
    --output_proj_dropout "$output_proj_dropout" \
    --itr 1 \
    --cycle 24 \
    --auto_cycle "$auto_cycle" \
    "${patience_args[@]}"
done
