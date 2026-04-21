#!/bin/bash

# EMAformer_hybrid_phase_token_k 超参数分析脚本
# token_k 取值范围: [1, 48], 步长为 5
# 测试值: 1, 6, 11, 16, 21, 26, 31, 36, 41, 46
# 数据集: ETTh1

set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Error: python/python3 not found in PATH"
  exit 127
fi

model_name=EMAformerHybridPhaseTokenK
data_name=ETTh1
data_path=ETTh1.csv
enc_in=7
dec_in=7
c_out=7

# token_k 取值列表
token_k_values=(1 6 11 16 21 26 31 36 41 46)

# 预测长度列表
pred_lens=(96 192 336 720)

for token_k in "${token_k_values[@]}"; do
  echo "=========================================="
  echo "Testing token_k=$token_k"
  echo "=========================================="

  for pred_len in "${pred_lens[@]}"; do
    # 根据 pred_len 调整 e_layers
    if [ "$pred_len" -eq 720 ]; then
      e_layers=2
      patience=5
      output_proj_dropout=0.5
    else
      e_layers=3
      patience=3
      output_proj_dropout=0.3
    fi

    echo "Running: pred_len=$pred_len, e_layers=$e_layers, token_k=$token_k"

    "$PYTHON_BIN" -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path "$data_path" \
      --model_id "${data_name}_96_${pred_len}_token_k${token_k}" \
      --model "$model_name" \
      --data "$data_name" \
      --features M \
      --seq_len 96 \
      --pred_len "$pred_len" \
      --e_layers "$e_layers" \
      --enc_in "$enc_in" \
      --dec_in "$dec_in" \
      --c_out "$c_out" \
      --des "Exp_token_k${token_k}" \
      --d_model 256 \
      --n_heads 4 \
      --d_ff 256 \
      --output_proj_dropout "$output_proj_dropout" \
      --itr 1 \
      --patience "$patience" \
      --cycle 24 \
      --auto_cycle True \
      --token_k "$token_k"
  done
done

echo "All experiments completed!"
