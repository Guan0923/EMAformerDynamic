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

model_name=EMAformerResidualGated

# ===== Residual ablation switches =====
# true/false
use_dynamic_channel_residual=true
use_dynamic_phase_residual=true
use_dynamic_joint_residual=true

# 周期策略: 仅当 cycle_index 为空时才会触发模型内部估计
auto_cycle=false

# gate 小初始化: sigmoid(-4) ≈ 0.018，保证初始接近静态 EMAformer
gate_init_bias=-4.0

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_ResidualGated_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_ResidualGated' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24 \
  --auto_cycle "$auto_cycle" \
  --use_dynamic_channel_residual "$use_dynamic_channel_residual" \
  --use_dynamic_phase_residual "$use_dynamic_phase_residual" \
  --use_dynamic_joint_residual "$use_dynamic_joint_residual" \
  --gate_init_bias "$gate_init_bias"

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_ResidualGated_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_ResidualGated' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24 \
  --auto_cycle "$auto_cycle" \
  --use_dynamic_channel_residual "$use_dynamic_channel_residual" \
  --use_dynamic_phase_residual "$use_dynamic_phase_residual" \
  --use_dynamic_joint_residual "$use_dynamic_joint_residual" \
  --gate_init_bias "$gate_init_bias"

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_ResidualGated_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_ResidualGated' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24 \
  --auto_cycle "$auto_cycle" \
  --use_dynamic_channel_residual "$use_dynamic_channel_residual" \
  --use_dynamic_phase_residual "$use_dynamic_phase_residual" \
  --use_dynamic_joint_residual "$use_dynamic_joint_residual" \
  --gate_init_bias "$gate_init_bias"

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_ResidualGated_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_ResidualGated' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.5 \
  --itr 1 \
  --patience 5 \
  --cycle 24 \
  --auto_cycle "$auto_cycle" \
  --use_dynamic_channel_residual "$use_dynamic_channel_residual" \
  --use_dynamic_phase_residual "$use_dynamic_phase_residual" \
  --use_dynamic_joint_residual "$use_dynamic_joint_residual" \
  --gate_init_bias "$gate_init_bias"
