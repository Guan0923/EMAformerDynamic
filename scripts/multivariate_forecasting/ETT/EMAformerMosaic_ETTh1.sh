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

model_name=EMAformerMosaic
#sym:auto_cycle=true|false
auto_cycle=true
#sym:patch_len_list='[4,8,16]'|'[2,4,8,16]'
patch_len_list='[4,8,16]'
#sym:num_latent_token=2|4|8
num_latent_token=4

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Mosaic_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Mosaic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --patch_len_list "$patch_len_list" \
  --num_latent_token "$num_latent_token" \
  --itr 1 \
  --cycle 24 \
  --auto_cycle "$auto_cycle"

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Mosaic_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Mosaic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --patch_len_list "$patch_len_list" \
  --num_latent_token "$num_latent_token" \
  --itr 1 \
  --cycle 24 \
  --auto_cycle "$auto_cycle"

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Mosaic_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Mosaic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --patch_len_list "$patch_len_list" \
  --num_latent_token "$num_latent_token" \
  --itr 1 \
  --cycle 24 \
  --auto_cycle "$auto_cycle"

"$PYTHON_BIN" -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Mosaic_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Mosaic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.5 \
  --patch_len_list "$patch_len_list" \
  --num_latent_token "$num_latent_token" \
  --itr 1 \
  --patience 5 \
  --cycle 24 \
  --auto_cycle "$auto_cycle"
