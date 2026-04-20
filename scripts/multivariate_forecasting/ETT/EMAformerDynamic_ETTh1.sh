#!/bin/bash

# EMAformerDynamic (fixed backend) - ETTh1 实验脚本
# 使用方法: bash scripts/multivariate_forecasting/ETT/EMAformerDynamic_ETTh1.sh

model_name=EMAformerDynamic

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Dynamic_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Dynamic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Dynamic_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Dynamic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Dynamic_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Dynamic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Dynamic_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Dynamic' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.5 \
  --itr 1 \
  --patience 5 \
  --cycle 24
