#!/bin/bash

# EMAformerDynamic - 动态嵌入护甲实验脚本
# 使用方法: bash scripts/multivariate_forecasting/ECL/EMAformerDynamic.sh

model_name=EMAformerDynamic

# ========== 实验 1: seq_len=96, pred_len=96 ==========
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_Dynamic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp_Dynamic' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --output_proj_dropout 0.1 \
  --learning_rate 0.0005 \
  --n_heads 4 \
  --itr 1 \
  --cycle 168

# ========== 实验 2: seq_len=96, pred_len=192 ==========
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_Dynamic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp_Dynamic' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --output_proj_dropout 0.1 \
  --learning_rate 0.0005 \
  --n_heads 4 \
  --itr 1 \
  --cycle 168

# ========== 实验 3: seq_len=96, pred_len=336 ==========
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_Dynamic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp_Dynamic' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --output_proj_dropout 0.1 \
  --learning_rate 0.0005 \
  --n_heads 4 \
  --itr 1 \
  --cycle 168

# ========== 实验 4: seq_len=96, pred_len=720 ==========
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_Dynamic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp_Dynamic' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --output_proj_dropout 0.1 \
  --learning_rate 0.0005 \
  --n_heads 4 \
  --itr 1 \
  --cycle 168
