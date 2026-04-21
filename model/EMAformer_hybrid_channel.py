import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.DynamicEmbedding_fixed import DynamicChannelEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    EMAformer 消融变体：仅将静态通道矩阵替换为动态通道矩阵

    嵌入策略：
    - channel_emb : 动态通道嵌入（从数据统计特征中提取）
    - phase_emb   : 静态相位嵌入（保留原版 nn.Embedding）
    - joint_emb   : 静态联合嵌入（保留原版 nn.Embedding）

    设计目的：验证"仅替换通道矩阵"对模型性能的影响
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.cycle_len = configs.cycle
        self.enc_in = configs.enc_in

        # ========== Inverted Embedding（与原版相同）==========
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout
        )

        # ========== 核心替换：动态通道嵌入 ==========
        self.dynamic_channel_embed = DynamicChannelEmbedding(
            fft_k=getattr(configs, 'fft_k', 5),
            autocorr_lags=getattr(configs, 'autocorr_lags', [1, 3, 6, 12, 24]),
            autocorr_mode=getattr(configs, 'autocorr_mode', 'fixed'),
            max_freq_levels=getattr(configs, 'max_freq_levels', 4),
            fft_feature_mode=getattr(configs, 'fft_feature_mode', 'hard_topk'),
            n_freq_bands=getattr(configs, 'n_freq_bands', 4),
            soft_select_temp=getattr(configs, 'soft_select_temp', 0.15)
        )

        # 将统计特征投影到 d_model 维度
        self.stat_proj = nn.Linear(
            self.dynamic_channel_embed.feat_dim,
            configs.d_model
        )

        # ========== Encoder（与原版相同）==========
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # ========== 输出投影（与原版相同）==========
        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 2, configs.d_model * 4),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 4, configs.pred_len),
        )

        # ========== 保留原版的静态相位嵌入和联合嵌入 ==========
        self.phase_embedding = nn.Embedding(configs.cycle, configs.d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)

        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.d_model)
        nn.init.xavier_normal_(self.joint_embedding.weight)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):
        B, L, N = x_enc.shape

        # ========== 【关键】在归一化之前提取动态通道嵌入 ==========
        # 原因：动态嵌入从原始数据的统计特征中提取，
        #       若先归一化会导致 mean->0, std->1，丢失统计信息
        stats = self.dynamic_channel_embed(x_enc)           # (B, N, feat_dim)
        channel_emb = self.stat_proj(stats)                  # (B, N, d_model)

        # ========== 归一化（与原版相同）==========
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # ========== 基础嵌入（与原版相同）==========
        enc_out = self.enc_embedding(x_enc, x_mark_enc)     # (B, N, d_model)

        # ========== 获取三种嵌入 ==========
        # 1. 动态通道嵌入（已提前计算）
        # channel_emb: (B, N, d_model)

        # 2. 静态相位嵌入（原版）
        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))  # (B, N, d_model)

        # 3. 静态联合嵌入（原版）
        joint_emb = self.joint_embedding(phase).reshape(B, self.enc_in, self.d_model)  # (B, N, d_model)

        # ========== 嵌入融合（与原版相同）==========
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        # ========== Transformer Encoder ==========
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # ========== 输出投影 ==========
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        # ========== 反归一化 ==========
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
