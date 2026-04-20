"""
EMAformerDynamic - 动态嵌入护甲版

创新点：
1. 使用数据驱动的动态统计特征替代静态可学习嵌入
2. 支持 Zero-Shot 跨域迁移
3. 自适应通道和相位表示

核心变化：
- channel_embedding: (N, E) 静态参数 -> DynamicChannelEmbedding: 从数据统计提取
- phase_embedding: (cycle, E) 查表 -> DynamicPhaseEmbedding: 自适应周期编码
- joint_embedding: (cycle, N*E) 查表 -> DynamicJointEmbedding: 统计感知的联合编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.DynamicEmbedding import DynamicEmbeddingArmor
import numpy as np


class Model(nn.Module):
    """
    EMAformerDynamic 模型

    与原版 EMAformer 的主要区别：
    - 使用 DynamicEmbeddingArmor 替代静态嵌入
    - 嵌入从数据动态计算，而非固定参数
    - 支持跨域迁移和 Zero-Shot 场景
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
        # 开启后忽略外部 cycle_index，改为模型内部自动估计主导周期
        self.auto_cycle = getattr(configs, 'auto_cycle', True)

        # ========== 核心创新：动态嵌入护甲 ==========
        # 原版：使用静态可学习的 channel_embedding, phase_embedding, joint_embedding

        # 新版：从数据中动态提取统计特征
        # Inverted Embedding (与原版相同)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout
        )

        # 动态嵌入护甲 (替代原版的三个静态嵌入)
        self.dynamic_embedding = DynamicEmbeddingArmor(
            d_model=configs.d_model,
            n_phases=configs.cycle,
            fft_k=getattr(configs, 'fft_k', 5),
            autocorr_lags=getattr(configs, 'autocorr_lags', [1, 3, 6, 12, 24]),
            autocorr_mode=getattr(configs, 'autocorr_mode', 'fixed'),
            max_freq_levels=getattr(configs, 'max_freq_levels', 4),
            fft_feature_mode=getattr(configs, 'fft_feature_mode', 'hard_topk'),
            n_freq_bands=getattr(configs, 'n_freq_bands', 4),
            soft_select_temp=getattr(configs, 'soft_select_temp', 0.15)
        )

        # ========== Encoder (与原版相同) ==========
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

        # ========== 输出投影 (与原版相同) ==========
        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 2, configs.d_model * 4),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 4, configs.pred_len),
        )

        # ========== 可选：保留部分静态嵌入作为残差 ==========
        # 原版证明静态嵌入有效，新版可以保留作为残差补充
        self.use_static_residual = getattr(configs, 'use_static_residual', False)
        if self.use_static_residual:
            self.static_channel_embedding = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
            self.static_phase_embedding = nn.Embedding(configs.cycle, configs.d_model)
            nn.init.xavier_normal_(self.static_channel_embedding)
            nn.init.xavier_normal_(self.static_phase_embedding.weight)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):
        """
        前向传播

        与原版主要区别：
        - 嵌入从动态计算，而非直接查表
        - 返回额外的统计特征信息
        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape

        # ========== 1. 基础嵌入 (与原版相同) ==========
        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # ========== 核心创新：动态嵌入护甲 ==========
        # 从输入数据动态提取统计特征
        phase_input = None if self.auto_cycle else phase
        dynamic_embs = self.dynamic_embedding(x_enc, phase_input)

        channel_emb = dynamic_embs['channel_emb']  # (B, N, E)
        phase_emb = dynamic_embs['phase_emb']      # (B, N, E)
        joint_emb = dynamic_embs['joint_emb']      # (B, N, E)
        stats = dynamic_embs['stats']              # (B, N, stat_dim) - 统计特征

        # 可选：添加静态残差
        if self.use_static_residual:
            static_channel = self.static_channel_embedding.unsqueeze(0).expand(B, -1, -1)[:, :N, :]

            if phase is None:
                phase_indices = torch.zeros(B, device=x_enc.device, dtype=torch.long)
            elif isinstance(phase, torch.Tensor):
                if phase.dim() == 0:
                    phase_indices = phase.view(1).expand(B)
                elif phase.dim() == 1:
                    phase_indices = phase
                else:
                    phase_indices = phase[:, 0]
            else:
                phase_indices = torch.tensor(phase, device=x_enc.device).view(1).expand(B)

            if phase_indices.shape[0] != B:
                phase_indices = phase_indices[:1].expand(B)

            phase_indices = phase_indices.long() % self.cycle_len
            static_phase = self.static_phase_embedding(phase_indices.unsqueeze(1).expand(B, N))
            channel_emb = channel_emb + static_channel
            phase_emb = phase_emb + static_phase

        # 融合嵌入
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        # ========== 2. 通过 Transformer Encoder ==========
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # ========== 3. 输出投影 ==========
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns, stats

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None, mask=None):
        """
        输入:
            x_enc: (B, L, N) - 编码器输入
            x_mark_enc: (B, L, _) - 时间戳特征
            x_dec: (B, label_len+pred_len, N) - 解码器输入
            x_mark_dec: (B, label_len+pred_len, _) - 解码器时间戳
            cycle_index: (B,) / (B,N) / 标量 / None
                - 当 auto_cycle=True 时会被忽略，模型自动估计主导周期
                - 当 auto_cycle=False 且为 None 时，使用默认相位

        输出:
            dec_out: (B, pred_len, N) - 预测结果
            attns: 注意力权重 (如果 output_attention=True)
            stats: 统计特征 (额外输出，用于分析)
        """
        dec_out, attns, stats = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, stats
        else:
            return dec_out[:, -self.pred_len:, :], None, stats


class EMAformerDynamicZeroShot(Model):
    """
    EMAformerDynamic Zero-Shot 变体

    专门为跨域 Zero-Shot 场景设计：
    1. 完全移除静态嵌入，全部使用动态统计特征
    2. 增强统计特征的表达能力
    3. 支持未见过的通道配置

    使用方法：
    - 在新数据集上直接推理，无需微调
    - 或作为少样本学习的初始化
    """

    def __init__(self, configs):
        super(EMAformerDynamicZeroShot, self).__init__(configs)

        # 完全依赖动态嵌入，禁用静态残差
        self.use_static_residual = False

        # 增强统计特征的投影
        self.stat_proj = nn.Sequential(
            nn.Linear(
                self.dynamic_embedding.channel_embed.feat_dim,
                configs.d_model * 2
            ),
            nn.LayerNorm(configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model * 2, configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None, mask=None):
        """
        Zero-Shot 前向传播
        """
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape

        # 基础嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 动态嵌入
        phase_input = None if self.auto_cycle else cycle_index
        dynamic_embs = self.dynamic_embedding(x_enc, phase_input)

        # 增强统计特征投影
        stats = dynamic_embs['stats']  # (B, N, feat_dim)
        channel_emb = self.stat_proj(stats)  # (B, N, d_model)

        phase_emb = dynamic_embs['phase_emb']
        joint_emb = dynamic_embs['joint_emb']

        # 融合
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 投影
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns, stats


class EMAformerDynamicTransfer(Model):
    """
    EMAformerDynamic 跨域迁移变体

    设计用于领域适应场景：
    1. 保留动态嵌入的计算图
    2. 添加域对齐正则化项
    3. 支持多源域训练

    训练时使用域分类器进行对抗训练
    """

    def __init__(self, configs, n_domains=2):
        super(EMAformerDynamicTransfer, self).__init__(configs)

        self.n_domains = n_domains

        # 域分类器 (用于对抗训练)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.dynamic_embedding.channel_embed.feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_domains)
        )

        # 统计特征的域不变表示
        self.domain_invariance = nn.Sequential(
            nn.Linear(
                self.dynamic_embedding.channel_embed.feat_dim,
                configs.d_model
            ),
            nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None,
                domain_labels=None, return_stats=False):
        """
        跨域迁移前向传播

        额外输出:
            domain_pred: 域预测 (用于对抗训练)
            stats: 统计特征 (用于域分类)
        """
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape

        # 基础嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 动态嵌入
        phase_input = None if self.auto_cycle else cycle_index
        dynamic_embs = self.dynamic_embedding(x_enc, phase_input)

        # 域不变表示
        stats = dynamic_embs['stats']  # (B, N, feat_dim)
        channel_emb = self.domain_invariance(stats)  # (B, N, d_model)

        phase_emb = dynamic_embs['phase_emb']
        joint_emb = dynamic_embs['joint_emb']

        # 融合
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 投影
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 域预测 (用于对抗训练)
        if domain_labels is not None and self.training:
            # 聚合统计特征用于域分类 (取平均)
            stats_pooled = stats.mean(dim=1)  # (B, feat_dim)
            domain_pred = self.domain_classifier(stats_pooled)  # (B, n_domains)
            return dec_out[:, -self.pred_len:, :], attns, stats, domain_pred

        if return_stats:
            return dec_out[:, -self.pred_len:, :], attns, stats

        return dec_out[:, -self.pred_len:, :], attns, None
