import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.DynamicEmbedding_fixed import StatisticalFeatures, DynamicJointEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    EMAformer 消融变体：仅将静态联合矩阵替换为动态联合矩阵

    嵌入策略：
    - channel_emb : 静态通道嵌入（保留原版 nn.Parameter）
    - phase_emb   : 静态相位嵌入（保留原版 nn.Embedding）
    - joint_emb   : 动态联合嵌入（通道-相位交互建模）

    设计目的：验证"仅替换联合矩阵"对模型性能的影响
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

        # ========== Inverted Embedding（与原版相同）==========
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout
        )

        # ========== 统计特征提取器（为动态联合嵌入提供统计特征）==========
        self.stat_features = StatisticalFeatures(
            fft_k=getattr(configs, 'fft_k', 5),
            autocorr_lags=getattr(configs, 'autocorr_lags', [1, 3, 6, 12, 24]),
            autocorr_mode=getattr(configs, 'autocorr_mode', 'fixed'),
            max_freq_levels=getattr(configs, 'max_freq_levels', 4),
            fft_feature_mode=getattr(configs, 'fft_feature_mode', 'hard_topk'),
            n_freq_bands=getattr(configs, 'n_freq_bands', 4),
            soft_select_temp=getattr(configs, 'soft_select_temp', 0.15)
        )

        # 计算统计特征维度
        n_basic = 4  # mean, std, skew, kurtosis
        n_fft = 2 * getattr(configs, 'fft_k', 5)
        n_autocorr = len(getattr(configs, 'autocorr_lags', [1, 3, 6, 12, 24]))
        self.stat_feat_dim = n_basic + n_fft + n_autocorr

        # ========== 核心替换：动态联合嵌入 ==========
        self.dynamic_joint_embed = DynamicJointEmbedding(
            d_model=configs.d_model,
            n_phases=configs.cycle,
            stat_feat_dim=self.stat_feat_dim
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

        # ========== 保留原版的静态通道嵌入和相位嵌入 ==========
        self.channel_embedding = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
        nn.init.xavier_normal_(self.channel_embedding)

        self.phase_embedding = nn.Embedding(configs.cycle, configs.d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)

    def _estimate_dominant_period(self, x):
        """
        基于输入序列自动估计每个样本的主导周期。
        
        返回:
            periods: (B,) 浮点周期长度（单位: 时间步）
        """
        B, L, N = x.shape

        fft_result = torch.fft.rfft(x, dim=1)           # (B, F, N)
        amplitude = torch.abs(fft_result).mean(dim=2)   # (B, F)

        if amplitude.shape[1] <= 1:
            return torch.full((B,), float(max(2, L)), device=x.device)

        amplitude[:, 0] = 0.0  # 去掉直流分量
        dom_idx = torch.argmax(amplitude[:, 1:], dim=1) + 1  # (B,)

        periods = L / dom_idx.float()
        periods = periods.clamp(min=2.0, max=float(max(2, L)))
        return periods

    def _build_continuous_phase(self, x, periods):
        """
        根据估计周期构造连续相位比例 (0~1)。
        使用序列末尾位置作为当前时间点。
        """
        B, _, N = x.shape
        t = torch.tensor(float(x.shape[1] - 1), device=x.device).expand(B)
        phase_ratio = torch.remainder(t, periods) / (periods + 1e-8)  # (B,)
        return phase_ratio.unsqueeze(1).expand(-1, N)  # (B, N)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):
        B, L, N = x_enc.shape

        # ========== 【关键】在归一化之前提取统计特征（用于动态联合嵌入）==========
        stats = self.stat_features(x_enc)  # (B, N, feat_dim)

        # ========== 归一化（与原版相同）==========
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # ========== 基础嵌入（与原版相同）==========
        enc_out = self.enc_embedding(x_enc, x_mark_enc)     # (B, N, d_model)

        # ========== 获取三种嵌入 ==========
        # 1. 静态通道嵌入（原版）
        channel_emb = self.channel_embedding.unsqueeze(0).expand(B, -1, -1)[:, :N, :]  # (B, N, d_model)

        # 2. 静态相位嵌入（原版）
        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))  # (B, N, d_model)

        # 3. 动态联合嵌入（替换版）
        # 如果 auto_cycle=True 且 phase 为 None，则自动估计周期
        if self.auto_cycle and phase is None:
            estimated_period = self._estimate_dominant_period(x_enc)
            phase_indices = self._build_continuous_phase(x_enc, estimated_period)
        else:
            phase_indices = phase
        
        joint_emb = self.dynamic_joint_embed(stats, B, N, phase_indices, x_enc.device)  # (B, N, d_model)

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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
