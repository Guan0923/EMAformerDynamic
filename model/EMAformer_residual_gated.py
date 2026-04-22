import torch
import torch.nn as nn
from layers.DynamicEmbedding_fixed import (
    StatisticalFeatures,
    DynamicChannelEmbedding,
    DynamicPhaseEmbedding,
    DynamicJointEmbedding,
)
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    EMAformer Residual-Gated Dynamic Embedding 版本

    关键设计：
    1. 保留原版静态 embedding（channel/phase/joint）
    2. 动态分支仅做 residual correction，不直接替换静态项
    3. 每个动态分支前增加可学习 gate（sigmoid 后在 0~1）
    4. gate 初始化为较小值，使初始行为接近原版 EMAformer
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

        # Ablation 开关：默认全开，但可按分支关闭
        self.use_dynamic_channel_residual = getattr(configs, 'use_dynamic_channel_residual', True)
        self.use_dynamic_phase_residual = getattr(configs, 'use_dynamic_phase_residual', True)
        self.use_dynamic_joint_residual = getattr(configs, 'use_dynamic_joint_residual', True)

        # 周期策略：仅在 phase 缺失且 auto_cycle=True 时内部估计
        self.auto_cycle = getattr(configs, 'auto_cycle', False)

        # gate 初始偏置：默认 -4.0，sigmoid 后约 0.018，初始接近静态模型
        gate_init_bias = float(getattr(configs, 'gate_init_bias', -4.0))

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.class_strategy = configs.class_strategy

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 2, configs.d_model * 4),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 4, configs.pred_len),
        )

        # ===== 原版静态 embedding（必须保留） =====
        self.channel_embedding = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
        self.phase_embedding = nn.Embedding(configs.cycle, configs.d_model)
        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        nn.init.xavier_normal_(self.channel_embedding)

        # ===== 动态分支模块 =====
        self.dynamic_channel_embed = DynamicChannelEmbedding(
            fft_k=getattr(configs, 'fft_k', 5),
            autocorr_lags=getattr(configs, 'autocorr_lags', [1, 3, 6, 12, 24]),
            autocorr_mode=getattr(configs, 'autocorr_mode', 'fixed'),
            max_freq_levels=getattr(configs, 'max_freq_levels', 4),
            fft_feature_mode=getattr(configs, 'fft_feature_mode', 'hard_topk'),
            n_freq_bands=getattr(configs, 'n_freq_bands', 4),
            soft_select_temp=getattr(configs, 'soft_select_temp', 0.15),
        )

        self.dynamic_phase_embed = DynamicPhaseEmbedding(
            d_model=configs.d_model,
            n_phases=configs.cycle,
        )

        n_basic = 4
        n_fft = 2 * getattr(configs, 'fft_k', 5)
        n_autocorr = len(getattr(configs, 'autocorr_lags', [1, 3, 6, 12, 24]))
        self.stat_feat_dim = n_basic + n_fft + n_autocorr

        self.stat_features = StatisticalFeatures(
            fft_k=getattr(configs, 'fft_k', 5),
            autocorr_lags=getattr(configs, 'autocorr_lags', [1, 3, 6, 12, 24]),
            autocorr_mode=getattr(configs, 'autocorr_mode', 'fixed'),
            max_freq_levels=getattr(configs, 'max_freq_levels', 4),
            fft_feature_mode=getattr(configs, 'fft_feature_mode', 'hard_topk'),
            n_freq_bands=getattr(configs, 'n_freq_bands', 4),
            soft_select_temp=getattr(configs, 'soft_select_temp', 0.15),
        )

        self.dynamic_joint_embed = DynamicJointEmbedding(
            d_model=configs.d_model,
            n_phases=configs.cycle,
            stat_feat_dim=self.stat_feat_dim,
        )

        # ===== 动态 delta 稳定投影层（避免统计特征直接冲击主干） =====
        self.channel_delta_proj = nn.Linear(self.dynamic_channel_embed.feat_dim, configs.d_model)
        self.phase_delta_proj = nn.Linear(configs.d_model, configs.d_model)
        self.joint_delta_proj = nn.Linear(configs.d_model, configs.d_model)

        self.channel_delta_norm = nn.LayerNorm(configs.d_model)
        self.phase_delta_norm = nn.LayerNorm(configs.d_model)
        self.joint_delta_norm = nn.LayerNorm(configs.d_model)

        # ===== 可学习 gate（按 d_model 维度门控，初始很小） =====
        self.gate_c = nn.Parameter(torch.full((1, 1, configs.d_model), gate_init_bias))
        self.gate_p = nn.Parameter(torch.full((1, 1, configs.d_model), gate_init_bias))
        self.gate_j = nn.Parameter(torch.full((1, 1, configs.d_model), gate_init_bias))

    def _estimate_dominant_period(self, x):
        """
        基于输入序列估计每个样本主导周期（仅在 phase 缺失且 auto_cycle=True 时使用）。
        """
        B, L, N = x.shape

        fft_result = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(fft_result).mean(dim=2)

        if amplitude.shape[1] <= 1:
            return torch.full((B,), float(max(2, L)), device=x.device)

        amplitude[:, 0] = 0.0
        dom_idx = torch.argmax(amplitude[:, 1:], dim=1) + 1

        periods = L / dom_idx.float()
        periods = periods.clamp(min=2.0, max=float(max(2, L)))
        return periods

    def _build_continuous_phase(self, x, periods):
        """
        根据估计周期构造连续相位比例 (0~1)，用于动态 phase/joint 分支。
        """
        B, _, N = x.shape
        t = torch.tensor(float(x.shape[1] - 1), device=x.device).expand(B)
        phase_ratio = torch.remainder(t, periods) / (periods + 1e-8)
        return phase_ratio.unsqueeze(1).expand(-1, N)

    def _to_discrete_phase(self, phase, x):
        """
        将 phase 输入统一到离散索引 (B,)：
        - 外部给 cycle_index 时优先使用离散 phase
        - phase 为空时使用安全回退
        """
        B, _, _ = x.shape
        device = x.device

        if phase is None:
            return torch.zeros(B, device=device, dtype=torch.long)

        if not isinstance(phase, torch.Tensor):
            phase = torch.tensor(phase, device=device)

        phase = phase.to(device)

        if phase.dim() == 0:
            phase = phase.view(1).expand(B)
        elif phase.dim() == 1:
            if phase.shape[0] != B:
                phase = phase[0].view(1).expand(B)
        else:
            # (B, N) / (B, 1) 回收为样本级离散相位索引
            phase = phase[:, 0]

        if phase.dtype.is_floating_point:
            # 支持 phase ratio 输入；否则按离散值取模
            if torch.all((phase >= 0.0) & (phase <= 1.0)):
                phase = torch.floor(phase * self.cycle_len)
            phase = phase.long()
        else:
            phase = phase.long()

        phase = torch.remainder(phase, self.cycle_len)
        return phase

    def _build_phase_inputs(self, x_raw, phase):
        """
        生成静态查表用离散 phase，以及动态分支用 phase 输入。
        规则：
        1. 外部有 cycle_index 时优先离散 phase
        2. 仅当 cycle_index is None 且 auto_cycle=True 时启用内部周期估计
        """
        B, _, N = x_raw.shape

        if phase is not None:
            discrete_phase = self._to_discrete_phase(phase, x_raw)
            dynamic_phase_input = discrete_phase.view(-1, 1).expand(B, N)
            return discrete_phase, dynamic_phase_input

        if self.auto_cycle:
            periods = self._estimate_dominant_period(x_raw)
            cont_phase = self._build_continuous_phase(x_raw, periods)
            discrete_phase = torch.floor(cont_phase[:, 0] * self.cycle_len).long()
            discrete_phase = torch.remainder(discrete_phase, self.cycle_len)
            return discrete_phase, cont_phase

        discrete_phase = torch.zeros(B, device=x_raw.device, dtype=torch.long)
        dynamic_phase_input = discrete_phase.view(-1, 1).expand(B, N)
        return discrete_phase, dynamic_phase_input

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):
        # 原始输入用于动态统计与周期估计（在归一化前提取）
        x_raw = x_enc

        B, L, N = x_enc.shape

        # 动态统计在归一化前提取：避免 mean/std 被强制到 0/1 后统计先验被抹平
        stats = None
        if self.use_dynamic_channel_residual or self.use_dynamic_joint_residual:
            stats = self.stat_features(x_raw)

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # ===== 静态 embedding（原版保留） =====
        discrete_phase, dynamic_phase_input = self._build_phase_inputs(x_raw, phase)

        channel_static = self.channel_embedding.unsqueeze(0).expand(B, -1, -1)[:, :N, :]
        phase_static = self.phase_embedding(discrete_phase.view(-1, 1).expand(B, N))
        joint_static = self.joint_embedding(discrete_phase).reshape(B, self.enc_in, self.d_model)[:, :N, :]

        # ===== 动态 residual delta（不替换静态，仅做修正） =====
        if self.use_dynamic_channel_residual:
            channel_stats = self.dynamic_channel_embed(x_raw)
            channel_delta = self.channel_delta_norm(self.channel_delta_proj(channel_stats))
        else:
            channel_delta = torch.zeros(B, N, self.d_model, device=x_enc.device, dtype=x_enc.dtype)

        if self.use_dynamic_phase_residual:
            phase_delta_raw = self.dynamic_phase_embed(B, N, dynamic_phase_input, x_enc.device)
            phase_delta = self.phase_delta_norm(self.phase_delta_proj(phase_delta_raw))
        else:
            phase_delta = torch.zeros(B, N, self.d_model, device=x_enc.device, dtype=x_enc.dtype)

        if self.use_dynamic_joint_residual:
            if stats is None:
                stats = self.stat_features(x_raw)
            joint_delta_raw = self.dynamic_joint_embed(stats, B, N, dynamic_phase_input, x_enc.device)
            joint_delta = self.joint_delta_norm(self.joint_delta_proj(joint_delta_raw))
        else:
            joint_delta = torch.zeros(B, N, self.d_model, device=x_enc.device, dtype=x_enc.dtype)

        # gate 用 sigmoid 压到 [0,1]，且小初始化让初始行为接近原版 EMAformer
        gc = torch.sigmoid(self.gate_c)
        gp = torch.sigmoid(self.gate_p)
        gj = torch.sigmoid(self.gate_j)

        channel_final = channel_static + gc * channel_delta
        phase_final = phase_static + gp * phase_delta
        joint_final = joint_static + gj * joint_delta

        # 正确语义：enc_out + 三个融合后的 embedding
        enc_out = enc_out[:, :N, :] + channel_final + phase_final + joint_final
        enc_orgin = enc_out

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
