import torch
import torch.nn as nn
from layers.DynamicEmbedding_fixed import DynamicPhaseEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    EMAformer 消融变体（token_k 版）：
    - 相位估计不再只取单一最大频率
    - 使用 top-k 频率加权融合估计主导周期
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
        self.auto_cycle = getattr(configs, 'auto_cycle', True)
        # token_k: 用于频域 top-k 融合的候选频率数
        self.token_k = int(getattr(configs, 'token_k', 3))

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout
        )

        self.dynamic_phase_embed = DynamicPhaseEmbedding(
            d_model=configs.d_model,
            n_phases=configs.cycle
        )

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
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
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

        self.channel_embedding = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
        nn.init.xavier_normal_(self.channel_embedding)

        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.d_model)
        nn.init.xavier_normal_(self.joint_embedding.weight)

    def _estimate_dominant_period(self, x):
        """
        基于输入序列自动估计每个样本的主导周期（top-k 融合版）。

        返回:
            periods: (B,) 浮点周期长度（单位: 时间步）
        """
        B, L, _ = x.shape

        fft_result = torch.fft.rfft(x, dim=1)         # (B, F, N)
        amplitude = torch.abs(fft_result).mean(dim=2) # (B, F)

        if amplitude.shape[1] <= 1:
            return torch.full((B,), float(max(2, L)), device=x.device)

        amplitude[:, 0] = 0.0  # 去掉直流分量
        spectrum = amplitude[:, 1:]  # (B, F-1)

        k = max(1, min(self.token_k, spectrum.shape[1]))
        topk_vals, topk_idx = torch.topk(spectrum, k=k, dim=1)

        freq_idx = (topk_idx + 1).float()
        weights = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-8)
        dom_idx = (weights * freq_idx).sum(dim=1)

        periods = L / (dom_idx + 1e-8)
        periods = periods.clamp(min=2.0, max=float(max(2, L)))
        return periods

    def _build_continuous_phase(self, x, periods):
        """
        根据估计周期构造连续相位比例 (0~1)。
        使用序列末尾位置作为当前时间点。
        """
        B, _, N = x.shape
        t = torch.tensor(float(x.shape[1] - 1), device=x.device).expand(B)
        phase_ratio = torch.remainder(t, periods) / (periods + 1e-8)
        return phase_ratio.unsqueeze(1).expand(-1, N)

    def _phase_to_joint_index(self, phase_indices):
        """
        将连续/离散 phase 映射到 joint embedding 的离散索引。
        """
        if phase_indices.dim() == 2:
            phase_for_joint = phase_indices[:, 0]
        else:
            phase_for_joint = phase_indices

        if phase_for_joint.dtype.is_floating_point:
            joint_index = torch.round(phase_for_joint * (self.cycle_len - 1)).long()
        else:
            joint_index = phase_for_joint.long()

        joint_index = joint_index % self.cycle_len
        return joint_index

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):
        B, _, N = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        channel_emb = self.channel_embedding.unsqueeze(0).expand(B, -1, -1)[:, :N, :]

        if self.auto_cycle and phase is None:
            estimated_period = self._estimate_dominant_period(x_enc)
            phase_indices = self._build_continuous_phase(x_enc, estimated_period)
        else:
            phase_indices = phase

        phase_emb = self.dynamic_phase_embed(B, N, phase_indices, x_enc.device)

        joint_index = self._phase_to_joint_index(phase_indices)
        joint_emb = self.joint_embedding(joint_index).reshape(B, self.enc_in, self.d_model)

        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_orgin = enc_out

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

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
        return dec_out[:, -self.pred_len:, :]
