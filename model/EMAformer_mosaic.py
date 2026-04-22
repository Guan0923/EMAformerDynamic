import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class AdaptivePatchEmbedding(nn.Module):
    """TimeMosaic-style adaptive chunking with region-wise patch-length selection."""

    def __init__(self, d_model, patch_len_list, dropout=0.0, seq_len=96, training=True):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.max_patch_len = max(patch_len_list)
        self.min_patch_len = min(patch_len_list)
        self.region_num = seq_len // self.max_patch_len
        self.training = training
        self.target_patch_num = self.max_patch_len // self.min_patch_len

        self.region_cls = nn.Sequential(
            nn.Linear(self.max_patch_len, 64),
            nn.ReLU(),
            nn.Linear(64, len(patch_len_list)),
        )

        self.embeddings = nn.ModuleList([
            nn.Linear(patch_len, d_model, bias=False) for patch_len in patch_len_list
        ])

        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, L]
        bsz, channels, seq_len = x.shape
        expected_len = self.region_num * self.max_patch_len
        assert seq_len == expected_len, f"Expected seq_len={expected_len}, but got {seq_len}"

        x = x.reshape(bsz * channels, self.region_num, self.max_patch_len)

        all_patches = []
        all_cls_pred = []

        for region_idx in range(self.region_num):
            region = x[:, region_idx, :]
            cls_logits = self.region_cls(region)

            if self.training:
                cls_soft = F.gumbel_softmax(cls_logits, tau=0.5, hard=True, dim=-1)
            else:
                cls_pred = torch.argmax(cls_logits, dim=-1)
                cls_soft = F.one_hot(cls_pred, num_classes=len(self.patch_len_list)).float()

            all_cls_pred.append(torch.argmax(cls_soft, dim=-1))

            patch_emb_list = []
            for idx, patch_len in enumerate(self.patch_len_list):
                patches = region.unfold(-1, patch_len, patch_len)
                repeat = self.target_patch_num - patches.size(1)
                if repeat > 0:
                    patches = patches.repeat_interleave(repeat + 1, dim=1)[:, :self.target_patch_num, :]
                patch_emb_list.append(self.embeddings[idx](patches))

            patch_emb_stack = torch.stack(patch_emb_list, dim=0)
            cls_soft_trans = cls_soft.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
            region_patches = (patch_emb_stack * cls_soft_trans).sum(dim=0)
            all_patches.append(region_patches)

        x_patch = torch.cat(all_patches, dim=1)
        x_patch = x_patch + self.position_embedding(x_patch)
        x_patch = self.dropout(x_patch)

        cls_pred = torch.cat(all_cls_pred, dim=0)
        return x_patch, channels, cls_pred


class Model(nn.Module):
    """
    EMAformerMosaic:
    - EMAformer: phase/channel/joint embeddings
    - TimeMosaic: adaptive region-wise patch chunking
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

        self.patch_len_list = eval(getattr(configs, "patch_len_list", "[4,8,16]"))
        self.num_latent_token = int(getattr(configs, "num_latent_token", 4))

        self.patch_embedding = AdaptivePatchEmbedding(
            d_model=configs.d_model,
            patch_len_list=self.patch_len_list,
            dropout=configs.dropout,
            seq_len=configs.seq_len,
            training=bool(configs.is_training),
        )

        self.patch_num = self.patch_embedding.region_num * self.patch_embedding.target_patch_num

        self.prompt_embeddings = nn.Embedding(self.num_latent_token, self.d_model)
        nn.init.xavier_uniform_(self.prompt_embeddings.weight)

        self.channel_embedding = nn.Embedding(self.enc_in, self.d_model)
        self.phase_embedding = nn.Embedding(self.cycle_len, self.d_model)
        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.d_model)
        nn.init.xavier_normal_(self.channel_embedding.weight)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        nn.init.xavier_normal_(self.joint_embedding.weight)

        # Keep a lightweight token for covariates, similar to EMAformer family style.
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

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
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.projector = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.patch_num * self.d_model, self.pred_len),
        )

    def _prepare_phase(self, cycle_index, bsz, device):
        if cycle_index is None:
            phase = torch.zeros(bsz, dtype=torch.long, device=device)
            return phase

        if cycle_index.dim() == 0:
            phase = cycle_index.view(1).repeat(bsz)
        elif cycle_index.dim() == 1:
            phase = cycle_index
        else:
            phase = cycle_index[:, 0]
        return phase.long() % self.cycle_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        bsz, _, channels = x_enc.shape

        # [B, C, L]
        patch_input = x_enc.permute(0, 2, 1)
        patch_tokens, n_vars, cls_pred = self.patch_embedding(patch_input)
        patch_tokens = patch_tokens[:, :self.patch_num, :]

        # EMAformer-style three embeddings on patch tokens.
        channel_ids = torch.arange(n_vars, device=x_enc.device).unsqueeze(0).expand(bsz, -1).reshape(-1)
        channel_emb = self.channel_embedding(channel_ids).unsqueeze(1).expand(-1, self.patch_num, -1)

        phase = self._prepare_phase(cycle_index, bsz, x_enc.device)
        phase_bc = phase.repeat_interleave(n_vars)
        phase_emb = self.phase_embedding(phase_bc).unsqueeze(1).expand(-1, self.patch_num, -1)

        joint_table = self.joint_embedding(phase).reshape(bsz, self.enc_in, self.d_model)[:, :n_vars, :]
        joint_emb = joint_table.reshape(bsz * n_vars, self.d_model).unsqueeze(1).expand(-1, self.patch_num, -1)

        patch_tokens = patch_tokens + channel_emb + phase_emb + joint_emb

        # Add global/calendrical token stream from DataEmbedding_inverted.
        global_tokens = self.enc_embedding(x_enc, x_mark_enc)
        cal_tokens = global_tokens[:, channels:, :]
        if cal_tokens.size(1) > 0:
            cal_tokens = cal_tokens.repeat_interleave(n_vars, dim=0)
        else:
            cal_tokens = patch_tokens.new_zeros((bsz * n_vars, 0, self.d_model))

        prompt = self.prompt_embeddings.weight.unsqueeze(0).expand(bsz * n_vars, -1, -1)
        enc_in = torch.cat([prompt, patch_tokens, cal_tokens], dim=1)

        enc_out, attns = self.encoder(enc_in, attn_mask=None)
        patch_out = enc_out[:, self.num_latent_token:self.num_latent_token + self.patch_num, :]

        dec_bc = self.projector(patch_out)
        dec_out = dec_bc.reshape(bsz, n_vars, self.pred_len).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns, cls_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None, mask=None):
        dec_out, attns, cls_pred = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, cls_pred
        return dec_out[:, -self.pred_len:, :]
