"""
Dynamic Embedding Layer - 数据驱动的动态统计特征嵌入（修复版）

修复内容：
- 修复 DynamicJointEmbedding 中硬编码 π 常量 3.14159265 与 math.pi 不一致的问题

创新点：
1. 基础统计特征：均值、标准差、偏度、峰度
2. 频域特征：FFT 提取前 k 个主导频率及振幅
3. 自相关特征：不同延迟下的自相关系数

这些特征从数据中动态计算，而非静态可学习参数，
实现跨域迁移和 Zero-Shot 能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, pi

class StatisticalFeatures(nn.Module):
    """
    提取时间序列的统计特征向量

    输入: (B, L, N) - B=batch, L=序列长度, N=通道数
    输出: (B, N, feat_dim) - 每个通道的统计特征向量
    """

    def __init__(
            self,
            fft_k=5,
            autocorr_lags=[1, 3, 6, 12, 24],
            autocorr_mode='fixed',
            max_freq_levels=4,
            fft_feature_mode='hard_topk',
            n_freq_bands=4,
            soft_select_temp=0.15):
        super(StatisticalFeatures, self).__init__()
        self.fft_k = fft_k          # 保留前 k 个主导频率
        self.autocorr_lags = autocorr_lags  # 自相关延迟
        self.autocorr_mode = autocorr_mode
        self.max_freq_levels = max_freq_levels
        self.fft_feature_mode = fft_feature_mode
        self.n_freq_bands = n_freq_bands
        self.soft_select_temp = soft_select_temp

        # 特征维度计算:
        # 基础统计: 4 (均值, 标准差, 偏度, 峰度)
        # FFT频域: fft_k * 2 (频率 + 振幅)
        # 自相关: len(autocorr_lags)
        self.fft_k = fft_k
        self.n_autocorr = len(autocorr_lags)
        self.autocorr_lags = autocorr_lags

        # 递归频域深度门控网络:
        # 输入: [残差能量占比, 当前层级比例]
        # 输出: need_deeper in [0,1]，表示是否有必要继续深入分析高频残差
        self.depth_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 1)
        )

        # 频带统计模式: 将多带宽统计特征投影到固定 2*fft_k 维
        self.band_proj = nn.Sequential(
            nn.Linear(2 * self.n_freq_bands, 2 * self.fft_k),
            nn.GELU(),
            nn.Linear(2 * self.fft_k, 2 * self.fft_k)
        )
        
    def _build_length_adaptive_lags(self, L):
        """
        策略1: 基于输入长度自动生成多尺度 lag。
        """
        if L <= 2:
            return [1]

        min_pow = 0
        max_pow = max(1, self.n_autocorr - 1)
        candidate = []
        for i in range(self.n_autocorr):
            # 近似几何尺度: L/2^k
            k = min_pow + int(round((max_pow - min_pow) * i / max(1, self.n_autocorr - 1)))
            lag = int(round(L / (2 ** (max_pow - k + 1))))
            lag = max(1, min(lag, L - 1))
            candidate.append(lag)

        # 去重并补齐
        lags = sorted(set(candidate))
        fill = 1
        while len(lags) < self.n_autocorr and fill < L:
            if fill not in lags:
                lags.append(fill)
            fill += 1
        return sorted(lags)[:self.n_autocorr]

    def _estimate_global_period(self, x):
        """基于 batch 平均频谱估计一个全局主导周期（用于 lag 生成）。"""
        _, L, _ = x.shape
        fft_result = torch.fft.rfft(x, dim=1)                 # (B, F, N)
        amp = torch.abs(fft_result).mean(dim=(0, 2))          # (F,)

        if amp.shape[0] <= 1:
            return float(max(2, L))

        amp[0] = 0.0
        dom_idx = int(torch.argmax(amp[1:]).item() + 1)
        period = float(L) / max(dom_idx, 1)
        return float(max(2.0, min(period, float(max(2, L)))))

    def _build_period_adaptive_lags(self, x, L):
        """
        策略2: 基于自动估计主导周期生成 lag。
        """
        if L <= 2:
            return [1]

        period = self._estimate_global_period(x)

        # 使用由短到长的周期比例，覆盖局部与全周期依赖
        base_ratios = [0.125, 0.25, 0.5, 1.0, 2.0, 3.0]
        if self.n_autocorr <= len(base_ratios):
            ratios = base_ratios[:self.n_autocorr]
        else:
            extra = [float(i + 4) for i in range(self.n_autocorr - len(base_ratios))]
            ratios = base_ratios + extra

        candidate = []
        for r in ratios:
            lag = int(round(period * r))
            lag = max(1, min(lag, L - 1))
            candidate.append(lag)

        lags = sorted(set(candidate))
        fill = 1
        while len(lags) < self.n_autocorr and fill < L:
            if fill not in lags:
                lags.append(fill)
            fill += 1
        return sorted(lags)[:self.n_autocorr]

    def _select_autocorr_lags(self, x, L):
        """根据 autocorr_mode 选择 lag 列表。"""
        if self.autocorr_mode == 'fixed':
            return [int(lag) for lag in self.autocorr_lags]
        if self.autocorr_mode == 'length_adaptive':
            return self._build_length_adaptive_lags(L)
        if self.autocorr_mode == 'period_adaptive':
            return self._build_period_adaptive_lags(x, L)

        raise ValueError(
            f"Unsupported autocorr_mode: {self.autocorr_mode}. "
            f"Choose from ['fixed', 'length_adaptive', 'period_adaptive']."
        )


    def _decode_hard_topk(self, spectrum, freqs_expanded):
        """硬选择 top-k：兼容原始实现。"""
        top_k = min(self.fft_k, spectrum.shape[-1])
        top_amplitude, top_idx = torch.topk(spectrum, k=top_k, dim=-1)
        top_freqs = torch.gather(freqs_expanded, dim=-1, index=top_idx)
        return top_freqs, top_amplitude

    def _decode_soft_select(self, spectrum, freqs_expanded):
        """软选择 top-k：通过概率加权迭代提取 k 组频率-振幅对。"""
        work = spectrum.clone()
        soft_freqs, soft_amps = [], []

        top_k = min(self.fft_k, spectrum.shape[-1])
        temp = max(self.soft_select_temp, 1e-4)

        for _ in range(top_k):
            weights = torch.softmax(work / temp, dim=-1)  # (B, N, F)
            sel_freq = (weights * freqs_expanded).sum(dim=-1, keepdim=True)  # (B, N, 1)
            sel_amp = (weights * work).sum(dim=-1, keepdim=True)  # (B, N, 1)

            soft_freqs.append(sel_freq)
            soft_amps.append(sel_amp)

            # 抑制已选区域，鼓励下一次选择覆盖剩余频段
            work = work * (1.0 - weights)

        top_freqs = torch.cat(soft_freqs, dim=-1)
        top_amplitude = torch.cat(soft_amps, dim=-1)
        return top_freqs, top_amplitude

    def _decode_band_stats(self, spectrum, freqs_expanded):
        """多带宽统计：先提取分带统计，再投影为固定 2*fft_k 维输出。"""
        B, N, F_bins = spectrum.shape
        band_feats = []

        for b in range(self.n_freq_bands):
            start = int(round(b * F_bins / self.n_freq_bands))
            end = int(round((b + 1) * F_bins / self.n_freq_bands))
            if end <= start:
                end = min(start + 1, F_bins)

            band_amp = spectrum[:, :, start:end]  # (B, N, f_b)
            band_freq = freqs_expanded[:, :, start:end]

            mean_energy = band_amp.mean(dim=-1, keepdim=True)  # (B, N, 1)
            centroid = (band_amp * band_freq).sum(dim=-1, keepdim=True) / (band_amp.sum(dim=-1, keepdim=True) + 1e-8)

            band_feats.extend([mean_energy, centroid])

        band_feats = torch.cat(band_feats, dim=-1)  # (B, N, 2*n_bands)
        proj = self.band_proj(band_feats)  # (B, N, 2*fft_k)

        freq_logits = proj[:, :, :self.fft_k]
        amp_logits = proj[:, :, self.fft_k:]

        top_freqs = 0.5 * torch.sigmoid(freq_logits)  # 归一化频率范围 [0, 0.5]
        top_amplitude = F.softplus(amp_logits)        # 振幅保持非负
        return top_freqs, top_amplitude

    def compute_mean(self, x):
        """均值 - 反映数据中心位置"""
        return x.mean(dim=1, keepdim=True)  # (B, 1, N)

    def compute_std(self, x):
        """标准差 - 反映数据波动水平"""
        return x.std(dim=1, keepdim=True)  # (B, 1, N)

    def compute_skewness(self, x):
        """
        偏度 - 反映数据分布的不对称性
        Skewness = E[(X-μ)³] / σ³
        """
        mean = self.compute_mean(x)
        std = self.compute_std(x) + 1e-8
        # (B, L, N) - mean -> (B, L, N)
        diff = x - mean
        # 三阶中心矩 / std³
        skew = (diff ** 3).mean(dim=1, keepdim=True) / (std ** 3 + 1e-8)
        return skew

    def compute_kurtosis(self, x):
        """
        峰度 - 反映数据分布的尖峰程度
        Kurtosis = E[(X-μ)⁴] / σ⁴ - 3 (超额峰度)
        """
        mean = self.compute_mean(x)
        std = self.compute_std(x) + 1e-8
        diff = x - mean
        # 四阶中心矩 / std⁴ - 3
        kurt = (diff ** 4).mean(dim=1, keepdim=True) / (std ** 4 + 1e-8) - 3
        return kurt

    def compute_fft_features(self, x):
        """
        频域特征 - FFT 提取主导频率和振幅

        对每个通道的时间序列做 FFT，取前 k 个主导频率成分
        返回: (B, 2*fft_k, N) - 拼接频率和振幅
        """
        B, L, N = x.shape

        # FFT: (B, L, N) -> (B, L, N)
        # 取 L//2+1 个频率 bins (对称性)
        fft_result = torch.fft.rfft(x, dim=1)

        # 振幅谱: |X(f)|
        amplitude = torch.abs(fft_result)  # (B, L//2+1, N)

        # 频率索引 (归一化)
        freqs = torch.arange(amplitude.shape[1], device=x.device).float()  # (L//2+1,)
        freqs = freqs / L  # 归一化到 [0, 0.5]

        # 频域张量: (B, N, F)
        amplitude_T = amplitude.transpose(1, 2)  # (B, N, L//2+1)
        freqs_expanded = freqs.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # (B, N, L//2+1)

        # ========== FreqEvo 风格: 多级递归频域增强 ==========
        # 从高振幅到低振幅逐级过滤：
        # 浅层先处理稳定主频，残差频谱交给更深层细化。
        F_bins = amplitude_T.shape[-1]
        residual = amplitude_T.clone()
        aggregated = torch.zeros_like(amplitude_T)

        total_energy = amplitude_T.mean(dim=-1, keepdim=True) + 1e-8
        running_depth_weight = torch.ones(B, N, 1, device=x.device)

        for level in range(self.max_freq_levels):
            # 层级越深，单层保留的频点数越少，聚焦困难高频残差
            k_level = max(1, int(round(self.fft_k / (2 ** level))))
            k_level = min(k_level, F_bins)

            level_amp, level_idx = torch.topk(residual, k=k_level, dim=-1)  # (B, N, k_level)

            level_map = torch.zeros_like(residual)
            level_map.scatter_(-1, level_idx, level_amp)

            # 可学习深度决策: 数据驱动判断是否值得继续递归
            residual_ratio = residual.mean(dim=-1, keepdim=True) / total_energy  # (B, N, 1)
            level_ratio = torch.full_like(residual_ratio, float(level + 1) / self.max_freq_levels)
            gate_input = torch.cat([residual_ratio, level_ratio], dim=-1)  # (B, N, 2)

            need_deeper = torch.sigmoid(self.depth_gate(gate_input))  # (B, N, 1)

            # 当前层输出采用累计深度权重，后续层按 need_deeper 递减
            level_weight = running_depth_weight
            aggregated = aggregated + level_weight * level_map

            residual = residual - level_map
            running_depth_weight = running_depth_weight * need_deeper

        # 递归增强后，支持三种频域解码策略:
        # 1) hard_topk: 硬选择 top-k
        # 2) soft_select: 软选择 top-k（全频信息加权）
        # 3) band_stats: 多带宽统计后投影
        if self.fft_feature_mode == 'hard_topk':
            top_freqs, top_amplitude = self._decode_hard_topk(aggregated, freqs_expanded)
        elif self.fft_feature_mode == 'soft_select':
            top_freqs, top_amplitude = self._decode_soft_select(aggregated, freqs_expanded)
        elif self.fft_feature_mode == 'band_stats':
            top_freqs, top_amplitude = self._decode_band_stats(aggregated, freqs_expanded)
        else:
            raise ValueError(
                f"Unsupported fft_feature_mode: {self.fft_feature_mode}. "
                f"Choose from ['hard_topk', 'soft_select', 'band_stats']."
            )

        # 归一化振幅 (相对于最大振幅)
        max_amp = amplitude_T.max(dim=-1, keepdim=True)[0] + 1e-8
        top_amplitude_norm = top_amplitude / (max_amp + 1e-8)

        # 若频率 bins 数量不足 fft_k，补零保证特征维度恒定
        top_k = top_freqs.shape[-1]
        if top_k < self.fft_k:
            pad_k = self.fft_k - top_k
            top_freqs = F.pad(top_freqs, (0, pad_k), mode='constant', value=0.0)
            top_amplitude_norm = F.pad(top_amplitude_norm, (0, pad_k), mode='constant', value=0.0)

        # 按通道拼接频率与振幅: (B, N, 2*fft_k) -> (B, 2*fft_k, N)
        features = torch.cat([top_freqs, top_amplitude_norm], dim=-1)
        return features.transpose(1, 2).contiguous()

    def compute_autocorr_features(self, x):
        """
        自相关特征 - 反映数据的"记忆性"

        计算不同延迟下的自相关系数
        lag τ 处的自相关: r(τ) = Cov(X_t, X_{t+τ}) / Var(X_t)

        返回: (B, n_autocorr, N)
        """
        B, L, N = x.shape

        # 标准化: (X - mean) / std
        mean = self.compute_mean(x)  # (B, 1, N)
        std = self.compute_std(x) + 1e-8  # (B, 1, N)
        x_norm = (x - mean) / std  # (B, L, N)

        autocorr_features = []
        selected_lags = self._select_autocorr_lags(x, L)

        for lag in selected_lags:
            if lag < L:
                # X_t 和 X_{t+lag} 的相关系数
                # Cov = E[X_t * X_{t+lag}]
                cov = (x_norm[:, :-lag, :] * x_norm[:, lag:, :]).mean(dim=1)  # (B, N)
                autocorr_features.append(cov.unsqueeze(1))  # (B, 1, N)

        # 保证输出维度固定为 n_autocorr
        if len(autocorr_features) == 0:
            autocorr_features.append(torch.zeros(B, 1, N, device=x.device))

        if len(autocorr_features) < self.n_autocorr:
            pad_num = self.n_autocorr - len(autocorr_features)
            autocorr_features.extend([
                torch.zeros(B, 1, N, device=x.device) for _ in range(pad_num)
            ])
        elif len(autocorr_features) > self.n_autocorr:
            autocorr_features = autocorr_features[:self.n_autocorr]

        autocorr = torch.cat(autocorr_features, dim=1)  # (B, n_autocorr, N)
        return autocorr

    def forward(self, x):
        """
        输入: x (B, L, N) - 原始时间序列
        输出: stats (B, N, feat_dim) - 每个通道的统计特征
        """
        B, L, N = x.shape

        # 1. 基础统计特征 (4 维)
        mean = self.compute_mean(x)                    # (B, 1, N)
        std = self.compute_std(x)                      # (B, 1, N)
        skewness = self.compute_skewness(x)            # (B, 1, N)
        kurtosis = self.compute_kurtosis(x)            # (B, 1, N)

        basic_stats = torch.cat([mean, std, skewness, kurtosis], dim=1)  # (B, 4, N)

        # 2. 频域特征 (2*fft_k 维)
        fft_features = self.compute_fft_features(x)    # (B, 2*fft_k, N)

        # 3. 自相关特征 (n_autocorr 维)
        autocorr = self.compute_autocorr_features(x)   # (B, n_autocorr, N)

        # 拼接所有特征: (B, 4 + 2*fft_k + n_autocorr, N)
        features = torch.cat([basic_stats, fft_features, autocorr], dim=1)

        # 调整维度顺序: (B, N, feat_dim)
        features = features.transpose(1, 2)  # (B, N, feat_dim)

        return features


class DynamicChannelEmbedding(nn.Module):
    """
    动态通道嵌入层

    从输入数据动态提取统计特征，替代静态可学习的 channel_embedding

    创新点：
    1. 数据驱动 - 嵌入从数据统计中提取，而非固定参数
    2. 零样本能力 - 新通道可通过统计特征匹配已有模式
    3. 跨域迁移 - 统计特征具有领域不变性
    """

    def __init__(
            self,
            fft_k=5,
            autocorr_lags=[1, 3, 6, 12, 24],
            autocorr_mode='fixed',
            max_freq_levels=4,
            fft_feature_mode='hard_topk',
            n_freq_bands=4,
            soft_select_temp=0.15):
        super(DynamicChannelEmbedding, self).__init__()

        self.stat_features = StatisticalFeatures(
            fft_k=fft_k,
            autocorr_lags=autocorr_lags,
            autocorr_mode=autocorr_mode,
            max_freq_levels=max_freq_levels,
            fft_feature_mode=fft_feature_mode,
            n_freq_bands=n_freq_bands,
            soft_select_temp=soft_select_temp
        )

        # 计算特征维度
        n_basic = 4  # mean, std, skew, kurtosis
        n_fft = 2 * fft_k  # freq + amplitude for each of k frequencies
        n_autocorr = len(autocorr_lags)
        self.feat_dim = n_basic + n_fft + n_autocorr

    def forward(self, x):
        """
        输入: x (B, L, N) - 原始时间序列
        输出: channel_emb (B, N, E) - 动态通道嵌入
        """
        # 提取统计特征: (B, N, feat_dim)
        stats = self.stat_features(x)
        return stats


class DynamicPhaseEmbedding(nn.Module):
    """
    动态相位嵌入层

    根据数据自身的周期特性动态计算相位嵌入，而非使用固定查表

    创新点：
    1. 数据驱动 - 从数据中检测真实周期
    2. 自适应 - 不同数据集自动适应不同周期
    """

    def __init__(self, d_model, n_phases=24):
        super(DynamicPhaseEmbedding, self).__init__()
        self.d_model = d_model
        self.n_phases = n_phases

        # 可学习的相位编码器 (但初始化为周期模式)
        self.phase_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 输入: [sin(2πt/T), cos(2πt/T), t/T]
            nn.GELU(),
            nn.Linear(16, d_model)
        )

    def forward(self, batch_size, n_channels, phase_indices, device):
        """
        输入:
            batch_size: B
            n_channels: N
            phase_indices: (B, N) 或标量 - 当前相位索引
            device: 计算设备

        输出: phase_emb (B, N, d_model)
        """
        # 将相位输入统一为归一化相位 [0, 1]
        # 兼容两种输入:
        # 1) 离散相位索引 [0, n_phases)
        # 2) 连续相位比例 [0, 1]
        if isinstance(phase_indices, torch.Tensor):
            phase = phase_indices.float().to(device)
            if phase.dim() == 0:
                phase = phase.view(1, 1).expand(batch_size, n_channels)
            elif phase.dim() == 1:
                phase = phase.unsqueeze(-1).expand(-1, n_channels)
            elif phase.dim() == 2 and phase.shape[1] == 1:
                phase = phase.expand(-1, n_channels)

            if phase.max() > 1.0 or phase.min() < 0.0:
                phase = (phase % self.n_phases) / self.n_phases
            else:
                phase = phase.clamp(0.0, 1.0)
        else:
            phase = torch.tensor(phase_indices, device=device).float()
            if phase > 1.0 or phase < 0.0:
                phase = (phase % self.n_phases) / self.n_phases
            phase = phase.view(1, 1).expand(batch_size, n_channels)

        # 计算正弦/余弦编码
        t = phase * 2 * pi  # 归一化到 2π，使用 math.pi 而非硬编码常量
        sin_t = torch.sin(t).unsqueeze(-1)    # (B, N, 1)
        cos_t = torch.cos(t).unsqueeze(-1)    # (B, N, 1)
        normalized_t = phase.unsqueeze(-1)    # (B, N, 1)

        # 拼接: (B, N, 3)
        phase_input = torch.cat([sin_t, cos_t, normalized_t], dim=-1)

        # 编码到 d_model: (B, N, d_model)
        phase_emb = self.phase_encoder(phase_input)

        return phase_emb


class DynamicJointEmbedding(nn.Module):
    """
    动态联合通道-相位嵌入

    根据通道的统计特征和相位信息联合计算嵌入

    创新点：
    1. 通道-相位交互建模
    2. 统计感知的相位编码
    """

    def __init__(self, d_model, n_phases=24, stat_feat_dim=None):
        super(DynamicJointEmbedding, self).__init__()
        self.d_model = d_model
        self.n_phases = n_phases

        # 统计特征 -> d_model 的投影
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_feat_dim, d_model),
            nn.GELU()
        ) if stat_feat_dim else None

        # 相位编码
        self.phase_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU()
        )

        # 联合交互
        self.joint_proj = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU()
        )

    def forward(self, stats, batch_size, n_channels, phase_indices, device):
        """
        输入:
            stats: (B, N, stat_feat_dim) - 统计特征
            batch_size: B
            n_channels: N
            phase_indices: 相位索引
            device: 计算设备

        输出: joint_emb (B, N, d_model)
        """
        # 统计特征投影
        if self.stat_proj is not None:
            stat_emb = self.stat_proj(stats)  # (B, N, d_model)
        else:
            stat_emb = torch.zeros(batch_size, n_channels, self.d_model, device=device)

        # 相位编码，兼容离散索引与连续相位比例
        if isinstance(phase_indices, torch.Tensor):
            phase = phase_indices.float().to(device)
            if phase.dim() == 0:
                phase = phase.view(1, 1).expand(batch_size, n_channels)
            elif phase.dim() == 1:
                phase = phase.unsqueeze(-1).expand(-1, n_channels)
            elif phase.dim() == 2 and phase.shape[1] == 1:
                phase = phase.expand(-1, n_channels)

            if phase.max() > 1.0 or phase.min() < 0.0:
                phase = (phase % self.n_phases) / self.n_phases
            else:
                phase = phase.clamp(0.0, 1.0)
        else:
            phase = torch.tensor(phase_indices, device=device).float()
            if phase > 1.0 or phase < 0.0:
                phase = (phase % self.n_phases) / self.n_phases
            phase = phase.view(1, 1).expand(batch_size, n_channels)

        # 【修复】使用 math.pi 替代硬编码 3.14159265，与 DynamicPhaseEmbedding 保持一致
        t = phase * 2 * pi
        sin_t = torch.sin(t).unsqueeze(-1)
        cos_t = torch.cos(t).unsqueeze(-1)
        normalized_t = phase.unsqueeze(-1)

        phase_input = torch.cat([sin_t, cos_t, normalized_t], dim=-1)
        phase_emb = self.phase_encoder(phase_input)  # (B, N, d_model//2)

        # 联合
        joint = torch.cat([stat_emb, phase_emb], dim=-1)  # (B, N, d_model + d_model//2)
        joint_emb = self.joint_proj(joint)  # (B, N, d_model)

        return joint_emb


class DynamicEmbeddingArmor(nn.Module):
    """
    动态嵌入护甲 (Dynamic Embedding Armor)

    整合三种动态嵌入：
    1. 动态通道嵌入 - 数据驱动的统计特征
    2. 动态相位嵌入 - 自适应周期编码
    3. 动态联合嵌入 - 通道-相位交互

    这是 EMAformer 的升级版：静态 -> 动态
    """

    def __init__(
            self,
            d_model,
            n_phases=24,
            fft_k=5,
            autocorr_lags=[1, 3, 6, 12, 24],
            autocorr_mode='fixed',
            max_freq_levels=4,
            fft_feature_mode='hard_topk',
            n_freq_bands=4,
            soft_select_temp=0.15):
        super(DynamicEmbeddingArmor, self).__init__()

        self.d_model = d_model
        self.n_phases = n_phases

        # 动态通道嵌入
        self.channel_embed = DynamicChannelEmbedding(
            fft_k=fft_k,
            autocorr_lags=autocorr_lags,
            autocorr_mode=autocorr_mode,
            max_freq_levels=max_freq_levels,
            fft_feature_mode=fft_feature_mode,
            n_freq_bands=n_freq_bands,
            soft_select_temp=soft_select_temp
        )
        self.stat_proj = nn.Linear(
            self.channel_embed.feat_dim,  # 统计特征维度
            d_model
        )

        # 动态相位嵌入
        self.phase_embed = DynamicPhaseEmbedding(
            d_model=d_model,
            n_phases=n_phases
        )

        # 动态联合嵌入
        self.joint_embed = DynamicJointEmbedding(
            d_model=d_model,
            n_phases=n_phases,
            stat_feat_dim=self.channel_embed.feat_dim
        )

    def _estimate_dominant_period(self, x):
        """
        基于输入序列自动估计每个样本的主导周期。
        TODO 局限是只取单峰主频，遇到多周期并存时可能不够全面。

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
        TODO 可以把周期估计改成通道级别的，这样相位构建就更细粒度了。
        """
        B, _, N = x.shape
        t = torch.tensor(float(x.shape[1] - 1), device=x.device).expand(B)
        phase_ratio = torch.remainder(t, periods) / (periods + 1e-8)  # (B,)
        return phase_ratio.unsqueeze(1).expand(-1, N)  # (B, N)

    def forward(self, x, phase_indices=None):
        """
        输入:
            x: (B, L, N) - 原始时间序列
            phase_indices: (B, N) 或标量 - 相位索引

        输出:
            embeddings: dict with channel_emb, phase_emb, joint_emb
        """
        B, L, N = x.shape

        # 1. 动态通道嵌入
        stats = self.channel_embed(x)  # (B, N, feat_dim)
        channel_emb = self.stat_proj(stats)  # (B, N, d_model)

        # 2. 动态相位嵌入
        estimated_period = None
        if phase_indices is None:
            # Continuous Frequency Encoding:
            # 自动估计主导周期并构造连续相位比例
            estimated_period = self._estimate_dominant_period(x)          # (B,)
            phase_indices = self._build_continuous_phase(x, estimated_period)  # (B, N), in [0, 1)

        phase_emb = self.phase_embed(B, N, phase_indices, x.device)

        # 3. 动态联合嵌入
        joint_emb = self.joint_embed(stats, B, N, phase_indices, x.device)

        return {
            'channel_emb': channel_emb,
            'phase_emb': phase_emb,
            'joint_emb': joint_emb,
            'stats': stats,  # 原始统计特征，可用于其他目的
            'estimated_period': estimated_period  # 自动估计周期，仅在 phase_indices=None 时非空
        }