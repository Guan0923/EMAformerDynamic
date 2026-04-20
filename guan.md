# EMAformerDynamic 技术说明书 V3.0

## 目录
1. [项目概述](#项目概述)
2. [核心架构](#核心架构)
3. [layers/DynamicEmbedding.py 详解](#layersdynamicembeddingpy-详解)
4. [model/EMAformerDynamic.py 详解](#modelemaformerdynamicpy-详解)
5. [数据流与计算流程](#数据流与计算流程)
6. [创新点与优势](#创新点与优势)
7. [使用指南](#使用指南)
8. [配置参数说明](#配置参数说明)

---

## 项目概述

**EMAformerDynamic** 是 EMAformer 的升级版本，核心创新在于将**静态可学习嵌入**替换为**数据驱动的动态统计特征嵌入**。最新版本引入了**三种频域解码策略**和**FreqEvo 多级递归频域增强**，使模型具备更强的频谱建模能力。

### 核心能力
- **Zero-Shot 跨域迁移**：新通道无需重新训练
- **三种频域解码策略**：hard_topk / soft_select / band_stats
- **FreqEvo 多级递归**：自适应分析深度
- **自动周期估计**：无需预设周期长度

---

## 核心架构

### 整体结构

```
输入: x_enc (B, L, N) + x_mark_enc + phase(可选)
        ↓
┌─────────────────────────────────────────┐
│  DataEmbedding_inverted                 │
│  (B,L,N) → (B,N,E)                      │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  DynamicEmbeddingArmor                  │
│  - StatisticalFeatures:                 │
│    - 基础统计: 均值/标准差/偏度/峰度    │
│    - 多级FFT: FreqEvo递归增强           │
│    - 三种解码: hard/soft/band_stats     │
│    - 自相关: 多延迟记忆性               │
│  - DynamicChannelEmbedding              │
│  - DynamicPhaseEmbedding                │
│  - DynamicJointEmbedding                │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  Encoder (Transformer)                  │
│  - Multi-head Self-Attention            │
│  - Feed-Forward Network                 │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  Projector                              │
│  (B,N,E) → (B,N,pred_len) → (B,pred_len,N)│
└─────────────────────────────────────────┘
        ↓
输出: 预测序列 (B, pred_len, N)
```

### 统计特征维度

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| 基础统计 | 4 | 均值、标准差、偏度、峰度 |
| FFT频域 | 2×k | k个频率 + k个振幅（三种解码策略）|
| 自相关 | m | m个延迟下的自相关系数 |
| **总计** | **4+2k+m** | 默认 k=5, m=5，共 19 维 |

---

## layers/DynamicEmbedding.py 详解

该文件实现了**动态嵌入护甲**的所有组件，共包含 5 个核心类。

### 1. StatisticalFeatures 类

**功能**：从原始时间序列中提取多维度统计特征

#### 初始化参数（V3.0 扩展）

```python
def __init__(
    self,
    fft_k=5,                           # 最终保留的主导频率数
    autocorr_lags=[1, 3, 6, 12, 24],   # 自相关延迟列表
    max_freq_levels=4,                 # FreqEvo递归层级数
    fft_feature_mode='hard_topk',      # 频域解码策略
    n_freq_bands=4,                    # 频带统计模式的频带数
    soft_select_temp=0.15              # 软选择温度参数
):
```

**新增参数详解**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fft_feature_mode` | str | 'hard_topk' | 频域解码策略：'hard_topk'/'soft_select'/'band_stats' |
| `n_freq_bands` | int | 4 | band_stats 模式下的频带数量 |
| `soft_select_temp` | float | 0.15 | soft_select 模式的温度参数（越小越硬）|
| `max_freq_levels` | int | 4 | FreqEvo 递归分析的层级数 |

#### 新增组件

```python
# 递归频域深度门控网络
self.depth_gate = nn.Sequential(
    nn.Linear(2, 8),
    nn.GELU(),
    nn.Linear(8, 1)
)

# 频带统计投影网络（仅 band_stats 模式使用）
self.band_proj = nn.Sequential(
    nn.Linear(2 * n_freq_bands, 2 * fft_k),
    nn.GELU(),
    nn.Linear(2 * fft_k, 2 * fft_k)
)
```

#### 三种频域解码策略详解

##### 策略 1: _decode_hard_topk（硬选择）

**原理**：直接从聚合频谱中选择振幅最大的 k 个频率

```python
def _decode_hard_topk(self, spectrum, freqs_expanded):
    """
    输入:
        spectrum: (B, N, F) - 聚合后的振幅谱
        freqs_expanded: (B, N, F) - 频率索引
    输出:
        top_freqs: (B, N, k) - 选中的频率
        top_amplitude: (B, N, k) - 对应的振幅
    """
    top_k = min(self.fft_k, spectrum.shape[-1])
    top_amplitude, top_idx = torch.topk(spectrum, k=top_k, dim=-1)
    top_freqs = torch.gather(freqs_expanded, dim=-1, index=top_idx)
    return top_freqs, top_amplitude
```

**特点**：
- ✓ 简单高效，无额外参数
- ✓ 可解释性强
- ✗ 不可微分（梯度无法通过索引传播）

##### 策略 2: _decode_soft_select（软选择）

**原理**：通过可微分的概率加权迭代提取频率，所有频率都参与计算

```python
def _decode_soft_select(self, spectrum, freqs_expanded):
    """
    软选择 top-k：通过概率加权迭代提取 k 组频率-振幅对
    """
    work = spectrum.clone()
    soft_freqs, soft_amps = [], []
    
    top_k = min(self.fft_k, spectrum.shape[-1])
    temp = max(self.soft_select_temp, 1e-4)  # 防止除零
    
    for _ in range(top_k):
        # 计算每个频率的选择概率
        weights = torch.softmax(work / temp, dim=-1)  # (B, N, F)
        
        # 加权求和得到"软"频率和振幅
        sel_freq = (weights * freqs_expanded).sum(dim=-1, keepdim=True)
        sel_amp = (weights * work).sum(dim=-1, keepdim=True)
        
        soft_freqs.append(sel_freq)
        soft_amps.append(sel_amp)
        
        # 抑制已选区域（类似注意力掩码）
        work = work * (1.0 - weights)
    
    top_freqs = torch.cat(soft_freqs, dim=-1)       # (B, N, k)
    top_amplitude = torch.cat(soft_amps, dim=-1)    # (B, N, k)
    return top_freqs, top_amplitude
```

**软选择机制详解**：

```
迭代过程示例（k=3, 温度=0.15）:

第1次迭代:
  work = 原始频谱
  weights = softmax(work / 0.15)
  → 高振幅频率获得高权重
  sel_freq_1 = Σ(weights[i] * freq[i])
  work = work * (1 - weights)  # 抑制已选区域

第2次迭代:
  work = 被抑制后的频谱（主峰已被削弱）
  weights = softmax(work / 0.15)
  → 次峰获得高权重
  sel_freq_2 = Σ(weights[i] * freq[i])

第3次迭代:
  work = 再次抑制
  → 第三峰获得高权重
  sel_freq_3 = Σ(weights[i] * freq[i])
```

**温度参数的影响**：

| 温度值 | 行为 | 适用场景 |
|--------|------|----------|
| temp → 0 | 接近硬选择（one-hot）| 需要明确峰值时 |
| temp = 0.15 | 平衡（默认）| 一般情况 |
| temp → ∞ | 均匀分布 | 需要平滑特征时 |

**特点**：
- ✓ 完全可微分，支持端到端训练
- ✓ 保留所有频率信息（软加权）
- ✓ 通过迭代机制逐步挖掘次要频率
- ✗ 计算量稍大（k次softmax）

##### 策略 3: _decode_band_stats（频带统计）

**原理**：将频谱划分为多个频带，提取每个频带的统计特征，再投影到固定维度

```python
def _decode_band_stats(self, spectrum, freqs_expanded):
    """
    多带宽统计：先提取分带统计，再投影为固定 2*fft_k 维输出
    """
    B, N, F_bins = spectrum.shape
    band_feats = []
    
    for b in range(self.n_freq_bands):
        # 划分频带
        start = int(round(b * F_bins / self.n_freq_bands))
        end = int(round((b + 1) * F_bins / self.n_freq_bands))
        
        band_amp = spectrum[:, :, start:end]   # (B, N, f_b)
        band_freq = freqs_expanded[:, :, start:end]
        
        # 计算频带统计特征
        mean_energy = band_amp.mean(dim=-1, keepdim=True)   # 平均能量
        centroid = (band_amp * band_freq).sum(dim=-1, keepdim=True) / (band_amp.sum(dim=-1, keepdim=True) + 1e-8)  # 频带质心
        
        band_feats.extend([mean_energy, centroid])
    
    # 拼接所有频带统计
    band_feats = torch.cat(band_feats, dim=-1)  # (B, N, 2*n_bands)
    
    # 投影到固定维度
    proj = self.band_proj(band_feats)  # (B, N, 2*fft_k)
    
    # 分离频率和振幅
    freq_logits = proj[:, :, :self.fft_k]
    amp_logits = proj[:, :, self.fft_k:]
    
    # 归一化输出
    top_freqs = 0.5 * torch.sigmoid(freq_logits)      # [0, 0.5]
    top_amplitude = F.softplus(amp_logits)            # 非负
    return top_freqs, top_amplitude
```

**频带划分示例（n_freq_bands=4）**：

```
频谱范围: [0, 0.5] (归一化频率)

Band 0: [0.00, 0.125)  - 低频（趋势/长期模式）
Band 1: [0.125, 0.25)  - 中低频（周期性）
Band 2: [0.25, 0.375) - 中高频（短期波动）
Band 3: [0.375, 0.50) - 高频（噪声/细节）

每个频带提取:
  - mean_energy: 该频带的平均能量（重要性）
  - centroid: 该频带的质心频率（该频带内能量集中位置）
```

**特点**：
- ✓ 捕捉频谱的整体结构而非单个峰值
- ✓ 对不同频率范围的特征都有响应
- ✓ 通过可学习投影适配不同数据集
- ✗ 丢失精确的峰值位置信息

#### 三种策略对比

| 特性 | hard_topk | soft_select | band_stats |
|------|-----------|-------------|------------|
| **可微性** | ✗ | ✓ | ✓ |
| **计算效率** | ★★★ | ★★☆ | ★★☆ |
| **峰值精度** | ★★★ | ★★☆ | ★☆☆ |
| **频谱完整性** | ★☆☆ | ★★★ | ★★★ |
| **端到端训练** | 部分 | 完全 | 完全 |
| **适用场景** | 传统方法 | 需要梯度回传 | 关注频谱结构 |

#### 核心方法详解

##### 1.1 compute_mean / compute_std / compute_skewness / compute_kurtosis
与之前版本相同，提取基础统计特征（4维）。

##### 1.2 compute_fft_features（核心更新）

**FreqEvo 多级递归 + 三种解码策略**

```python
def compute_fft_features(self, x):
    B, L, N = x.shape
    
    # FFT变换
    fft_result = torch.fft.rfft(x, dim=1)
    amplitude = torch.abs(fft_result)      # 振幅谱
    freqs = torch.arange(F) / L            # 归一化频率 [0, 0.5]
    
    # 转置为 (B, N, F) 便于处理
    amplitude_T = amplitude.transpose(1, 2)
    freqs_expanded = freqs.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
    
    # ========== FreqEvo: 多级递归频域增强 ==========
    residual = amplitude_T.clone()         # 初始残差
    aggregated = torch.zeros_like(amplitude_T)
    total_energy = amplitude_T.mean(dim=-1, keepdim=True) + 1e-8
    running_depth_weight = torch.ones(B, N, 1)
    
    for level in range(self.max_freq_levels):
        # 层级越深，提取的频点数越少
        k_level = max(1, int(round(self.fft_k / (2 ** level))))
        
        # 在当前残差中提取 top-k
        level_amp, level_idx = torch.topk(residual, k=k_level, dim=-1)
        
        # 构建当前层频谱映射
        level_map = torch.zeros_like(residual)
        level_map.scatter_(-1, level_idx, level_amp)
        
        # 深度门控: 判断是否值得继续深入
        residual_ratio = residual.mean(dim=-1, keepdim=True) / total_energy
        level_ratio = torch.full_like(residual_ratio, float(level + 1) / self.max_freq_levels)
        gate_input = torch.cat([residual_ratio, level_ratio], dim=-1)
        need_deeper = torch.sigmoid(self.depth_gate(gate_input))
        
        # 加权聚合
        level_weight = running_depth_weight
        aggregated = aggregated + level_weight * level_map
        
        # 更新残差和权重
        residual = residual - level_map
        running_depth_weight = running_depth_weight * need_deeper
    
    # ========== 三种解码策略选择 ==========
    if self.fft_feature_mode == 'hard_topk':
        top_freqs, top_amplitude = self._decode_hard_topk(aggregated, freqs_expanded)
    elif self.fft_feature_mode == 'soft_select':
        top_freqs, top_amplitude = self._decode_soft_select(aggregated, freqs_expanded)
    elif self.fft_feature_mode == 'band_stats':
        top_freqs, top_amplitude = self._decode_band_stats(aggregated, freqs_expanded)
    else:
        raise ValueError(f"Unsupported fft_feature_mode: {self.fft_feature_mode}")
    
    # 归一化和输出
    max_amp = amplitude_T.max(dim=-1, keepdim=True)[0] + 1e-8
    top_amplitude_norm = top_amplitude / max_amp
    
    # 维度补齐（若 F_bins < fft_k）
    if top_freqs.shape[-1] < self.fft_k:
        pad_k = self.fft_k - top_freqs.shape[-1]
        top_freqs = F.pad(top_freqs, (0, pad_k), value=0.0)
        top_amplitude_norm = F.pad(top_amplitude_norm, (0, pad_k), value=0.0)
    
    features = torch.cat([top_freqs, top_amplitude_norm], dim=-1)
    return features.transpose(1, 2).contiguous()  # (B, 2*fft_k, N)
```

---

### 2. DynamicChannelEmbedding 类

**V3.0 更新**：支持传递所有 StatisticalFeatures 参数

```python
def __init__(
    self,
    fft_k=5,
    autocorr_lags=[1, 3, 6, 12, 24],
    max_freq_levels=4,
    fft_feature_mode='hard_topk',      # 新增
    n_freq_bands=4,                    # 新增
    soft_select_temp=0.15              # 新增
):
    self.stat_features = StatisticalFeatures(
        fft_k=fft_k,
        autocorr_lags=autocorr_lags,
        max_freq_levels=max_freq_levels,
        fft_feature_mode=fft_feature_mode,      # 传递
        n_freq_bands=n_freq_bands,              # 传递
        soft_select_temp=soft_select_temp       # 传递
    )
```

---

### 3. DynamicEmbeddingArmor 类

**V3.0 更新**：支持所有新参数

```python
def __init__(
    self,
    d_model,
    n_phases=24,
    fft_k=5,
    autocorr_lags=[1, 3, 6, 12, 24],
    max_freq_levels=4,                 # 新增
    fft_feature_mode='hard_topk',      # 新增
    n_freq_bands=4,                    # 新增
    soft_select_temp=0.15              # 新增
):
    self.channel_embed = DynamicChannelEmbedding(
        fft_k=fft_k,
        autocorr_lags=autocorr_lags,
        max_freq_levels=max_freq_levels,
        fft_feature_mode=fft_feature_mode,
        n_freq_bands=n_freq_bands,
        soft_select_temp=soft_select_temp
    )
```

---

## model/EMAformerDynamic.py 详解

三个模型变体与之前版本基本一致，但配置对象可以包含新的频域参数：

```python
class Config:
    # 基础参数
    seq_len = 96
    pred_len = 96
    d_model = 256
    
    # 频域参数（V3.0 新增）
    fft_k = 5
    max_freq_levels = 4
    fft_feature_mode = 'soft_select'  # 'hard_topk' / 'soft_select' / 'band_stats'
    n_freq_bands = 4
    soft_select_temp = 0.15
```

---

## 配置参数说明

### StatisticalFeatures 完整参数表

| 参数 | 类型 | 默认值 | 说明 | 推荐设置 |
|------|------|--------|------|----------|
| `fft_k` | int | 5 | 最终保留的频率-振幅对数 | 3-10 |
| `autocorr_lags` | list | [1,3,6,12,24] | 自相关延迟 | 根据周期调整 |
| `max_freq_levels` | int | 4 | FreqEvo递归层级 | 3-5 |
| `fft_feature_mode` | str | 'hard_topk' | 频域解码策略 | 见下方 |
| `n_freq_bands` | int | 4 | band_stats频带数 | 4-8 |
| `soft_select_temp` | float | 0.15 | 软选择温度 | 0.1-0.5 |

### 频域模式选择指南

```python
# 场景1: 需要端到端训练 + 关注精确峰值
fft_feature_mode='soft_select'
soft_select_temp=0.15

# 场景2: 需要端到端训练 + 关注频谱结构
fft_feature_mode='band_stats'
n_freq_bands=6

# 场景3: 传统方法 + 高效率
fft_feature_mode='hard_topk'

# 场景4: 复杂信号 + 深度分析
fft_feature_mode='soft_select'
max_freq_levels=5
soft_select_temp=0.1
```

---

## 使用示例

### 示例1: 使用软选择模式

```python
from model.EMAformerDynamic import Model

class Config:
    seq_len = 96
    pred_len = 96
    d_model = 256
    # 频域配置
    fft_k = 5
    max_freq_levels = 4
    fft_feature_mode = 'soft_select'  # 使用软选择
    soft_select_temp = 0.15

config = Config()
model = Model(config)

# 前向传播
dec_out, attns, stats = model(x_enc, x_mark_enc, x_dec, x_mark_dec, phase)
```

### 示例2: 使用频带统计模式

```python
class Config:
    seq_len = 96
    pred_len = 96
    d_model = 256
    # 频域配置
    fft_k = 5
    fft_feature_mode = 'band_stats'
    n_freq_bands = 6  # 6个频带

config = Config()
model = Model(config)
```

### 示例3: 对比三种模式

```python
import torch
from layers.DynamicEmbedding import StatisticalFeatures

# 测试数据
x = torch.randn(2, 96, 7)  # (B, L, N)

# 三种模式对比
modes = ['hard_topk', 'soft_select', 'band_stats']
for mode in modes:
    stat_feat = StatisticalFeatures(
        fft_k=5,
        fft_feature_mode=mode,
        soft_select_temp=0.15 if mode == 'soft_select' else None,
        n_freq_bands=4 if mode == 'band_stats' else None
    )
    features = stat_feat(x)
    print(f"{mode}: {features.shape}")  # 都是 (2, 7, 19)
```

---

## 创新点总结

### V3.0 核心创新

1. **三种频域解码策略**
   - hard_topk: 传统硬选择
   - soft_select: 可微分软选择
   - band_stats: 频带统计

2. **FreqEvo 多级递归**
   - 自适应分析深度
   - 门控网络动态决策

3. **完全端到端训练**
   - soft_select 和 band_stats 完全可微
   - 支持梯度回传优化

---

**文档版本**: 3.0  
**更新内容**: 新增三种频域解码策略完整说明  
**最后更新**: 2026年  
**作者**: Sisyphus
