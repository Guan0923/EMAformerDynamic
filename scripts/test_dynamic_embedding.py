"""
测试动态嵌入护甲 - EMAformerDynamic

运行方式:
    python scripts/test_dynamic_embedding.py

这个脚本验证:
1. 动态统计特征提取是否正确
2. 动态嵌入与静态嵌入的维度是否匹配
3. 模型能否正常前向传播
"""

import torch
import sys
sys.path.insert(0, '.')

from layers.DynamicEmbedding import (
    StatisticalFeatures,
    DynamicChannelEmbedding,
    DynamicPhaseEmbedding,
    DynamicJointEmbedding,
    DynamicEmbeddingArmor
)
from model.EMAformerDynamic import Model, EMAformerDynamicZeroShot


def test_statistical_features():
    """测试统计特征提取"""
    print("=" * 60)
    print("测试 1: 统计特征提取")
    print("=" * 60)

    # 模拟数据: (B=4, L=96, N=7)
    B, L, N = 4, 96, 7
    x = torch.randn(B, L, N)

    # 测试不同配置
    configs = [
        {"fft_k": 3, "autocorr_lags": [1, 3, 6]},
        {"fft_k": 5, "autocorr_lags": [1, 3, 6, 12, 24]},
    ]

    for cfg in configs:
        stat_feat = StatisticalFeatures(
            fft_k=cfg["fft_k"],
            autocorr_lags=cfg["autocorr_lags"]
        )

        features = stat_feat(x)
        print(f"\n配置: fft_k={cfg['fft_k']}, autocorr_lags={cfg['autocorr_lags']}")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {features.shape}")
        print(f"  特征维度: {stat_feat.feat_dim}")

        # 验证形状
        assert features.shape == (B, N, stat_feat.feat_dim), "形状不匹配!"
        print("  ✓ 形状验证通过")

        # 检查特征值是否合理 (不是 NaN 或 Inf)
        assert not torch.isnan(features).any(), "包含 NaN!"
        assert not torch.isinf(features).any(), "包含 Inf!"
        print("  ✓ 数值验证通过 (无 NaN/Inf)")

    print("\n测试 1 通过!\n")


def test_dynamic_channel_embedding():
    """测试动态通道嵌入"""
    print("=" * 60)
    print("测试 2: 动态通道嵌入")
    print("=" * 60)

    B, L, N = 4, 96, 7
    x = torch.randn(B, L, N)

    dce = DynamicChannelEmbedding(fft_k=5, autocorr_lags=[1, 3, 6, 12, 24])
    output = dce(x)

    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  统计特征维度: {dce.feat_dim}")

    assert output.shape == (B, N, dce.feat_dim)
    assert not torch.isnan(output).any()
    print("  ✓ 动态通道嵌入测试通过\n")


def test_dynamic_phase_embedding():
    """测试动态相位嵌入"""
    print("=" * 60)
    print("测试 3: 动态相位嵌入")
    print("=" * 60)

    B, N = 4, 7
    d_model = 512
    phase_indices = torch.randint(0, 24, (B,))

    dpe = DynamicPhaseEmbedding(d_model=d_model, n_phases=24)
    output = dpe(B, N, phase_indices, x.device)

    print(f"  相位索引形状: {phase_indices.shape}")
    print(f"  输出形状: {output.shape}")

    assert output.shape == (B, N, d_model)
    assert not torch.isnan(output).any()
    print("  ✓ 动态相位嵌入测试通过\n")


def test_dynamic_joint_embedding():
    """测试动态联合嵌入"""
    print("=" * 60)
    print("测试 4: 动态联合嵌入")
    print("=" * 60)

    B, L, N = 4, 96, 7
    d_model = 512
    stat_dim = 15  # 4 + 2*5 + 5 = 15

    x = torch.randn(B, L, N)
    phase_indices = torch.randint(0, 24, (B,))

    dje = DynamicJointEmbedding(
        d_model=d_model,
        n_phases=24,
        stat_feat_dim=stat_dim
    )

    stats = torch.randn(B, N, stat_dim)
    output = dje(stats, B, N, phase_indices, x.device)

    print(f"  统计特征输入形状: {stats.shape}")
    print(f"  输出形状: {output.shape}")

    assert output.shape == (B, N, d_model)
    assert not torch.isnan(output).any()
    print("  ✓ 动态联合嵌入测试通过\n")


def test_dynamic_embedding_armor():
    """测试完整的动态嵌入护甲"""
    print("=" * 60)
    print("测试 5: 动态嵌入护甲 (完整)")
    print("=" * 60)

    B, L, N = 4, 96, 7
    d_model = 512

    x = torch.randn(B, L, N)
    phase_indices = torch.randint(0, 24, (B,))

    dea = DynamicEmbeddingArmor(
        d_model=d_model,
        n_phases=24,
        fft_k=5,
        autocorr_lags=[1, 3, 6, 12, 24]
    )

    embeddings = dea(x, phase_indices)

    print(f"  输入形状: {x.shape}")
    print(f"  channel_emb 形状: {embeddings['channel_emb'].shape}")
    print(f"  phase_emb 形状: {embeddings['phase_emb'].shape}")
    print(f"  joint_emb 形状: {embeddings['joint_emb'].shape}")
    print(f"  stats 形状: {embeddings['stats'].shape}")

    assert embeddings['channel_emb'].shape == (B, N, d_model)
    assert embeddings['phase_emb'].shape == (B, N, d_model)
    assert embeddings['joint_emb'].shape == (B, N, d_model)
    assert embeddings['stats'].shape[0] == B and embeddings['stats'].shape[2] == N

    print("  ✓ 动态嵌入护甲测试通过\n")


def test_emaformer_dynamic_model():
    """测试完整的 EMAformerDynamic 模型"""
    print("=" * 60)
    print("测试 6: EMAformerDynamic 模型")
    print("=" * 60)

    # 创建配置对象
    class Config:
        seq_len = 96
        pred_len = 96
        d_model = 256
        n_heads = 4
        e_layers = 2
        d_ff = 512
        dropout = 0.1
        factor = 1
        activation = 'gelu'
        output_attention = False
        use_norm = True
        cycle = 24
        enc_in = 7
        dec_in = 7
        c_out = 7
        embed = 'timeF'
        freq = 'h'
        output_proj_dropout = 0.1

    config = Config()

    # 初始化模型
    model = Model(config)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    B = 4
    x_enc = torch.randn(B, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(B, config.seq_len, 4)  # 时间特征
    x_dec = torch.zeros(B, config.label_len + config.pred_len if hasattr(config, 'label_len') else 48 + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(B, config.pred_len, 4)
    cycle_index = torch.randint(0, 24, (B,))

    # 前向传播
    model.eval()
    with torch.no_grad():
        dec_out, attns, stats = model(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index)

    print(f"  输入 x_enc 形状: {x_enc.shape}")
    print(f"  输出 dec_out 形状: {dec_out.shape}")
    print(f"  注意力形状: {attns[0].shape if attns else None}")
    print(f"  统计特征形状: {stats.shape}")

    # 验证输出
    assert dec_out.shape == (B, config.pred_len, config.enc_in)
    assert not torch.isnan(dec_out).any()
    print("  ✓ EMAformerDynamic 模型测试通过\n")


def test_zeroshot_variant():
    """测试 Zero-Shot 变体"""
    print("=" * 60)
    print("测试 7: EMAformerDynamicZeroShot 变体")
    print("=" * 60)

    class Config:
        seq_len = 96
        pred_len = 96
        d_model = 256
        n_heads = 4
        e_layers = 2
        d_ff = 512
        dropout = 0.1
        factor = 1
        activation = 'gelu'
        output_attention = False
        use_norm = True
        cycle = 24
        enc_in = 7
        dec_in = 7
        c_out = 7
        embed = 'timeF'
        freq = 'h'
        output_proj_dropout = 0.1

    config = Config()
    model = EMAformerDynamicZeroShot(config)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    B = 4
    x_enc = torch.randn(B, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(B, config.seq_len, 4)
    x_dec = torch.zeros(B, 48 + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(B, config.pred_len, 4)
    cycle_index = torch.randint(0, 24, (B,))

    model.eval()
    with torch.no_grad():
        dec_out, attns, stats = model(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index)

    print(f"  Zero-Shot 输出形状: {dec_out.shape}")
    assert dec_out.shape == (B, config.pred_len, config.enc_in)
    print("  ✓ Zero-Shot 变体测试通过\n")


def main():
    print("\n" + "=" * 60)
    print("EMAformerDynamic 动态嵌入护甲测试")
    print("=" * 60 + "\n")

    test_statistical_features()
    test_dynamic_channel_embedding()
    test_dynamic_phase_embedding()
    test_dynamic_joint_embedding()
    test_dynamic_embedding_armor()
    test_emaformer_dynamic_model()
    test_zeroshot_variant()

    print("=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
    print("\n创新点总结:")
    print("1. StatisticalFeatures: 从数据中提取均值、标准差、偏度、峰度、FFT频域、自相关")
    print("2. DynamicChannelEmbedding: 数据驱动的通道嵌入,替代静态参数")
    print("3. DynamicPhaseEmbedding: 自适应周期编码,替代固定查表")
    print("4. DynamicJointEmbedding: 通道-相位联合建模")
    print("5. DynamicEmbeddingArmor: 整合以上三个动态嵌入")
    print("6. EMAformerDynamic: 使用动态嵌入的完整模型")
    print("7. EMAformerDynamicZeroShot: 专为跨域Zero-Shot设计")
    print("\n优势:")
    print("- 无需为每个域学习静态嵌入")
    print("- 新域可通过统计特征匹配实现 Zero-Shot")
    print("- 支持跨域迁移和少样本学习")


if __name__ == "__main__":
    main()
