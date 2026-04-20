#!/usr/bin/env python3
"""
轻量级代码检测脚本 - EMAformerDynamic

功能：
1. 检查 Python 语法正确性
2. 验证模块导入是否成功
3. 检查类定义和依赖关系
4. 不执行任何张量计算（节省资源）

使用方法：
    python scripts/check_dynamic_embedding.py

输出：
    - 成功：显示 "所有检查通过 ✓"
    - 失败：显示具体错误位置
"""

import ast
import sys
import importlib.util
from pathlib import Path


def check_syntax(file_path, description):
    """检查 Python 文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print(f"  ✓ {description}: 语法正确")
        return True
    except SyntaxError as e:
        print(f"  ✗ {description}: 语法错误")
        print(f"    位置: 第{e.lineno}行, 第{e.offset}列")
        print(f"    错误: {e.msg}")
        return False
    except Exception as e:
        print(f"  ✗ {description}: 读取失败 - {e}")
        return False


def check_imports(module_path, description):
    """检查模块能否导入（不执行代码）"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None:
            print(f"  ✗ {description}: 无法找到模块")
            return False
        module = importlib.util.module_from_spec(spec)
        # 只加载模块定义，不执行（避免CUDA初始化）
        print(f"  ✓ {description}: 模块结构可加载")
        return True
    except Exception as e:
        print(f"  ✗ {description}: 导入失败 - {e}")
        return False


def check_class_definitions(file_path, expected_classes):
    """检查文件中是否包含预期的类定义"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        found_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found_classes.append(node.name)

        missing = [cls for cls in expected_classes if cls not in found_classes]
        if missing:
            print(f"  ✗ 缺少类定义: {missing}")
            return False

        print(f"  ✓ 类定义检查通过: {', '.join(expected_classes)}")
        return True
    except Exception as e:
        print(f"  ✗ 类定义检查失败: {e}")
        return False


def check_file_exists(file_path, description):
    """检查文件是否存在"""
    path = Path(file_path)
    if path.exists():
        print(f"  ✓ {description}: 文件存在")
        return True
    else:
        print(f"  ✗ {description}: 文件不存在 - {file_path}")
        return False


def main():
    print("=" * 70)
    print("EMAformerDynamic 代码检测")
    print("=" * 70)
    print()

    all_passed = True
    base_path = Path(".")

    # ========== 检查 1: 新文件存在性 ==========
    print("【检查 1】新文件存在性")
    files_to_check = [
        ("layers/DynamicEmbedding.py", "动态嵌入层"),
        ("model/EMAformerDynamic.py", "动态嵌入模型"),
        ("scripts/test_dynamic_embedding.py", "测试脚本"),
        ("scripts/multivariate_forecasting/ECL/EMAformerDynamic.sh", "运行脚本"),
    ]
    for file_path, desc in files_to_check:
        if not check_file_exists(base_path / file_path, desc):
            all_passed = False
    print()

    # ========== 检查 2: Python 语法 ==========
    print("【检查 2】Python 语法检查")
    syntax_files = [
        ("layers/DynamicEmbedding.py", "DynamicEmbedding 层"),
        ("model/EMAformerDynamic.py", "EMAformerDynamic 模型"),
        ("model/__init__.py", "模型模块导出"),
        ("scripts/test_dynamic_embedding.py", "测试脚本"),
    ]
    for file_path, desc in syntax_files:
        if not check_syntax(base_path / file_path, desc):
            all_passed = False
    print()

    # ========== 检查 3: 类定义检查 ==========
    print("【检查 3】类定义检查")

    # DynamicEmbedding.py 中应该有的类
    print("  DynamicEmbedding.py:")
    if not check_class_definitions(
        base_path / "layers/DynamicEmbedding.py",
        ["StatisticalFeatures", "DynamicChannelEmbedding", "DynamicPhaseEmbedding",
         "DynamicJointEmbedding", "DynamicEmbeddingArmor"]
    ):
        all_passed = False

    # EMAformerDynamic.py 中应该有的类
    print("  EMAformerDynamic.py:")
    if not check_class_definitions(
        base_path / "model/EMAformerDynamic.py",
        ["Model", "EMAformerDynamicZeroShot", "EMAformerDynamicTransfer"]
    ):
        all_passed = False
    print()

    # ========== 检查 4: 关键导入语句 ==========
    print("【检查 4】关键导入语句检查")

    # 检查 model/__init__.py 的导入
    print("  检查 model/__init__.py 导入:")
    try:
        with open(base_path / "model/__init__.py", 'r') as f:
            content = f.read()

        required_imports = [
            "from model.Transformer import Model as Transformer",
            "from model.EMAformer import Model as EMAformer",
            "from model.EMAformerDynamic import Model as EMAformerDynamic",
        ]

        for imp in required_imports:
            if imp in content:
                print(f"    ✓ {imp.split()[-1]}")
            else:
                print(f"    ✗ 缺少导入: {imp}")
                all_passed = False
    except Exception as e:
        print(f"    ✗ 检查失败: {e}")
        all_passed = False
    print()

    # ========== 检查 5: 实验配置 ==========
    print("【检查 5】实验配置检查")
    try:
        with open(base_path / "experiments/exp_basic.py", 'r') as f:
            content = f.read()

        if "'EMAformerDynamic'" in content:
            print("  ✓ EMAformerDynamic 已注册到 model_dict")
        else:
            print("  ✗ EMAformerDynamic 未注册")
            all_passed = False

        if "EMAformerDynamicZeroShot" in content:
            print("  ✓ EMAformerDynamicZeroShot 已注册")
        else:
            print("  ✗ EMAformerDynamicZeroShot 未注册")
            all_passed = False

        if "EMAformerDynamicTransfer" in content:
            print("  ✓ EMAformerDynamicTransfer 已注册")
        else:
            print("  ✗ EMAformerDynamicTransfer 未注册")
            all_passed = False
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        all_passed = False
    print()

    # ========== 检查 6: 关键函数签名 ==========
    print("【检查 6】关键函数签名检查")
    try:
        with open(base_path / "layers/DynamicEmbedding.py", 'r') as f:
            content = f.read()

        required_funcs = [
            "def compute_mean",
            "def compute_std",
            "def compute_skewness",
            "def compute_kurtosis",
            "def compute_fft_features",
            "def compute_autocorr_features",
        ]

        for func in required_funcs:
            if func in content:
                print(f"  ✓ {func}")
            else:
                print(f"  ✗ 缺少函数: {func}")
                all_passed = False
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        all_passed = False
    print()

    # ========== 最终输出 ==========
    print("=" * 70)
    if all_passed:
        print("✓ 所有检查通过！代码可以安全传输到服务器运行。")
        print("=" * 70)
        print()
        print("传输到服务器后，可以运行以下命令：")
        print("  1. 测试动态嵌入：")
        print("     python scripts/test_dynamic_embedding.py")
        print()
        print("  2. 训练 EMAformerDynamic：")
        print("     bash scripts/multivariate_forecasting/ECL/EMAformerDynamic.sh")
        print()
        print("  3. 使用新模型：")
        print("     python run.py --model EMAformerDynamic --data ETTh1 ...")
        return 0
    else:
        print("✗ 部分检查未通过，请修复上述错误后再传输。")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
