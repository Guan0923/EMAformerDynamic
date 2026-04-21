#!/usr/bin/env python3
"""
result_long_term_forecast.txt 结果解析工具

将实验结果从 txt 格式转换为 CSV，便于查看和分析。

使用方法:
    python parse_results.py
    
会自动读取 result_long_term_forecast.txt 并生成 result_long_term_forecast.csv
"""

import re
import csv
import os
from pathlib import Path


def parse_setting(setting_line):
    """
    解析实验设置字符串
    
    setting 格式示例:
    ETTh1_96_96_EMAformer_ETTh1_M_ft96_sl48_ll96_pl256_dm4_nh3_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0
    
    参数映射:
    - ft: seq_len (input sequence length)
    - sl: label_len (start token length)
    - ll: pred_len (prediction length)
    - pl: d_model (dimension of model)
    - dm: n_heads (number of heads)
    - nh: e_layers (encoder layers)
    - el: d_layers (decoder layers)
    - dl: d_ff (dimension of fcn)
    - df: factor (attention factor)
    - eb: embed (time feature encoding)
    - dt: distil (whether to use distilling)
    
    返回: dict 包含所有解析出的参数
    """
    parts = setting_line.strip().split('_')
    
    result = {
        'model_id': '',
        'model': '',
        'data': '',
        'features': '',
        'seq_len': '',
        'label_len': '',
        'pred_len': '',
        'd_model': '',
        'n_heads': '',
        'e_layers': '',
        'd_layers': '',
        'd_ff': '',
        'factor': '',
        'embed': '',
        'distil': '',
        'des': '',
        'class_strategy': '',
        'iteration': ''
    }
    
    # 解析 model_id (前几个下划线分隔的部分，直到遇到 model 名称)
    # model_id 通常包含数据集和预测长度信息，如 ETTh1_96_96 或 ETTh1_Dynamic_96_96
    model_name_candidates = ['EMAformer', 'EMAformerDynamic', 'EMAformer_hybrid_channel', 
                             'EMAformer_hybrid_phase', 'EMAformer_hybrid_joint',
                             'iInformer', 'iReformer', 'iFlowformer', 'iFlashformer']
    
    model_idx = -1
    model_name = ''
    for i, part in enumerate(parts):
        if part in model_name_candidates or any(part.startswith(m) for m in model_name_candidates):
            model_idx = i
            model_name = part
            break
    
    if model_idx > 0:
        result['model_id'] = '_'.join(parts[:model_idx])
        result['model'] = model_name
    else:
        # 如果找不到 model 名称，尝试从第3个位置推断
        result['model_id'] = '_'.join(parts[:3]) if len(parts) >= 3 else parts[0]
        result['model'] = parts[3] if len(parts) >= 4 else ''
        model_idx = 3
    
    # 从 model_idx 之后继续解析
    remaining_parts = parts[model_idx + 1:]
    
    # data 和 features
    if len(remaining_parts) >= 2:
        result['data'] = remaining_parts[0]
        result['features'] = remaining_parts[1]
        remaining_parts = remaining_parts[2:]
    
    # 使用正则表达式解析带前缀的参数
    i = 0
    while i < len(remaining_parts):
        part = remaining_parts[i]
        
        # ft -> seq_len
        if part.startswith('ft'):
            result['seq_len'] = part[2:]
        # sl -> label_len
        elif part.startswith('sl'):
            result['label_len'] = part[2:]
        # ll -> pred_len
        elif part.startswith('ll'):
            result['pred_len'] = part[2:]
        # pl -> d_model
        elif part.startswith('pl'):
            result['d_model'] = part[2:]
        # dm -> n_heads
        elif part.startswith('dm'):
            result['n_heads'] = part[2:]
        # nh -> e_layers
        elif part.startswith('nh'):
            result['e_layers'] = part[2:]
        # el -> d_layers
        elif part.startswith('el'):
            result['d_layers'] = part[2:]
        # dl -> d_ff
        elif part.startswith('dl'):
            result['d_ff'] = part[2:]
        # df -> factor
        elif part.startswith('df'):
            result['factor'] = part[2:]
        # fc -> embed
        elif part.startswith('fc'):
            result['embed'] = part[2:]
        # eb -> distil
        elif part.startswith('eb'):
            result['distil'] = part[2:]
        # dt -> des
        elif part.startswith('dt'):
            result['des'] = part[2:]
        else:
            # 可能是 class_strategy 或 iteration
            if i == len(remaining_parts) - 1:
                # 最后一个数字是 iteration
                if part.isdigit():
                    result['iteration'] = part
                else:
                    result['class_strategy'] = part
            elif i == len(remaining_parts) - 2:
                # 倒数第二个是 class_strategy，最后一个是 iteration
                result['class_strategy'] = part
                if i + 1 < len(remaining_parts) and remaining_parts[i + 1].isdigit():
                    result['iteration'] = remaining_parts[i + 1]
                    i += 1
            else:
                # 可能是 des 的一部分（des 可能包含多个下划线）
                if not result['des']:
                    result['des'] = part
                elif not result['class_strategy']:
                    result['class_strategy'] = part
        
        i += 1
    
    return result


def parse_metrics(metrics_line):
    """
    解析性能指标行
    
    格式: mse:0.3863406777381897, mae:0.4001886546611786
    
    返回: dict 包含 mse 和 mae
    """
    result = {'mse': '', 'mae': ''}
    
    # 使用正则表达式提取数值
    mse_match = re.search(r'mse:([\d.]+)', metrics_line)
    mae_match = re.search(r'mae:([\d.]+)', metrics_line)
    
    if mse_match:
        result['mse'] = mse_match.group(1)
    if mae_match:
        result['mae'] = mae_match.group(1)
    
    return result


def parse_results_file(input_file):
    """
    解析结果文件
    
    返回: list of dicts，每个 dict 是一条实验记录
    """
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过空行
        if not line:
            i += 1
            continue
        
        # 检查是否是 setting 行（通常包含 model 名称）
        if any(model in line for model in ['EMAformer', 'iInformer', 'iReformer', 'iFlowformer', 'iFlashformer']):
            setting_line = line
            
            # 下一行应该是指标行
            metrics_line = ''
            if i + 1 < len(lines):
                metrics_line = lines[i + 1].strip()
            
            # 解析 setting 和 metrics
            setting_data = parse_setting(setting_line)
            metrics_data = parse_metrics(metrics_line)
            
            # 合并数据
            record = {**setting_data, **metrics_data}
            results.append(record)
            
            i += 2  # 跳过 metrics 行
        else:
            i += 1
    
    return results


def write_csv(results, output_file):
    """将结果写入 CSV 文件"""
    if not results:
        print("警告: 没有解析到任何结果")
        return
    
    fieldnames = [
        'model_id', 'model', 'data', 'features',
        'seq_len', 'label_len', 'pred_len',
        'd_model', 'n_heads', 'e_layers', 'd_layers',
        'd_ff', 'factor', 'embed', 'distil',
        'des', 'class_strategy', 'iteration',
        'mse', 'mae'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"成功写入 {len(results)} 条记录到 {output_file}")


def print_summary(results):
    """打印结果摘要"""
    if not results:
        return
    
    print("\n" + "="*80)
    print("结果摘要")
    print("="*80)
    
    # 统计不同的模型
    models = set(r['model'] for r in results if r['model'])
    print(f"\n模型数量: {len(models)}")
    for model in sorted(models):
        count = sum(1 for r in results if r['model'] == model)
        print(f"  - {model}: {count} 条记录")
    
    # 统计不同的数据集
    datasets = set(r['data'] for r in results if r['data'])
    print(f"\n数据集: {', '.join(sorted(datasets))}")
    
    # 统计不同的预测长度
    pred_lens = set(r['pred_len'] for r in results if r['pred_len'])
    print(f"预测长度: {', '.join(sorted(pred_lens, key=int))}")
    
    # 每个模型的最佳结果
    print("\n" + "-"*80)
    print("各模型最佳 MSE 结果（按 pred_len 分组）:")
    print("-"*80)
    
    for model in sorted(models):
        model_results = [r for r in results if r['model'] == model]
        pred_len_groups = {}
        
        for r in model_results:
            pl = r.get('pred_len', '')
            if pl:
                if pl not in pred_len_groups:
                    pred_len_groups[pl] = []
                pred_len_groups[pl].append(r)
        
        print(f"\n{model}:")
        for pl in sorted(pred_len_groups.keys(), key=int):
            group = pred_len_groups[pl]
            best = min(group, key=lambda x: float(x['mse']) if x['mse'] else float('inf'))
            print(f"  pred_len={pl}: MSE={best['mse']}, MAE={best['mae']}")


def main():
    """主函数"""
    # 默认文件路径
    input_file = 'result_long_term_forecast.txt'
    output_file = 'result_long_term_forecast.csv'
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        # 尝试在项目根目录中查找
        script_dir = Path(__file__).parent
        input_file = script_dir / input_file
        output_file = script_dir / output_file
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"错误: 找不到输入文件 {input_path}")
        print(f"请确保 {input_path.name} 存在于当前目录或脚本所在目录")
        return
    
    print(f"正在解析: {input_path}")
    
    # 解析结果
    results = parse_results_file(input_path)
    
    if not results:
        print("错误: 未能解析到任何有效记录")
        return
    
    print(f"成功解析 {len(results)} 条记录")
    
    # 写入 CSV
    write_csv(results, output_path)
    
    # 打印摘要
    print_summary(results)
    
    print(f"\n完成! CSV 文件已保存到: {output_path}")


if __name__ == '__main__':
    main()
