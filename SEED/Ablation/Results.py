import argparse
import json
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook


EXPORT_TO_LOG = False
EXPORT_TO_EXCEL = True
METRIC_COLS = ['Acc', 'Precision', 'Recall', 'F1', 'Kappa']

# 消融模式和数据集排序
MODE_ORDER = ['OT', 'DECOUPLE', 'PC', 'SC', 'EEG', 'EYE']
DATASET_ORDER = ['SEEDIV', 'SEEDV']

def sort_key(checkpoint_name):
    """按 mode 优先、dataset 其次的固定顺序排序。"""
    dataset = checkpoint_name.split('_')[1]
    mode = checkpoint_name.split('_')[2]
    return (MODE_ORDER.index(mode), DATASET_ORDER.index(dataset))

def load_subject_results(results_root, num_subjects=16):
    """读取每个受试者的 xlsx，返回 {subject: DataFrame} 以及缺失的受试者列表。"""
    results = {}
    for subject in range(1, num_subjects + 1):
        path = os.path.join(results_root, f'{subject}.xlsx')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_excel(path)
        except Exception as exc:
            # 损坏/未写完整的 xlsx 直接跳过，避免中断整个导出
            print(f'[WARN] 跳过无法读取的文件 {path}: {exc}')
            continue
        # 去掉data frame中的loss列
        df = df.drop(columns=[c for c in df.columns if 'loss' in c.lower()], errors='ignore')
        results[subject] = df
    return results

def subject_best(df):
    """取该受试者验证 Acc 最高的那个 epoch 的指标。"""
    idx = df['Acc'].idxmax()
    return df.loc[idx, METRIC_COLS].astype(float)

def subject_top_x(df, x):
    """取该受试者 Acc 排名前 x% 的 epoch，对这些 epoch 的指标求平均。"""
    n = max(1, int(np.ceil(len(df) * x)))
    top = df.nlargest(n, 'Acc')
    return top[METRIC_COLS].astype(float).mean()

def subject_mean(df):
    """取该受试者所有 epoch 的指标平均。"""
    return df[METRIC_COLS].astype(float).mean()

def aggregate_df(df, mode, x=None):
    """对 Split 模式下单份 results.xlsx 的 epoch 行做聚合（df 已去除 loss 列）。"""
    if mode == 'best':
        return df.loc[df['Acc'].idxmax(), METRIC_COLS].astype(float)
    elif mode == 'x%':
        n = max(1, int(np.ceil(len(df) * x)))
        return df.nlargest(n, 'Acc')[METRIC_COLS].astype(float).mean()
    elif mode == 'mean':
        return df[METRIC_COLS].astype(float).mean()
    return None

def aggregate(results, mode, x=None):
    """对每个受试者计算一组指标，再对受试者求平均，返回一行指标。"""
    per_subject = []
    for subject, df in sorted(results.items()):
        if mode == 'best':
            per_subject.append(subject_best(df))
        elif mode == 'x%':
            per_subject.append(subject_top_x(df, x))
        elif mode == 'mean':
            per_subject.append(subject_mean(df))
    if not per_subject:
        return None
    return pd.DataFrame(per_subject)[METRIC_COLS].mean()

def format_metrics(row):
    return ', '.join(f'{col}={row[col]:.2f}' for col in METRIC_COLS)

def export_all_to_excel(excel_path, checkpoint_names, all_modes):
    """为了保证排版布局，现在导出为表格是overwrite写入
    
    每个 checkpoint 一个列块：
      第一行：checkpoint_name
      第二行：Mode + 各指标表头
      第三行起：每行最左是模式名（best / Top 10% ...），右侧是各指标值

    布局：块与块之间空一列；每个 mode 的 checkpoint 排成一行（IV / V ...），
    换到下一个 mode 时空一行，再从下一行开始。
    """
    wb = Workbook()
    ws = wb.active
    ws.title = 'Source'

    n_cols = 1 + len(METRIC_COLS)
    block_height = 2 + max(
        (len([r for r in modes.values() if r is not None]) for modes in all_modes),
        default=0,
    )

    group = 0
    idx_in_group = 0
    prev_mode = None
    for i, (checkpoint_name, modes) in enumerate(zip(checkpoint_names, all_modes)):
        mode = checkpoint_name.split('_')[2]
        if i > 0 and mode != prev_mode:
            group += 1
            idx_in_group = 0
        prev_mode = mode

        start_col = idx_in_group * (n_cols + 1) + 1
        start_row = group * (block_height + 1) + 1
        idx_in_group += 1

        ws.cell(row=start_row, column=start_col, value=checkpoint_name)
        ws.cell(row=start_row + 1, column=start_col, value='Mode')
        for j, col_name in enumerate(METRIC_COLS):
            ws.cell(row=start_row + 1, column=start_col + 1 + j, value=col_name)

        row_idx = start_row + 2
        for name, row in modes.items():
            if row is None:
                continue
            ws.cell(row=row_idx, column=start_col, value=name)
            for j, col_name in enumerate(METRIC_COLS):
                ws.cell(row=row_idx, column=start_col + 1 + j, value=round(float(row[col_name]), 2))
            row_idx += 1

    wb.save(excel_path)

if __name__ == '__main__':
    # 以当前文件绝对目录的上一级作为base_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'Results')
    
    checkpoint_names = [
        folder for folder in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, folder)) and folder != 'Temp'
    ]
    checkpoint_names.sort(key=sort_key)
    
    x1 = 0.1
    x2 = 0.2
    x3 = 0.3
    x4 = 0.4
    x5 = 0.5
    x6 = 0.6
    x7 = 0.7
    x8 = 0.8
    x9 = 0.9
    x10 = 1.0
    
    all_modes = []
    for checkpoint_name in checkpoint_names:
        dataset = checkpoint_name.split('_')[1]
        num_subjects = 15 if dataset == 'SEEDIV' else 16
        
        results_root = os.path.join(base_dir, 'Results', checkpoint_name)

        # Split 模式：单份 results.xlsx（epoch 行）+ Test_Metrics.json（holdout 测试指标）
        split_path = os.path.join(results_root, 'results.xlsx')
        if os.path.exists(split_path):
            df = pd.read_excel(split_path)
            df = df.drop(columns=[c for c in df.columns if 'loss' in c.lower()], errors='ignore')
            modes = {
                'best': aggregate_df(df, 'best'),
                f'Top {x1:.0%}': aggregate_df(df, 'x%', x=x1),
                f'Top {x2:.0%}': aggregate_df(df, 'x%', x=x2),
                f'Top {x3:.0%}': aggregate_df(df, 'x%', x=x3),
                f'Top {x4:.0%}': aggregate_df(df, 'x%', x=x4),
                f'Top {x5:.0%}': aggregate_df(df, 'x%', x=x5),
                f'Top {x6:.0%}': aggregate_df(df, 'x%', x=x6),
                f'Top {x7:.0%}': aggregate_df(df, 'x%', x=x7),
                f'Top {x8:.0%}': aggregate_df(df, 'x%', x=x8),
                f'Top {x9:.0%}': aggregate_df(df, 'x%', x=x9),
                f'Top {x10:.0%}': aggregate_df(df, 'x%', x=x10),
                'mean': aggregate_df(df, 'mean'),
            }
            test_path = os.path.join(results_root, 'Test_Metrics.json')
            if os.path.exists(test_path):
                with open(test_path, 'r', encoding='utf-8') as f:
                    test_metrics = json.load(f)
                modes['test'] = pd.Series({c: float(test_metrics.get(c, 0.0)) for c in METRIC_COLS})
        else:
            # 兼容旧的 LOSO 结果目录（按受试者 xlsx）
            results = load_subject_results(results_root, num_subjects)
            modes = {
                'best': aggregate(results, 'best'),
                f'Top {x1:.0%}': aggregate(results, 'x%', x=x1),
                f'Top {x2:.0%}': aggregate(results, 'x%', x=x2),
                f'Top {x3:.0%}': aggregate(results, 'x%', x=x3),
                f'Top {x4:.0%}': aggregate(results, 'x%', x=x4),
                f'Top {x5:.0%}': aggregate(results, 'x%', x=x5),
                f'Top {x6:.0%}': aggregate(results, 'x%', x=x6),
                f'Top {x7:.0%}': aggregate(results, 'x%', x=x7),
                f'Top {x8:.0%}': aggregate(results, 'x%', x=x8),
                f'Top {x9:.0%}': aggregate(results, 'x%', x=x9),
                f'Top {x10:.0%}': aggregate(results, 'x%', x=x10),
            }
        all_modes.append(modes)

        lines = [checkpoint_name]
        for name, row in modes.items():
            if row is None:
                continue
            lines.append(f'mode={name}')
            lines.append(f'{format_metrics(row)}')

        content = '\n' + '\n'.join(lines) + '\n'
        print(content)
        
        if EXPORT_TO_LOG:
            log_path = os.path.join(base_dir, 'Results', 'Results.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(content)

    if EXPORT_TO_EXCEL:
        excel_path = os.path.join(base_dir, 'Results', 'Results.xlsx')
        export_all_to_excel(excel_path, checkpoint_names, all_modes)
