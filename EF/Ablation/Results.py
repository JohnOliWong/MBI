import argparse
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook


EXPORT_TO_LOG = False
EXPORT_TO_EXCEL = True
METRIC_COLS = ['Acc', 'Precision', 'Recall', 'F1', 'Kappa']

# 消融模式和数据集排序
MODE_ORDER = ['OT', 'DECOUPLE', 'PC', 'SC', 'EEG', 'fNIRS']
DATASET_ORDER = ['EFMI', 'EFMA', 'EFWG']

def sort_key(checkpoint_name):
    """按 mode 优先、dataset 其次的固定顺序排序。"""
    dataset = checkpoint_name.split('_')[1]
    mode = checkpoint_name.split('_')[2]
    return (MODE_ORDER.index(mode), DATASET_ORDER.index(dataset))

def load_subject_results(results_root, num_subjects=29):
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

    布局：块与块之间空一列；每个 mode 的 checkpoint 排成一行（MI / WG ...），
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

def summarize_results(base_dir=None, checkpoint_names=None):
    """LOSO 汇总：对每个受试者 best/top-x%，再对受试者求平均。返回 (汇总文本, all_modes)。

    checkpoint_names 为 None 时，扫描 Results/ 下所有 checkpoint 目录。
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'Results')

    if checkpoint_names is None:
        checkpoint_names = [
            folder for folder in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, folder)) and folder != 'Temp'
        ]
        checkpoint_names.sort(key=sort_key)

    all_modes = []
    blocks = []
    for checkpoint_name in checkpoint_names:
        dataset = checkpoint_name.split('_')[1]
        num_subjects = 29 if dataset in ['EFMI', 'EFMA'] else 26

        results_root = os.path.join(results_dir, checkpoint_name)
        results = load_subject_results(results_root, num_subjects)

        modes = {'best': aggregate(results, 'best')}
        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            modes[f'Top {x:.0%}'] = aggregate(results, 'x%', x=x)
        all_modes.append(modes)

        lines = [checkpoint_name]
        for name, row in modes.items():
            if row is None:
                continue
            lines.append(f'mode={name}')
            lines.append(f'{format_metrics(row)}')
        blocks.append('\n'.join(lines))

    text = '\n\n'.join(blocks)
    return text, all_modes

if __name__ == '__main__':
    # 以当前文件绝对目录的上一级作为base_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'Results')

    # checkpoint_names = ['EEGNet_EFMA_Sep-24-2026-14:45:33', 'EEGNet_EFWG_Sep-24-2026-15:06:07']
    checkpoint_names = [
        folder for folder in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, folder)) and folder != 'Temp'
    ]
    checkpoint_names.sort(key=sort_key)

    text, all_modes = summarize_results(base_dir=base_dir, checkpoint_names=checkpoint_names)
    print(text)

    if EXPORT_TO_LOG:
        log_path = os.path.join(base_dir, 'Results', 'Results.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    if EXPORT_TO_EXCEL:
        excel_path = os.path.join(base_dir, 'Results', 'Results.xlsx')
        export_all_to_excel(excel_path, checkpoint_names, all_modes)