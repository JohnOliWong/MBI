import argparse
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook


EXPORT_TO_LOG = False
EXPORT_TO_EXCEL = True

# results.xlsx 结构：前 num_epochs 行是 train/val 过程，最后一行是 test。
# 报告指标（不再输出 val acc，用 B-ACC 做 best / top-x% 排序依据）：
PROCESS_COLS = ['B-ACC', 'precision', 'W-F1', 'kappa', 'AUC']
SELECTION_COL = 'B-ACC'
# test 行（最后一行）报告的指标（与过程行一致）：
TEST_COLS = ['B-ACC', 'precision', 'W-F1', 'kappa', 'AUC']

# 模型和数据集排序
MODEL_ORDER = ['EEGNet', 'Conformer', 'EFNet', 'VigilanceNet', 'TSMMF', 'EFBook']
DATASET_ORDER = ['SEEDIV', 'SEEDV']

def sort_key(checkpoint_name):
    """按 dataset 优先、model 其次的固定顺序排序。"""
    model = checkpoint_name.split('_')[0]
    dataset = checkpoint_name.split('_')[1]
    return (DATASET_ORDER.index(dataset), MODEL_ORDER.index(model))

def load_results_df(results_root):
    """读取单个 checkpoint 的 results.xlsx，返回完整 DataFrame（不存在返回 None）。"""
    path = os.path.join(results_root, 'results.xlsx')
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    # 去掉 loss 列
    df = df.drop(columns=[c for c in df.columns if 'loss' in c.lower()], errors='ignore')
    return df

def process_df(df):
    """排除最后一行（test），返回 train/val 过程行。"""
    return df.iloc[:-1]

def process_best(df):
    """过程行中 val acc 最高的那个 epoch 的指标。"""
    d = process_df(df)
    if len(d) == 0:
        return None
    return d.loc[d[SELECTION_COL].idxmax(), PROCESS_COLS].astype(float)

def process_top_x(df, x):
    """过程行中 val acc 排名前 x% 的 epoch，指标求平均。"""
    d = process_df(df)
    if len(d) == 0:
        return None
    n = max(1, int(np.ceil(len(d) * x)))
    top = d.nlargest(n, SELECTION_COL)
    return top[PROCESS_COLS].astype(float).mean()

def test_metrics(df):
    """最后一行（test）的指标。"""
    return df.iloc[-1][TEST_COLS].astype(float)

def format_metrics(row):
    return ', '.join(f'{col}={row[col]:.2f}' for col in row.index)

def export_all_to_excel(excel_path, checkpoint_names, all_modes, cols):
    """导出为表格（overwrite 写入）。每个 checkpoint 一个列块：
    第一行 checkpoint_name，第二行 Mode + 指标表头，第三行起每行一个模式。
    """
    wb = Workbook()
    ws = wb.active
    ws.title = 'Source'

    n_cols = 1 + len(cols)
    block_height = 2 + max((len(modes) for modes in all_modes), default=0)

    for i, (checkpoint_name, modes) in enumerate(zip(checkpoint_names, all_modes)):
        group = i // 3
        idx_in_group = i % 3
        start_col = idx_in_group * (n_cols + 1) + 1
        start_row = group * (block_height + 1) + 1

        ws.cell(row=start_row, column=start_col, value=checkpoint_name)
        ws.cell(row=start_row + 1, column=start_col, value='Mode')
        for j, col_name in enumerate(cols):
            ws.cell(row=start_row + 1, column=start_col + 1 + j, value=col_name)

        row_idx = start_row + 2
        for name, row in modes.items():
            if row is None:
                continue
            ws.cell(row=row_idx, column=start_col, value=name)
            for j, col_name in enumerate(cols):
                val = row.get(col_name)
                if val is not None and not pd.isna(val):
                    ws.cell(row=row_idx, column=start_col + 1 + j, value=round(float(val), 2))
            row_idx += 1

    wb.save(excel_path)

def summarize_results(base_dir=None, checkpoint_names=None):
    """汇总指定 checkpoints 的指标，返回 (汇总文本, 有效 checkpoint 列表, all_modes)。

    每个 checkpoint 输出：best / Top x% / ... / test（最后一行）。
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

    valid_names = []
    all_modes = []
    blocks = []
    for checkpoint_name in checkpoint_names:
        results_root = os.path.join(results_dir, checkpoint_name)
        df = load_results_df(results_root)
        if df is None:
            continue
        valid_names.append(checkpoint_name)

        modes = {'best': process_best(df)}
        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            modes[f'Top {x:.0%}'] = process_top_x(df, x)
        modes['test'] = test_metrics(df)   # 最后一行（test）
        all_modes.append(modes)

        lines = [checkpoint_name]
        for name, row in modes.items():
            if row is None:
                continue
            lines.append(f'mode={name}')
            lines.append(f'{format_metrics(row)}')
        blocks.append('\n'.join(lines))

    text = '\n\n'.join(blocks)
    return text, valid_names, all_modes


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'Results')

    checkpoint_names = [
        folder for folder in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, folder)) and folder != 'Temp'
    ]
    checkpoint_names.sort(key=sort_key)

    text, valid_names, all_modes = summarize_results(base_dir=base_dir, checkpoint_names=checkpoint_names)
    print(text)

    if EXPORT_TO_LOG:
        log_path = os.path.join(base_dir, 'Results', 'Results.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    if EXPORT_TO_EXCEL:
        excel_path = os.path.join(base_dir, 'Results', 'Results.xlsx')
        export_all_to_excel(excel_path, valid_names, all_modes, PROCESS_COLS)