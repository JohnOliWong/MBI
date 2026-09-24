import argparse
import os
import numpy as np
import pandas as pd


METRIC_COLS = ['Loss', 'Acc', 'Precision', 'Recall', 'F1', 'Kappa']

def load_subject_results(results_root, num_subjects=29):
    """读取每个受试者的 xlsx，返回 {subject: DataFrame} 以及缺失的受试者列表。"""
    results = {}
    for subject in range(1, num_subjects + 1):
        path = os.path.join(results_root, f'{subject}.xlsx')
        if os.path.exists(path):
            results[subject] = pd.read_excel(path)
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

if __name__ == '__main__':
    # 以当前文件绝对目录的上一级作为base_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    checkpoint_names = ['EEGNet_EFMA_Sep-24-2026-14:45:33', 'EEGNet_EFWG_Sep-24-2026-15:06:07']
    
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
    
    for checkpoint_name in checkpoint_names:
        dataset = checkpoint_name.split('_')[1]
        num_subjects = 29 if dataset in ['EFMI', 'EFMA'] else 26
        
        results_root = os.path.join(base_dir, 'Results', checkpoint_name)
        results = load_subject_results(results_root, num_subjects)

        # 三种模式，best, x%, mean
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

        lines = [checkpoint_name]
        for name, row in modes.items():
            if row is None:
                continue
            lines.append(f'mode={name}')
            lines.append(f'{format_metrics(row)}')

        log_path = os.path.join(base_dir, 'Results', 'Results.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        content = '\n' + '\n'.join(lines) + '\n'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(content)

        print(content)