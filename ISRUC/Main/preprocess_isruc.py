#!/usr/bin/env python3
"""ISRUC 预处理：.rec(EDF) + _1.txt 标签 -> 按受试者存储的 .pkl（EF 风格）。

通道选择（共 11 导，采样率均为 200 Hz，按文件物理顺序）：
  EOG: LOC-A2, ROC-A1          (文件第 0, 1 通道)
  EEG: F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1   (第 2-7 通道)
  EMG: X1, X2, X3 (下颌 EMG)   (第 8-10 通道)
不同受试者参考电极命名不一致，通过别名归一化：
  -A1/-A2（耳垂）与 -M1/-M2（乳突）；EMG 的 X1/X2/X3 与数字 24/25/26。
任一逻辑通道在所有别名中都不存在 → 跳过该受试者。

每个 pkl: {
  'eeg':     (S, 11, 120000) float32   # 20 个 30s epoch 在时间维 concat
  'labels':  (S, 20)        int64      # 每个 epoch 一个标签，20 个/样本
  'ch_names': [11,]
}
标签映射（与 CBraMod prepare_ISRUC_1.py 一致）：
  0=Wake, 1=N1, 2=N2, 3=N3, 5=REM -> 0,1,2,3,4

处理流程（与 CBraMod 一致）：0.3-35 Hz 带通 + 50 Hz 陷波；30s epoch = 6000 点；
每 20 个 epoch 组成一个样本（trial），尾部不足 20 的 epoch 丢弃。
"""
import argparse
import os
import pickle
import sys
import numpy as np
from tqdm import tqdm

# 使用 CBraMod 的补丁版 EDF 读取器（允许 .rec 扩展名）
CBraMod_PREPROC_DIR = '/data_main/junlin/BFM/Models/CBraMod/preprocessing/ISRUC'
sys.path.insert(0, CBraMod_PREPROC_DIR)
from edf_ import read_raw_edf  # noqa: E402

ROOT = '/data_main/junlin/BFM/Datasets/ISRUC'
OUT_ROOT = os.path.join(ROOT, 'Preproc-PKL')

# EOG(2) + EEG(6) + EMG(3)，按 .rec 文件内通道顺序（规范名，存入 pkl 的 ch_names）
CH_NAMES = ['LOC-A2', 'ROC-A1',
            'F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1',
            'X1', 'X2', 'X3']

# 每个逻辑通道的别名（不同受试者参考电极命名不同）
CHANNEL_ALIASES = [
    ['LOC-A2', 'E1-M2'],   # EOG 左眼
    ['ROC-A1', 'E2-M1'],   # EOG 右眼
    ['F3-A2', 'F3-M2'],    # EEG F3
    ['C3-A2', 'C3-M2'],    # EEG C3
    ['O1-A2', 'O1-M2'],    # EEG O1
    ['F4-A1', 'F4-M1'],    # EEG F4
    ['C4-A1', 'C4-M1'],    # EEG C4
    ['O2-A1', 'O2-M1'],    # EEG O2
    ['X1', '24'],          # 下颌 EMG 1
    ['X2', '25'],          # 下颌 EMG 2
    ['X3', '26'],          # 下颌 EMG 3
]

LABEL2ID = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}   # REM(5) -> 4

SFREQ = 200
EPOCH_SAMPLES = 30 * SFREQ      # 6000
SEQ_LEN = 20                    # 每个样本 20 个 epoch


def build_subject(subject_id):
    rec = os.path.join(ROOT, str(subject_id), f'{subject_id}.rec')
    label_file = os.path.join(ROOT, str(subject_id), f'{subject_id}_1.txt')

    raw = read_raw_edf(rec, preload=True)

    # 通道别名归一化：任一逻辑通道缺失则跳过该受试者
    present, missing = [], []
    for aliases in CHANNEL_ALIASES:
        hit = next((a for a in aliases if a in raw.ch_names), None)
        if hit is None:
            missing.append(aliases[0])
        else:
            present.append(hit)
    if missing:
        return None, missing

    raw.pick_channels(present)
    raw.filter(0.3, 35, fir_design='firwin')
    raw.notch_filter(50)

    # (11, n_times) -> (n_times, 11)；EDF 通道单位为 V，强制转 µV
    data = raw.get_data(units='uV').T

    # 按 30s 切 epoch
    n_epochs = data.shape[0] // EPOCH_SAMPLES
    data = data[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES, len(CH_NAMES))

    # 标签：每行一个 epoch
    labels = []
    for line in open(label_file):
        line = line.strip()
        if line:
            labels.append(LABEL2ID[line])
    labels = np.array(labels, dtype=np.int64)

    # 信号与标签对齐（理论上相等，取小者保险）
    n_epochs = min(n_epochs, len(labels))
    data = data[:n_epochs]
    labels = labels[:n_epochs]

    # 每 20 个 epoch 组成一个样本，尾部不足 20 的丢弃
    n_trials = n_epochs // SEQ_LEN
    data = data[:n_trials * SEQ_LEN]
    labels = labels[:n_trials * SEQ_LEN]

    # (n_trials, 20, 6000, 11) -> (n_trials, 20, 11, 6000) -> (n_trials, 11, 120000)
    data = data.reshape(n_trials, SEQ_LEN, EPOCH_SAMPLES, len(CH_NAMES)) \
               .transpose(0, 1, 3, 2) \
               .reshape(n_trials, len(CH_NAMES), SEQ_LEN * EPOCH_SAMPLES)
    labels = labels.reshape(n_trials, SEQ_LEN)

    return {
        'eeg': data.astype(np.float32),
        'labels': labels.astype(np.int64),
        'ch_names': CH_NAMES,
    }, None


def parse_subjects(s):
    """支持 '1,2,3' 或 '1-10' 或组合 '1-5,8,10-20'。"""
    ids = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISRUC 预处理')
    parser.add_argument('--subjects', type=str, default='1-100', help='受试者 id，如 1-100 或 1,2,3')
    parser.add_argument('--out', type=str, default='Preproc-PKL', help='输出目录名（相对于 ROOT）')
    args = parser.parse_args()

    out_root = os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    subjects = parse_subjects(args.subjects)
    for subject in tqdm(subjects, desc='ISRUC preprocess'):
        out_path = os.path.join(out_root, f'{subject}.pkl')
        if os.path.exists(out_path):
            print(f'subject {subject}: already exists, skip')
            continue
        data, missing = build_subject(subject)
        if data is None:
            print(f'subject {subject}: SKIP (missing channels: {missing})')
            continue
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'subject {subject}: eeg {data["eeg"].shape}, labels {data["labels"].shape} -> {out_path}')
