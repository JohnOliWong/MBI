#!/usr/bin/env python3
"""SEED-IV / SEED-V PKL 数据完整性校验。

检查 6 个目录：
  SEED-IV/{PKL-K1, PKL-K3, PKL-K5}
  SEED-V/{PKL-K1, PKL-K3, PKL-K5}

对每个受试者 pkl 校验：
  - 文件存在性（SEED-IV 15 人 / SEED-V 16 人）
  - keys / dtype / shape 是否符合对应 K 的约定
  - eeg、eye、labels 三者样本数是否一致
  - eeg / eye 是否含 NaN / Inf
  - labels 是否落在合法类别范围内
"""

import argparse
import os
import pickle
import sys

import numpy as np

DEFAULT_BASE = '/data_main/junlin/BFM/Datasets'

# 每个数据集的约定
CONFIGS = {
    'SEED-IV': {'num_subjects': 15, 'num_classes': 4, 'eye_feats': 31},
    'SEED-V': {'num_subjects': 16, 'num_classes': 5, 'eye_feats': 33},
}
# 目录名 -> K
K_VARIANTS = {'PKL-K1': 1, 'PKL-K3': 3, 'PKL-K5': 5}

EEG_CH = 62
SEG_LEN = 800  # 每个 4s 段的采样点数（200Hz * 4s）


def check_subject_pkl(path, eye_feats, K, num_classes):
    """校验单个 pkl，返回 (errors, n_samples)。"""
    errors = []
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return [f'无法读取: {type(e).__name__}: {e}'], None

    if not isinstance(data, dict):
        return [f'内容不是 dict，而是 {type(data).__name__}'], None

    if set(data.keys()) != {'eeg', 'eye', 'labels'}:
        errors.append(f'keys 应为 eeg/eye/labels，实际 {sorted(data.keys())}')
        return errors, None

    eeg, eye, labels = data['eeg'], data['eye'], data['labels']

    if not isinstance(eeg, np.ndarray) or not isinstance(eye, np.ndarray) or not isinstance(labels, np.ndarray):
        errors.append('字段存在非 ndarray 类型')
        return errors, None

    S = eeg.shape[0]

    # 形状
    if eeg.shape != (S, EEG_CH, SEG_LEN * K):
        errors.append(f'eeg.shape={eeg.shape}, 期望=({S},{EEG_CH},{SEG_LEN * K})')
    if eye.shape != (S, eye_feats, K):
        errors.append(f'eye.shape={eye.shape}, 期望=({S},{eye_feats},{K})')
    if labels.shape != (S,):
        errors.append(f'labels.shape={labels.shape}, 期望=({S},)')

    # 样本数一致
    if not (len(eeg) == len(eye) == len(labels)):
        errors.append(f'样本数不一致: eeg={len(eeg)}, eye={len(eye)}, labels={len(labels)}')

    # 非空
    if S == 0:
        errors.append('样本数为 0')

    # dtype
    if eeg.dtype != np.float32:
        errors.append(f'eeg.dtype={eeg.dtype}, 期望 float32')
    if eye.dtype != np.float32:
        errors.append(f'eye.dtype={eye.dtype}, 期望 float32')
    if labels.dtype != np.int64:
        errors.append(f'labels.dtype={labels.dtype}, 期望 int64')

    # NaN / Inf
    if not np.isfinite(eeg).all():
        errors.append('eeg 含 NaN/Inf')
    if not np.isfinite(eye).all():
        errors.append('eye 含 NaN/Inf')

    # 标签范围
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        errors.append(f'label 越界，范围应为 [0,{num_classes})，实际 [{labels.min()},{labels.max()}]')

    return errors, S


def main():
    parser = argparse.ArgumentParser(description='SEED PKL 数据完整性校验')
    parser.add_argument('--base', default=DEFAULT_BASE,
                        help='数据集根目录，默认 %s' % DEFAULT_BASE)
    args = parser.parse_args()

    ok = True
    for ds, cfg in CONFIGS.items():
        for dirname, K in K_VARIANTS.items():
            root = os.path.join(args.base, ds, dirname)
            tag = f'[{ds}/{dirname}]'
            if not os.path.isdir(root):
                print(f'{tag} 目录不存在: {root}')
                ok = False
                continue

            expected = [f'{s}.pkl' for s in range(1, cfg['num_subjects'] + 1)]
            missing = [s for s in range(1, cfg['num_subjects'] + 1)
                       if not os.path.isfile(os.path.join(root, f'{s}.pkl'))]
            extras = sorted(set(os.listdir(root)) - set(expected))

            if missing:
                print(f'{tag} 缺少受试者文件: {missing}')
                ok = False
            if extras:
                print(f'{tag} 多余文件: {extras}')
                ok = False

            counts = {}
            for s in range(1, cfg['num_subjects'] + 1):
                p = os.path.join(root, f'{s}.pkl')
                if not os.path.isfile(p):
                    continue
                errs, n = check_subject_pkl(p, cfg['eye_feats'], K, cfg['num_classes'])
                if errs:
                    print(f'{tag} subject {s}: ' + '; '.join(errs))
                    ok = False
                else:
                    counts[s] = n

            if counts:
                print(f'{tag} 通过 {len(counts)}/{cfg["num_subjects"]} 个受试者；'
                      f'各受试者样本数: {dict(sorted(counts.items()))}')
            else:
                print(f'{tag} 无受试者通过检查')

    print()
    if ok:
        print('ALL CHECKS PASSED ✅')
        return 0
    print('发现完整性/格式问题 ❌')
    return 1


if __name__ == '__main__':
    sys.exit(main())