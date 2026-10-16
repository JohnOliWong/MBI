import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


# =====================================================================
# ISRUC（睡眠分期，5 分类）：EEG(6) + EOG(2) + EMG(3) = 11 通道 @ 200Hz
#   pkl: {'eeg':[S,11,120000] float32, 'labels':[S,20] int64, 'ch_names':[11]}
#   eeg 的 120000 = 20 个 30s epoch（每 epoch 6000 点）在时间维 concat
#   标签 0=Wake 1=N1 2=N2 3=N3 4=REM
#
# 标准化策略：pkl 里存原始 µV，统计值（每受试者每通道 median/MAD）首次计算后
#   原子回写进 pkl；之后直接读统计值。z-score 在 Dataset 内完成，可随时换 scaler。
#
# 注意：ISRUCDataset 会把所有受试者数据拼进内存（float32），
#   全部 98 人约 23GB（train 78 人约 18GB），若内存不足可改为按受试者懒加载。
# =====================================================================

ROOT = '/data_main/junlin/BFM/Datasets/ISRUC'
PKL_ROOT = os.path.join(ROOT, 'Preproc-PKL')

N_CH = 11
EPOCH_SAMPLES = 6000    # 30s @ 200Hz
SEQ_LEN = 20            # 每个 trial 20 个 epoch
N_CLASS = 5

# 通道索引（按 ch_names 顺序：EOG(0-1), EEG(2-7), EMG(8-10)）
EOG_CH = slice(0, 2)   # LOC-A2, ROC-A1
EEG_CH = slice(2, 8)   # F3, C3, O1, F4, C4, O2
EMG_CH = slice(8, 11)  # X1, X2, X3

# 模态名 -> 通道切片
MODAL_SLICE = {'eeg': EEG_CH, 'emg': EMG_CH, 'eog': EOG_CH}

# 预处理跳过的受试者
SKIP_SUBJECTS = {8, 40}
VALID_SUBJECTS = [i for i in range(1, 101) if i not in SKIP_SUBJECTS]

# holdout 默认划分（对齐 CBraMod 的 80/10/10，受试者 id 分段）
DEFAULT_TRAIN = [i for i in VALID_SUBJECTS if 1 <= i <= 80]
DEFAULT_VAL = [i for i in VALID_SUBJECTS if 81 <= i <= 90]
DEFAULT_TEST = [i for i in VALID_SUBJECTS if 91 <= i <= 100]


def get_or_compute_stats(pkl_path):
    '''读 pkl；若缺少 ch_median/ch_mad，则按原始 eeg 逐通道计算并原子回写。

    统计量为 robust 统计：median 与 MAD（对 ±19000 µV 的电极尖峰不敏感）。
    返回完整 dict（含 ch_median/ch_mad，float64）。'''
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if 'ch_median' in data and 'ch_mad' in data:
        return data

    eeg = data['eeg']                                          # [S, 11, 120000]
    median = np.median(eeg, axis=(0, 2)).astype(np.float64)    # [11]
    mad = np.median(np.abs(eeg - median[None, :, None]), axis=(0, 2)).astype(np.float64)  # [11]
    data['ch_median'] = median
    data['ch_mad'] = mad

    # 原子回写：先写临时文件再替换，避免中途中断损坏原文件
    tmp = pkl_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp, pkl_path)
    return data


def standardize(eeg, median, mad, clip=None):
    '''robust z-score：(x - median) / (1.4826 * mad)，可选 clip 到 [-clip, clip]。'''
    z = (eeg - median[None, :, None]) / (1.4826 * mad[None, :, None])
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z


class ISRUCDataset(Dataset):
    '''返回 (eeg, labels)：
    默认（trial 形式）：eeg [20, 11, 6000]（20 个 epoch，已 z-score），labels [20]
    flatten_epochs=True：eeg [11, 6000]（单个 epoch），labels 标量 —— 供单 epoch 分类器（如 EEGNet）使用
    '''
    def __init__(self, subjects, clip=8.0, flatten_epochs=False):
        eeg_list, label_list = [], []
        for i in subjects:
            pkl_path = os.path.join(PKL_ROOT, f'{i}.pkl')
            data = get_or_compute_stats(pkl_path)
            eeg = standardize(data['eeg'], data['ch_median'], data['ch_mad'], clip)
            # [S, 11, 120000] -> [S, 11, 20, 6000] -> [S, 20, 11, 6000]
            eeg = eeg.reshape(-1, N_CH, SEQ_LEN, EPOCH_SAMPLES).transpose(0, 2, 1, 3)
            eeg_list.append(torch.from_numpy(np.ascontiguousarray(eeg)))
            label_list.append(torch.from_numpy(data['labels']))
        self.eeg = torch.cat(eeg_list, dim=0)      # [N_total, 20, 11, 6000]
        self.labels = torch.cat(label_list, dim=0)  # [N_total, 20]

        if flatten_epochs:
            self.eeg = self.eeg.reshape(-1, N_CH, EPOCH_SAMPLES)  # [N*20, 11, 6000]
            self.labels = self.labels.reshape(-1)                 # [N*20]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx].float(), self.labels[idx].long()


def read_isruc_split(train_subs=None, val_subs=None, test_subs=None, clip=8.0, flatten_epochs=False):
    '''holdout 划分读取，返回 (train_set, val_set, test_set)。
    默认 78/10/10 受试者（train=1-80 去掉 #8/#40，val=81-90，test=91-100）。
    flatten_epochs=True 时把每个 trial 的 20 个 epoch 展开为独立样本（单 epoch 分类器用）。'''
    train_subs = DEFAULT_TRAIN if train_subs is None else train_subs
    val_subs = DEFAULT_VAL if val_subs is None else val_subs
    test_subs = DEFAULT_TEST if test_subs is None else test_subs

    train_set = ISRUCDataset(train_subs, clip=clip, flatten_epochs=flatten_epochs)
    val_set = ISRUCDataset(val_subs, clip=clip, flatten_epochs=flatten_epochs)
    test_set = ISRUCDataset(test_subs, clip=clip, flatten_epochs=flatten_epochs)

    print(f'ISRUC Train {len(train_subs)} subs: eeg {tuple(train_set.eeg.shape)}, labels {tuple(train_set.labels.shape)}')
    print(f'ISRUC Val   {len(val_subs)} subs: eeg {tuple(val_set.eeg.shape)}, labels {tuple(val_set.labels.shape)}')
    print(f'ISRUC Test  {len(test_subs)} subs: eeg {tuple(test_set.eeg.shape)}, labels {tuple(test_set.labels.shape)}')
    return train_set, val_set, test_set


class ISRUCDualDataset(Dataset):
    '''双模态（EEG + EMG）数据集，返回 (eeg, emg, labels)。
    eeg: EEG_CH 6 通道，emg: EMG_CH 3 通道（均为 @200Hz）。
    flatten_epochs=True（默认）：eeg [6,6000]，emg [3,6000]，labels 标量 —— 单 epoch 分类器用
    flatten_epochs=False：eeg [20,6,6000]，emg [20,3,6000]，labels [20]（trial 级）
    '''
    def __init__(self, subjects, clip=8.0, flatten_epochs=True):
        eeg_list, emg_list, label_list = [], [], []
        for i in subjects:
            pkl_path = os.path.join(PKL_ROOT, f'{i}.pkl')
            data = get_or_compute_stats(pkl_path)
            x = standardize(data['eeg'], data['ch_median'], data['ch_mad'], clip)
            # [S, 11, 120000] -> [S, 11, 20, 6000] -> [S, 20, 11, 6000]
            x = x.reshape(-1, N_CH, SEQ_LEN, EPOCH_SAMPLES).transpose(0, 2, 1, 3)
            if flatten_epochs:
                x = x.reshape(-1, N_CH, EPOCH_SAMPLES)      # [S*20, 11, 6000]
                labels = data['labels'].reshape(-1)         # [S*20]
                eeg = x[:, EEG_CH, :]                       # [S*20, 6, 6000]
                emg = x[:, EMG_CH, :]                       # [S*20, 3, 6000]
            else:
                labels = data['labels']                     # [S, 20]
                eeg = x[:, :, EEG_CH, :]                    # [S, 20, 6, 6000]
                emg = x[:, :, EMG_CH, :]                    # [S, 20, 3, 6000]
            eeg_list.append(torch.from_numpy(np.ascontiguousarray(eeg)))
            emg_list.append(torch.from_numpy(np.ascontiguousarray(emg)))
            label_list.append(torch.from_numpy(labels))
        self.eeg = torch.cat(eeg_list, dim=0)
        self.emg = torch.cat(emg_list, dim=0)
        self.labels = torch.cat(label_list, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx].float(), self.emg[idx].float(), self.labels[idx].long()


def read_isruc_dual_split(train_subs=None, val_subs=None, test_subs=None, clip=8.0, flatten_epochs=True):
    '''双模态（EEG+EMG）holdout 划分读取，返回 (train_set, val_set, test_set)。
    默认 78/10/10 受试者；flatten_epochs=True（默认）把每个 trial 的 20 个 epoch 展开为独立样本。'''
    train_subs = DEFAULT_TRAIN if train_subs is None else train_subs
    val_subs = DEFAULT_VAL if val_subs is None else val_subs
    test_subs = DEFAULT_TEST if test_subs is None else test_subs

    train_set = ISRUCDualDataset(train_subs, clip=clip, flatten_epochs=flatten_epochs)
    val_set = ISRUCDualDataset(val_subs, clip=clip, flatten_epochs=flatten_epochs)
    test_set = ISRUCDualDataset(test_subs, clip=clip, flatten_epochs=flatten_epochs)

    print(f'ISRUC-Dual Train {len(train_subs)} subs: eeg {tuple(train_set.eeg.shape)}, emg {tuple(train_set.emg.shape)}, labels {tuple(train_set.labels.shape)}')
    print(f'ISRUC-Dual Val   {len(val_subs)} subs: eeg {tuple(val_set.eeg.shape)}, emg {tuple(val_set.emg.shape)}, labels {tuple(val_set.labels.shape)}')
    print(f'ISRUC-Dual Test  {len(test_subs)} subs: eeg {tuple(test_set.eeg.shape)}, emg {tuple(test_set.emg.shape)}, labels {tuple(test_set.labels.shape)}')
    return train_set, val_set, test_set


class ISRUCModalDataset(Dataset):
    '''按模态列表返回数据。modalities: ['eeg'] / ['eeg','emg'] / ['eeg','emg','eog']。
    __getitem__ 返回 (modal_0, modal_1, ..., labels)，模态顺序与 modalities 一致。'''
    def __init__(self, subjects, modalities, clip=8.0, flatten_epochs=True):
        self.modalities = list(modalities)
        modal_list = {name: [] for name in self.modalities}
        label_list = []
        for i in subjects:
            pkl_path = os.path.join(PKL_ROOT, f'{i}.pkl')
            data = get_or_compute_stats(pkl_path)
            x = standardize(data['eeg'], data['ch_median'], data['ch_mad'], clip)
            # [S, 11, 120000] -> [S, 11, 20, 6000] -> [S, 20, 11, 6000]
            x = x.reshape(-1, N_CH, SEQ_LEN, EPOCH_SAMPLES).transpose(0, 2, 1, 3)
            if flatten_epochs:
                x = x.reshape(-1, N_CH, EPOCH_SAMPLES)          # [S*20, 11, 6000]
                labels = data['labels'].reshape(-1)             # [S*20]
                for name in self.modalities:
                    modal_list[name].append(torch.from_numpy(np.ascontiguousarray(x[:, MODAL_SLICE[name], :])))
            else:
                labels = data['labels']                         # [S, 20]
                for name in self.modalities:
                    modal_list[name].append(torch.from_numpy(np.ascontiguousarray(x[:, :, MODAL_SLICE[name], :])))
            label_list.append(torch.from_numpy(labels))
        self.data = {name: torch.cat(v, dim=0) for name, v in modal_list.items()}
        self.labels = torch.cat(label_list, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return tuple(self.data[name][idx].float() for name in self.modalities) + (self.labels[idx].long(),)


def read_isruc_modal_split(modalities, train_subs=None, val_subs=None, test_subs=None, clip=8.0, flatten_epochs=True):
    '''多模态 holdout 划分读取，返回 (train_set, val_set, test_set)。
    modalities: 模态名列表，如 ['eeg'], ['eeg','emg'], ['eeg','emg','eog']。'''
    train_subs = DEFAULT_TRAIN if train_subs is None else train_subs
    val_subs = DEFAULT_VAL if val_subs is None else val_subs
    test_subs = DEFAULT_TEST if test_subs is None else test_subs

    train_set = ISRUCModalDataset(train_subs, modalities, clip=clip, flatten_epochs=flatten_epochs)
    val_set = ISRUCModalDataset(val_subs, modalities, clip=clip, flatten_epochs=flatten_epochs)
    test_set = ISRUCModalDataset(test_subs, modalities, clip=clip, flatten_epochs=flatten_epochs)

    def _shape(ds):
        return {name: tuple(ds.data[name].shape) for name in modalities}
    print(f'ISRUC-Modal Train {len(train_subs)} subs: {_shape(train_set)}, labels {tuple(train_set.labels.shape)}')
    print(f'ISRUC-Modal Val   {len(val_subs)} subs: {_shape(val_set)}, labels {tuple(val_set.labels.shape)}')
    print(f'ISRUC-Modal Test  {len(test_subs)} subs: {_shape(test_set)}, labels {tuple(test_set.labels.shape)}')
    return train_set, val_set, test_set


if __name__ == '__main__':
    train_set, val_set, test_set = read_isruc_split()
    eeg, labels = train_set[0]
    print('sample eeg:', tuple(eeg.shape), 'labels:', tuple(labels.shape))
    print('label values:', labels.tolist())
