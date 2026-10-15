#!/usr/bin/env python3
"""SEED-IV 预处理：把 eeg_raw_data + eye_feature_smooth 转换为按受试者存储的 .pkl（EF 风格）。

每个 pkl: {'eeg': (S, 62, 800*K) float32, 'eye': (S, 31, K) float32, 'labels': (S,) int64}
样本 = K 个连续 4s 段（同一 trial 内不重叠，尾部不足 K 段的部分丢弃），
同一受试者的 3 session x 24 trial 全部拼接。留一受试者交叉验证用。
"""
import argparse
import os
import pickle
import numpy as np
import scipy.io

ROOT = '/data_main/junlin/BFM/Datasets/SEED-IV'
OUT_ROOT = os.path.join(ROOT, 'Preproc-PKL')

SUBJECTS = ['cz', 'ha', 'hql', 'ldy', 'ly', 'mhw', 'mz', 'qyt', 'rx', 'tyc', 'whh', 'wll', 'wq', 'zjd', 'zjy']
LABELS = {
	'1': [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
	'2': [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
	'3': [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
SEG_LEN = 800   # 一个 4s 段的采样点数（200Hz * 4s）
EEG_CH = 62
EYE_FEATS = 31


def build_subject(subject, K):
	eeg_list, eye_list, label_list = [], [], []
	for session in ('1', '2', '3'):
		eeg_dir = os.path.join(ROOT, 'eeg_raw_data', session)
		eye_dir = os.path.join(ROOT, 'eye_feature_smooth', session)
		eeg_files = sorted(os.listdir(eeg_dir), key=lambda f: int(f.split('_')[0]))
		eye_files = sorted(os.listdir(eye_dir), key=lambda f: int(f.split('_')[0]))

		eeg_file = eeg_files[subject - 1]
		eye_file = eye_files[subject - 1]
		eeg_mat = scipy.io.loadmat(os.path.join(eeg_dir, eeg_file), squeeze_me=True, struct_as_record=False)
		eye_mat = scipy.io.loadmat(os.path.join(eye_dir, eye_file), squeeze_me=True, struct_as_record=False)

		sub_name = SUBJECTS[subject - 1]
		for trial in range(1, 25):
			label = LABELS[session][trial - 1]
			eeg_raw = eeg_mat[f'{sub_name}_eeg{trial}']   # (62, L) @ 200Hz
			eye_feat = eye_mat[f'eye_{trial}']            # (31, W)，W 为 4s 段数

			n_seg = min(eeg_raw.shape[1] // SEG_LEN, eye_feat.shape[1])
			usable = (n_seg // K) * K
			if usable == 0:
				continue
			n_sample = usable // K

			eeg_chunks = eeg_raw[:, :usable * SEG_LEN].reshape(EEG_CH, n_sample, SEG_LEN * K).transpose(1, 0, 2)
			eye_chunks = eye_feat[:, :usable].reshape(EYE_FEATS, n_sample, K).transpose(1, 0, 2)

			eeg_list.append(eeg_chunks)
			eye_list.append(eye_chunks)
			label_list.append(np.full(n_sample, label, dtype=np.int64))

	eeg = np.concatenate(eeg_list, axis=0).astype(np.float32)
	eye = np.concatenate(eye_list, axis=0).astype(np.float32)
	labels = np.concatenate(label_list, axis=0).astype(np.int64)
	return {'eeg': eeg, 'eye': eye, 'labels': labels}


def main():
	parser = argparse.ArgumentParser(description='SEED-IV 预处理')
	parser.add_argument('--K', type=int, default=5, help='每个样本包含的 4s 段数')
	parser.add_argument('--out', type=str, default='Preproc-PKL', help='输出目录名（相对于 ROOT）')
	args = parser.parse_args()

	out_root = os.path.join(ROOT, args.out)
	os.makedirs(out_root, exist_ok=True)
	for subject in range(1, 16):
		data = build_subject(subject, args.K)
		out_path = os.path.join(out_root, f'{subject}.pkl')
		with open(out_path, 'wb') as f:
			pickle.dump(data, f)
		print(f'subject {subject}: eeg {data["eeg"].shape}, eye {data["eye"].shape}, labels {data["labels"].shape} -> {out_path}')


if __name__ == '__main__':
	main()
