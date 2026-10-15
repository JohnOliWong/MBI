#!/usr/bin/env python3
"""SEED-V 预处理：把 EEG_Raw(.cnt) + Eye_Movement_Features(.npz) 转换为按受试者存储的 .pkl（EF 风格）。

每个 pkl: {'eeg': (S, 62, 800*K) float32, 'eye': (S, 33, K) float32, 'labels': (S,) int64}
样本 = K 个连续 4s 段（同一 trial 内不重叠，尾部不足 K 段的部分丢弃），
同一受试者的 3 session x 15 trial 全部拼接。留一受试者交叉验证用。

处理流程与官方 preprocessing_seedv.py 一致：
  .cnt -> 去除 M1/M2/VEO/HEO -> 重采样 200Hz -> 0.3-75Hz 带通滤波 -> 按 trial 时间戳切分。
眼动特征使用官方 npz（每 4s 段 33 维）。
"""
import argparse
import os
import pickle
import warnings
import numpy as np
import mne

# .cnt 文件头无有效日期，忽略 mne 的Runtime Warning
# 注意：mne 的告警文本以两个空格开头（"  Could not parse meas date ..."），
# filterwarnings 的 message 是 re.match 锚定开头的正则，因此用 \s* 吞掉前导空白。
warnings.filterwarnings('ignore', message=r'\s*Could not parse meas date', category=RuntimeWarning)

ROOT = '/data_main/junlin/BFM/Datasets/SEED-V'
OUT_ROOT = os.path.join(ROOT, 'Preproc-PKL')

# 每个 session 中 15 个 trial 的开始/结束位置（秒）
TRIALS_OF_SESSIONS = {
	'1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
	      'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},
	'2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
	      'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},
	'3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
	      'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}
LABELS_OF_SESSIONS = {
	'1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
	'2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
	'3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}
USELESS_CH = ['M1', 'M2', 'VEO', 'HEO']
STANDARD_CH = None   # 缓存标准 62 通道顺序（仅 PADDING=True 时使用）
PADDING = False      # True：通道数不足 62 的文件按标准顺序补零对齐；False：直接跳过该 session

SEG_LEN = 800   # 一个 4s 段的采样点数（200Hz * 4s）
EEG_CH = 62
EYE_FEATS = 33


def find_cnt_file(subject, session):
	"""找到受试者 subject 的第 session 个 session 的 .cnt 文件。
	受试者 7 的 session 1 优先用原始文件，失败时回退到 repaired 版本。"""
	prefix = f'{subject}_{session}_'
	files = [f for f in os.listdir(os.path.join(ROOT, 'EEG_Raw')) if f.startswith(prefix) and f.endswith('.cnt')]
	originals = sorted([f for f in files if 'repaired' not in f])
	repaired = sorted([f for f in files if 'repaired' in f])
	return originals, repaired


def load_cnt(subject, session):
	global STANDARD_CH
	originals, repaired = find_cnt_file(subject, session)
	paths = [os.path.join(ROOT, 'EEG_Raw', f) for f in originals + repaired]
	last_err = None
	for path in paths:
		try:
			raw = mne.io.read_raw_cnt(path, preload=True, verbose='ERROR')
			remove_ch = [ch for ch in USELESS_CH if ch in raw.ch_names]
			if remove_ch:
				raw.drop_channels(remove_ch)
			raw.resample(200, verbose='ERROR')
			raw.filter(l_freq=0.3, h_freq=75, verbose='ERROR')
			data_matrix = raw.get_data(units='uV')   # (n_ch, L)
			names = raw.ch_names
			if STANDARD_CH is None and len(names) == 62:
				STANDARD_CH = names
			if len(names) != 62:
				if PADDING and STANDARD_CH is not None:
					# 缺通道（如 7_1 repaired 缺 FC4）：按标准顺序对齐，缺失通道补零
					aligned = np.zeros((62, data_matrix.shape[1]), dtype=data_matrix.dtype)
					for i, name in enumerate(names):
						if name in STANDARD_CH:
							aligned[STANDARD_CH.index(name)] = data_matrix[i]
					print(f'    [align] {os.path.basename(path)}: {len(names)} -> 62 channels (zero-padded)')
					data_matrix = aligned
				else:
					raise ValueError(f'channel count {len(names)} != 62 (PADDING={PADDING})')
			return data_matrix
		except Exception as e:
			last_err = e
			continue
	raise RuntimeError(f'failed to load cnt for subject {subject} session {session}: {last_err}')


def load_eye_and_labels(subject):
	npz = np.load(os.path.join(ROOT, 'Eye_Movement_Features', f'{subject}_123.npz'), allow_pickle=True)
	eye = pickle.loads(npz['data'].item())    # dict: key = (session-1)*15 + trial -> (W, 33)
	labels = pickle.loads(npz['label'].item())
	return eye, labels


def build_subject(subject, K):
	eye_dict, label_dict = load_eye_and_labels(subject)
	eeg_list, eye_list, label_list = [], [], []
	for session in (1, 2, 3):
		session_str = str(session)
		try:
			data_matrix = load_cnt(subject, session)
		except Exception as e:
			print(f'  [skip] subject {subject} session {session}: {e}')
			continue
		starts = TRIALS_OF_SESSIONS[session_str]['start']
		ends = TRIALS_OF_SESSIONS[session_str]['end']
		for trial in range(15):
			key = (session - 1) * 15 + trial
			label = int(np.unique(label_dict[key])[0])
			eye_feat = np.asarray(eye_dict[key])                    # (W, 33)
			eeg_trial = data_matrix[:, starts[trial] * 200: ends[trial] * 200]   # (62, L)

			n_seg = min(eeg_trial.shape[1] // SEG_LEN, eye_feat.shape[0])
			usable = (n_seg // K) * K
			if usable == 0:
				continue
			n_sample = usable // K

			eeg_chunks = eeg_trial[:, :usable * SEG_LEN].reshape(EEG_CH, n_sample, SEG_LEN * K).transpose(1, 0, 2)
			eye_chunks = eye_feat[:usable].reshape(n_sample, K, EYE_FEATS).transpose(0, 2, 1)

			eeg_list.append(eeg_chunks)
			eye_list.append(eye_chunks)
			label_list.append(np.full(n_sample, label, dtype=np.int64))

	eeg = np.concatenate(eeg_list, axis=0).astype(np.float32)
	eye = np.concatenate(eye_list, axis=0).astype(np.float32)
	labels = np.concatenate(label_list, axis=0).astype(np.int64)
	return {'eeg': eeg, 'eye': eye, 'labels': labels}


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SEED-V 预处理')
	parser.add_argument('--K', type=int, default=5, help='每个样本包含的 4s 段数')
	parser.add_argument('--out', type=str, default='Preproc-PKL', help='输出目录名（相对于 ROOT）')
	args = parser.parse_args()

	out_root = os.path.join(ROOT, args.out)
	os.makedirs(out_root, exist_ok=True)
	subjects = list(range(1, 17))
	for subject in subjects:
		out_path = os.path.join(out_root, f'{subject}.pkl')
		# 跳过已经生成的pickle文件
		if os.path.exists(out_path):
			print(f'subject {subject}: already exists, skip')
			continue
		data = build_subject(subject, args.K)
		with open(out_path, 'wb') as f:
			pickle.dump(data, f)
		print(f'subject {subject}: eeg {data["eeg"].shape}, eye {data["eye"].shape}, labels {data["labels"].shape} -> {out_path}')