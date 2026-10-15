import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
import pickle


# =====================================================================
# SEED-IV / SEED-V（本目录专用）：EEG(62ch@200Hz) + 眼动特征
#   SEED-IV：眼动 31 维/4s 段，4 类；  SEED-V：眼动 33 维/4s 段，5 类
# 样本 = K 个连续 4s 段；同一受试者的全部 session/trial 拼接（留一交叉验证）
# eeg: [S, 1, 62, 800*K]  eye: [S, 1, eye_feats, K]  labels: [S,]
# 数据由 preprocess_seediv.py / preprocess_seedv.py 生成按受试者的 .pkl
# =====================================================================

SEED_IV_ROOT = '/data_main/junlin/BFM/Datasets/SEED-IV'
SEED_V_ROOT = '/data_main/junlin/BFM/Datasets/SEED-V'


def get_seed_pkl_root(dataset, K):
	'''返回 SEED-IV/SEED-V 的 PKL-K{K} 目录；目录不存在时抛 RuntimeError。'''
	base = SEED_IV_ROOT if dataset == 'SEEDIV' else SEED_V_ROOT
	root = os.path.join(base, f'PKL-K{K}')
	if not os.path.isdir(root):
		raise RuntimeError(f'PKL-K{K} 目录不存在: {root}。请先运行 preprocess_seediv.py/preprocess_seedv.py --K {K} 生成，或检查 --K 参数。')
	return root


class SeedDataset(Dataset):
	'''SEED 数据集：__getitem__ 返回 (eeg, eye, label)'''
	def __init__(self, data):
		self.eeg = data['eeg']
		self.eye = data['eye']
		self.labels = data['labels']

		assert len(self.eeg) == len(self.eye) == len(self.labels), 'Data lengths do not match'

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.eeg[index].float(), self.eye[index].float(), self.labels[index].long()


def read_seed_pkls(subject, dataset, dim=4, num_sub=15, K=1):
	'''通用 LOSO 读取（EF 风格）：读按受试者的 .pkl，其余受试者样本拼接为训练集，
	当前受试者为测试集。
	eeg: [S, 1, 62, 800*K] / dim=3 时为 [S, 62, 800*K]
	eye: [S, 1, eye_feats, K] / dim=3 时为 [S, eye_feats, K]'''
	root = get_seed_pkl_root(dataset, K)
	training_set = {}
	testing_set = {}
	train_eeg, train_eye, train_labels = [], [], []

	for i in range(1, num_sub + 1):
		with open(f'{root}/{i}.pkl', 'rb') as f:
			data = pickle.load(f)
		eeg = torch.tensor(data['eeg'], dtype=torch.float32)
		eye = torch.tensor(data['eye'], dtype=torch.float32)
		labels = torch.tensor(data['labels'], dtype=torch.long)
		if i == subject:
			if dim == 4:
				eeg = eeg.unsqueeze(1)
				eye = eye.unsqueeze(1)
			testing_set = {'eeg': eeg, 'eye': eye, 'labels': labels}
		else:
			train_eeg.append(eeg)
			train_eye.append(eye)
			train_labels.append(labels)

	training_set['eeg'] = torch.cat(train_eeg, dim=0)
	training_set['eye'] = torch.cat(train_eye, dim=0)
	training_set['labels'] = torch.cat(train_labels, dim=0)
	if dim == 4:
		training_set['eeg'] = training_set['eeg'].unsqueeze(1)
		training_set['eye'] = training_set['eye'].unsqueeze(1)

	print(f'{dataset} {subject} Train Set', 'EEG', training_set['eeg'].shape, 'Eye', training_set['eye'].shape, 'Labels', training_set['labels'].shape)
	print(f'{dataset} {subject} Test Set', 'EEG', testing_set['eeg'].shape, 'Eye', testing_set['eye'].shape, 'Labels', testing_set['labels'].shape)

	return SeedDataset(training_set), SeedDataset(testing_set)


def read_seediv(subject, dataset='SEEDIV', dim=4, K=1):
	return read_seed_pkls(subject, dataset, dim=dim, num_sub=15, K=K)


def read_seedv(subject, dataset='SEEDV', dim=4, K=1):
	return read_seed_pkls(subject, dataset, dim=dim, num_sub=16, K=K)


def read_seed_split(train_subs, val_subs, test_subs, dataset, dim=4, K=1):
	'''按受试者列表读取 train/val/test 三组（holdout 划分，非 LOSO）。
	train_subs/val_subs/test_subs：受试者 id 列表，如 [1,2,...,8]
	eeg: [S, 1, 62, 800*K] / dim=3 时为 [S, 62, 800*K]
	eye: [S, 1, eye_feats, K] / dim=3 时为 [S, eye_feats, K]
	返回 (SeedDataset, SeedDataset, SeedDataset)。'''
	root = get_seed_pkl_root(dataset, K)

	def load_subjects(subs):
		eeg_list, eye_list, label_list = [], [], []
		for i in subs:
			with open(f'{root}/{i}.pkl', 'rb') as f:
				data = pickle.load(f)
			eeg = torch.tensor(data['eeg'], dtype=torch.float32)
			eye = torch.tensor(data['eye'], dtype=torch.float32)
			labels = torch.tensor(data['labels'], dtype=torch.long)
			eeg_list.append(eeg)
			eye_list.append(eye)
			label_list.append(labels)
		out = {
			'eeg': torch.cat(eeg_list, dim=0),
			'eye': torch.cat(eye_list, dim=0),
			'labels': torch.cat(label_list, dim=0),
		}
		if dim == 4:
			out['eeg'] = out['eeg'].unsqueeze(1)
			out['eye'] = out['eye'].unsqueeze(1)
		return out

	train_set = load_subjects(train_subs)
	val_set = load_subjects(val_subs)
	test_set = load_subjects(test_subs)

	print(f'{dataset} Train {len(train_subs)} subs', 'EEG', train_set['eeg'].shape, 'Eye', train_set['eye'].shape, 'Labels', train_set['labels'].shape)
	print(f'{dataset} Val   {len(val_subs)} subs', 'EEG', val_set['eeg'].shape, 'Eye', val_set['eye'].shape, 'Labels', val_set['labels'].shape)
	print(f'{dataset} Test  {len(test_subs)} subs', 'EEG', test_set['eeg'].shape, 'Eye', test_set['eye'].shape, 'Labels', test_set['labels'].shape)

	return SeedDataset(train_set), SeedDataset(val_set), SeedDataset(test_set)


# =====================================================================
# 以下为从 EF 项目复制的旧数据加载函数（EF-MI / EF-MA / EF-WG，SEED-IV 不使用）
# =====================================================================

class MultimodalDataset(Dataset):
	def __init__(self, data):
		self.eeg = data['eeg']
		self.nirs = data['nirs']
		self.labels = data['labels']

		assert len(self.eeg) == len(self.nirs) == len(self.labels), 'Data lengths do not match'

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.eeg[index].float(), self.nirs[index].float(), self.labels[index].long()


def read_ef_train_si(subject, dataset, dim=4):
	'''
	Training Set:
	eeg.shape = [1680, 1, 30, 4000]
	nirs.shape = [1680, 1, 72, 200]
	labels [1680,]

	Testing Set:
	eeg.shape = [60, 1, 30, 4000]
	nirs.shape = [60, 1, 72, 200]
	labels [60,]
	'''
	num_sub = 29
	training_set = {}
	testing_set = {}
	training_eeg = []
	training_nirs = []
	training_labels = []

	for i in range(1, num_sub+1):
		if dataset == 'EFMI':
			data_root = '/data_main/junlin/BFM/Datasets/EF-MI/' + str(i) + '.pkl'
		elif dataset == 'EFMA':
			data_root = '/data_main/junlin/BFM/Datasets/EF-MA/' + str(i) + '.pkl'
		with open(data_root, 'rb') as f:
			data = pickle.load(f)
		eeg = data['eeg']
		nirs = data['nirs']
		labels = data['labels']
		if i == subject:
			if dim == 4:
				eeg = eeg.unsqueeze(1)
				nirs = nirs.unsqueeze(1)
			testing_set = {
				'eeg': eeg,
				'nirs': nirs,
				'labels': labels
			}
		else:
			training_eeg.append(eeg)
			training_nirs.append(nirs)
			training_labels.append(labels)
	
	training_set['eeg'] = torch.stack(training_eeg, dim=0)
	training_set['eeg'] = training_set['eeg'].reshape(-1, training_set['eeg'].shape[2], training_set['eeg'].shape[3])
	if dim == 4:
		training_set['eeg'] = training_set['eeg'].unsqueeze(1)

	training_set['nirs'] = torch.stack(training_nirs, dim=0)
	training_set['nirs'] = training_set['nirs'].reshape(-1, training_set['nirs'].shape[2], training_set['nirs'].shape[3])
	if dim == 4:
		training_set['nirs'] = training_set['nirs'].unsqueeze(1)
	
	training_set['labels'] = torch.stack(training_labels, dim=0)
	training_set['labels'] = training_set['labels'].reshape(-1)

	print(f'{subject} Training Set', 'EEG', training_set['eeg'].shape, 'fNIRS', training_set['nirs'].shape, 'Labels', training_set['labels'].shape)
	print(f'{subject} Testing Set', 'EEG', testing_set['eeg'].shape, 'fNIRS', testing_set['nirs'].shape, 'Labels', testing_set['labels'].shape)

	training_set = MultimodalDataset(training_set)
	testing_set = MultimodalDataset(testing_set)

	return training_set, testing_set


def read_wg_pkl_si(subject, dataset, dim=4):
	'''
	Training Set:
	eeg [1500, 1, 30, 2000]
	nirs.shape = [1500, 1, 72, 100]
	labels [1500,]

	Testing Set:
	eeg [60, 1, 30, 2000]
	nirs.shape = [60, 1, 72, 100]
	labels [60,]
	'''
	num_sub = 26
	training_set = {}
	testing_set = {}
	training_eeg = []
	training_nirs = []
	training_labels = []

	for i in range(1, num_sub+1):
		data_root = '/data_main/junlin/BFM/Datasets/EF-WG/WG/' + str(i) + '.pkl'
		with open(data_root, 'rb') as f:
			data = pickle.load(f)
		eeg = data['eeg']
		nirs = data['nirs']
		labels = data['labels']
		eeg = torch.tensor(eeg, dtype=torch.float32)
		nirs = torch.tensor(nirs, dtype=torch.float32)
		labels = torch.tensor(labels, dtype=torch.long)
		if i == subject:
			if dim == 4:
				eeg = eeg.unsqueeze(1)
				nirs = nirs.unsqueeze(1)
			testing_set = {
				'eeg': eeg,
				'nirs': nirs,
				'labels': labels
			}
		else:
			training_eeg.append(eeg)
			training_nirs.append(nirs)
			training_labels.append(labels)
	
	training_set['eeg'] = torch.stack(training_eeg, dim=0)
	training_set['eeg'] = training_set['eeg'].reshape(-1, training_set['eeg'].shape[2], training_set['eeg'].shape[3])
	if dim == 4:
		training_set['eeg'] = training_set['eeg'].unsqueeze(1)

	training_set['nirs'] = torch.stack(training_nirs, dim=0)
	training_set['nirs'] = training_set['nirs'].reshape(-1, training_set['nirs'].shape[2], training_set['nirs'].shape[3])
	if dim == 4:
		training_set['nirs'] = training_set['nirs'].unsqueeze(1)

	training_set['labels'] = torch.stack(training_labels, dim=0)
	training_set['labels'] = training_set['labels'].reshape(-1)

	print(f'{subject} Training Set', 'EEG', training_set['eeg'].shape, 'fNIRS', training_set['nirs'].shape, 'Labels', training_set['labels'].shape)
	print(f'{subject} Testing Set', 'EEG', testing_set['eeg'].shape, 'fNIRS', testing_set['nirs'].shape, 'Labels', testing_set['labels'].shape)

	training_set = MultimodalDataset(training_set)
	testing_set = MultimodalDataset(testing_set)
	
	return training_set, testing_set