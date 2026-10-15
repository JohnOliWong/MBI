import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
from Data_Aug import data_augmentation


class MultimodalDataset(Dataset):
	def __init__(self, data):
		self.eeg = data['eeg']
		self.nirs = data['nirs']
		self.labels = data['labels']

		assert len(self.eeg) == len(self.nirs) == len(self.labels), 'Data lengths do not match'

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.eeg[index], self.nirs[index], self.labels[index]

def read_ef_excel_sd(subject, mode):
	'''
	outputs:
	eeg [60, 30, 4000]
	nirs.shape = [60, 72, 200]
	labels [60,]
	'''
	# 1. set directory
	if not mode:
		eeg_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MI/{subject}'
		eeg_key = f'{subject}_EEG.xls'
		nirs_path = f'../../Dataset/EF-MI-MA/NIRS-EXCEL-MI/{subject}'
	else:
		eeg_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MA/{subject}'
		eeg_key = f'{subject}_EEG-MA.xls'
		nirs_path = f'../../Dataset/EF-MI-MA/NIRS-EXCEL-MA/{subject}'

	eeg_data_path = os.path.join(eeg_path, eeg_key)
	oxy_key = f'{subject}_oxy.xls'
	oxy_path = os.path.join(nirs_path, oxy_key)
	deoxy_key = f'{subject}_deoxy.xls'
	deoxy_path = os.path.join(nirs_path, deoxy_key)
	labels_key = f'{subject}_desc.xls'
	labels_path = os.path.join(eeg_path, labels_key)

	# 2. read excel files
	# raw data format: 60 sheets, for eeg, each sheet contains a matrix with shape (7000, 36)
	raw_eeg = pd.read_excel(eeg_data_path, header=None, sheet_name=None)
	raw_oxy = pd.read_excel(oxy_path, header=None, sheet_name=None)
	raw_deoxy = pd.read_excel(deoxy_path, header=None, sheet_name=None)
	desc = pd.read_excel(labels_path, header=None)

	# 3. concatenate all sheets into a single numpy array
	eeg = []
	HbO = []
	HbR = []
	for i in range(1, 61):
		sheet_name = f'Sheet{i}'
		eeg.append(raw_eeg[sheet_name].values)
		HbO.append(raw_oxy[sheet_name].values)
		HbR.append(raw_deoxy[sheet_name].values)
	eeg = np.array(eeg).transpose((0, 2, 1)) # eeg.shape = (trial, channel, time)
	HbO = np.array(HbO).transpose((0, 2, 1)) # HbO.shape = (60, 36, 350)
	HbR = np.array(HbR).transpose((0, 2, 1))
	desc = np.array(desc)

	# 4. extract trials belonging to different categories (note that the labels used for fnirs data are 1 and 2)
	eeg_L = []
	eeg_R = []
	HbO_L = []
	HbO_R = []
	HbR_L = []
	HbR_R = []
	eeg_start = 2000
	eeg_end = 6000
	nirs_start = 100
	nirs_end = 300
	for i in range(60):
		if desc[i, 0] == 16:
			eeg_L.append(eeg[i, :, eeg_start:eeg_end])
			HbO_L.append(HbO[i, :, nirs_start:nirs_end])
			HbR_L.append(HbR[i, :, nirs_start:nirs_end])
		elif desc[i, 0] == 32:
			eeg_R.append(eeg[i, :, eeg_start:eeg_end])
			HbO_R.append(HbO[i, :, nirs_start:nirs_end])
			HbR_R.append(HbR[i, :, nirs_start:nirs_end])
	
	# 5. concatenate trials one by one and generate labels (0 for left hand, 1 for right hand)
	eeg_L = np.array(eeg_L) # eeg_L.shape = (30, 30, 4000)
	eeg_R = np.array(eeg_R)
	HbO_L = np.array(HbO_L)
	HbO_R = np.array(HbO_R)
	HbR_L = np.array(HbR_L)
	HbR_R = np.array(HbR_R)
	HbO_L = np.concatenate([HbO_L, HbR_L], axis=1)
	HbO_R = np.concatenate([HbO_R, HbR_R], axis=1)

	eeg = []
	nirs = []
	labels = []
	for i in range(30):
		eeg.append(eeg_L[i, :, :])
		eeg.append(eeg_R[i, :, :])
		nirs.append(HbO_L[i, :, :])
		nirs.append(HbO_R[i, :, :])
		labels.append(0)
		labels.append(1)
	
	eeg = np.array(eeg)
	eeg = np.stack(eeg, axis=0)
	nirs = np.array(nirs)
	nirs = np.stack(nirs, axis=0)
	labels = np.array(labels)

	# convert data to torch.tensor type
	eeg = torch.tensor(eeg, dtype=torch.float64)
	nirs = torch.tensor(nirs, dtype=torch.float64)
	labels = torch.tensor(labels, dtype=torch.long)
	print(f'{subject}', 'EEG', eeg.shape, 'fNIRS', nirs.shape, 'Labels', labels.shape)

	return eeg, nirs, labels

def read_ef_excel_si(subject, mode):
	'''
	Training Set:
	eeg [1680, 30, 4000]
	nirs.shape = [1680, 72, 200]
	labels [1680,]

	Testing Set:
	eeg [60, 30, 4000]
	nirs.shape = [60, 72, 200]
	labels [60,]
	'''
	num_sub = 5
	training_set = {}
	testing_set = {}
	training_eeg = []
	training_nirs = []
	training_labels = []

	for i in range(num_sub):
		i += 1
		eeg, nirs, labels = read_ef_excel_sd(i, mode)
		if i == subject:
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
	training_set['nirs'] = torch.stack(training_nirs, dim=0)
	training_set['nirs'] = training_set['nirs'].reshape(-1, training_set['nirs'].shape[2], training_set['nirs'].shape[3])
	training_set['labels'] = torch.stack(training_labels, dim=0)
	training_set['labels'] = training_set['labels'].reshape(-1)

	print(f'{subject} Training Set', 'EEG', training_set['eeg'].shape, 'fNIRS', training_set['nirs'].shape, 'Labels', training_set['labels'].shape)
	print(f'{subject} Testing Set', 'EEG', testing_set['eeg'].shape, 'fNIRS', testing_set['nirs'].shape, 'Labels', testing_set['labels'].shape)

	return training_set, testing_set

def read_ef_pkl_si(subject, data_root):
	'''
	Training Set:
	eeg [1680, 30, 4000]
	nirs.shape = [1680, 72, 200]
	labels [1680,]

	Testing Set:
	eeg [60, 30, 4000]
	nirs.shape = [60, 72, 200]
	labels [60,]
	'''
	num_sub = 29
	training_set = {}
	testing_set = {}
	training_eeg = []
	training_nirs = []
	training_labels = []

	for i in range(num_sub):
		i += 1
		file_path = os.path.join(data_root, f'{i}.pkl')
		with open(file_path, 'rb') as f:
			data = pickle.load(f)
		if i == subject:
			testing_set['eeg'] = data_augmentation(data['eeg'])
			testing_set['eeg'] = testing_set['eeg'].unsqueeze(1)
			testing_set['nirs'] = data_augmentation(data['nirs'])
			testing_set['nirs'] = testing_set['nirs'].unsqueeze(1)
			testing_set['labels'] = data['labels']
		else:
			training_eeg.append(data['eeg'])
			training_nirs.append(data['nirs'])
			training_labels.append(data['labels'])
	
	training_set['eeg'] = torch.stack(training_eeg, dim=0)
	training_set['eeg'] = training_set['eeg'].reshape(-1, training_set['eeg'].shape[2], training_set['eeg'].shape[3])
	training_set['eeg'] = data_augmentation(training_set['eeg'])
	training_set['eeg'] = training_set['eeg'].unsqueeze(1)
	training_set['nirs'] = torch.stack(training_nirs, dim=0)
	training_set['nirs'] = training_set['nirs'].reshape(-1, training_set['nirs'].shape[2], training_set['nirs'].shape[3])
	training_set['nirs'] = data_augmentation(training_set['nirs'])
	training_set['nirs'] = training_set['nirs'].unsqueeze(1)
	training_set['labels'] = torch.stack(training_labels, dim=0)
	training_set['labels'] = training_set['labels'].reshape(-1)

	print(f'{subject} Training Set', 'EEG', training_set['eeg'].shape, 'fNIRS', training_set['nirs'].shape, 'Labels', training_set['labels'].shape)
	print(f'{subject} Testing Set', 'EEG', testing_set['eeg'].shape, 'fNIRS', testing_set['nirs'].shape, 'Labels', testing_set['labels'].shape)

	training_set = MultimodalDataset(training_set)
	testing_set = MultimodalDataset(testing_set)
	return training_set, testing_set