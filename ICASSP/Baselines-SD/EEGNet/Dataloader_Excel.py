import os
import numpy as np
import torch
import pandas as pd
import pickle


def read_excel_eeg(subject_id, mode):
	# 1. set directory
	if not mode:
		root_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MI/{subject_id}'
		eeg_key = f'{subject_id}_EEG.xls'
	else:
		root_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MA/{subject_id}'
		eeg_key = f'{subject_id}_EEG-MA.xls'
	eeg_path = os.path.join(root_path, eeg_key)
	labels_key = f'{subject_id}_desc.xls'
	labels_path = os.path.join(root_path, labels_key)

	# 2. read excel files
	# raw data format: 60 sheets, each containing a matrix with shape (7000, 36)
	raw_eeg = pd.read_excel(eeg_path, header=None, sheet_name=None)
	desc = pd.read_excel(labels_path, header=None)

	# 3. concatenate all sheets into a single tensor
	eeg = []
	for i in range(1, 61):
		sheet_name = f'Sheet{i}'
		eeg.append(raw_eeg[sheet_name].values)
	eeg = np.array(eeg).transpose((0, 2, 1)) # eeg.shape = (trial, channel, sample_points)
	desc = np.array(desc)

	# 4. extract trials belonging to different categories
	eeg_L = []
	eeg_R = []
	start = 2000
	end = 6000
	for i in range(60):
		if desc[i, 0] == 16:
			eeg_L.append(eeg[i, :, start:end])
		elif desc[i, 0] == 32:
			eeg_R.append(eeg[i, :, start:end])
	
	# 5. concatenate trials one by one and generate labels (1 for left hand, 2 for right hand)
	eeg_L = np.array(eeg_L) # eeg_L.shape = (30, 30, 4000)
	eeg_R = np.array(eeg_R)
	eeg = []
	labels = []
	for i in range(30):
		eeg.append(eeg_L[i, :, :])
		eeg.append(eeg_R[i, :, :])
		labels.append(0)
		labels.append(1)
	eeg = np.array(eeg)
	eeg = np.stack(eeg, axis=0)
	labels = np.array(labels)
	eeg = torch.tensor(eeg, dtype=torch.float64)
	labels = torch.tensor(labels, dtype=torch.long)

	data = {
		'eeg': eeg,
		'labels': labels,
	}
	file_path = str(subject_id) + '.pkl'
	with open(file_path, 'wb') as f:
		pickle.dump(data, f)
	print(f'{subject_id}', 'EEG', eeg.shape, 'Labels', labels.shape)

	return eeg, labels

def read_excel_nirs(subject_id, mode):
	if not mode:
		root_path = f'../../Data/EF-MI-MA/Pre-MI/{subject_id}'
	else:
		root_path = f'../../Data/EF-MI-MA/Pre-MA/{subject_id}'
	oxy_key = f'{subject_id}_oxy.xls'
	oxy_path = os.path.join(root_path, oxy_key)
	deoxy_key = f'{subject_id}_deoxy.xls'
	deoxy_path = os.path.join(root_path, deoxy_key)
	labels_key = f'{subject_id}_desc.xls'
	labels_path = os.path.join(root_path, labels_key)

	raw_oxy = pd.read_excel(oxy_path, header=None, sheet_name=None)
	raw_deoxy = pd.read_excel(deoxy_path, header=None, sheet_name=None)
	desc = pd.read_excel(labels_path, header=None)

	HbO = []
	HbR = []
	for i in range(1, 61):
		sheet_name = f'Sheet{i}'
		HbO.append(raw_oxy[sheet_name].values)
		HbR.append(raw_deoxy[sheet_name].values)
	
	HbO = np.array(HbO).transpose((0, 2, 1)) # HbO.shape = (60, 36, 350)
	HbR = np.array(HbR).transpose((0, 2, 1))
	desc = np.array(desc)

	HbO_L = []
	HbO_R = []
	HbR_L = []
	HbR_R = []
	start = 100
	end = 300
	for i in range(60):
		if desc[i, 0] == 1:
			HbO_L.append(HbO[i, :, start:end])
			HbR_L.append(HbR[i, :, start:end])
		elif desc[i, 0] == 2:
			HbO_R.append(HbO[i, :, start:end])
			HbR_R.append(HbR[i, :, start:end])
	
	HbO_L = np.array(HbO_L)
	HbO_R = np.array(HbO_R)

	nirs = []
	labels = []
	for i in range(30):
		nirs.append(HbO_L[i, :, :])
		nirs.append(HbO_R[i, :, :])
		labels.append(0)
		labels.append(1)
	
	nirs = np.array(nirs)
	nirs = np.stack(nirs, axis=0)
	labels = np.array(labels)
	nirs = torch.tensor(nirs, dtype=torch.float64)
	labels = torch.tensor(labels, dtype=torch.long)
	print(f'{subject_id}', 'fNIRS', nirs.shape, 'Labels', labels.shape)

	return nirs, labels