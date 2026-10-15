import os
import numpy as np
import torch
import pandas as pd


def read_excel_eeg(subject, mode):
	# 1. set directory
	if not mode:
		root_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MI/{subject}'
		eeg_key = f'{subject}_EEG.xls'
	else:
		root_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MA/{subject}'
		eeg_key = f'{subject}_EEG-MA.xls'
	eeg_path = os.path.join(root_path, eeg_key)
	labels_key = f'{subject}_desc.xls'
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
	labels = np.array(labels).astype(int)

	# # moving window
	# n_trial = eeg.shape[0]
	# n_sample = 5
	# n_channel = eeg.shape[1]
	# window = 800

	# eeg_data = np.zeros((n_trial * n_sample, n_channel, window))
	# labels_data = []

	# for trial in range(n_trial):
	# 	for sample in range(n_sample):
	# 		start = sample * window
	# 		end = start + window
	# 		eeg_data[trial * n_sample + sample] = eeg[trial, :, start:end]
	# 		labels_data.append(labels[trial])
	# eeg = eeg_data
	# labels = labels_data

	eeg = normalize(eeg)
	eeg = torch.tensor(eeg, dtype=torch.float64)
	labels = torch.tensor(labels, dtype=torch.long)
	print(f'{subject}', 'EEG', eeg.shape, 'Labels', labels.shape)

	return eeg, labels

def read_excel_nirs(subject, mode):
	if not mode:
		root_path = f'../../Dataset/EF-MI-MA/NIRS-EXCEL-MI/{subject}'
	else:
		root_path = f'../../Dataset/EF-MI-MA/NIRS-EXCEL-MA/{subject}'
	oxy_key = f'{subject}_oxy.xls'
	oxy_path = os.path.join(root_path, oxy_key)
	deoxy_key = f'{subject}_deoxy.xls'
	deoxy_path = os.path.join(root_path, deoxy_key)
	labels_key = f'{subject}_desc.xls'
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
	
	# # moving window
	# n_trial = nirs.shape[0]
	# n_sample = 5
	# n_channel = nirs.shape[1]
	# window = 40

	# nirs_data = np.zeros((n_trial * n_sample, n_channel, window))

	# for trial in range(n_trial):
	# 	for sample in range(n_sample):
	# 		start = sample * window
	# 		end = start + window
	# 		nirs_data[trial * n_sample + sample] = nirs[trial, :, start:end]
	# nirs = nirs_data

	nirs = normalize(nirs)
	nirs = torch.tensor(nirs, dtype=torch.float64)
	print(f'{subject}', 'fNIRS', nirs.shape)

	return nirs

def normalize(x):
	# find the mean and std along the channel dimension
    mean = np.mean(x, axis=(0, 2), keepdims=True)
    std = np.std(x, axis=(0, 2), keepdims=True)
    norm_x = (x - mean) / (std + 1e-8)
    return norm_x