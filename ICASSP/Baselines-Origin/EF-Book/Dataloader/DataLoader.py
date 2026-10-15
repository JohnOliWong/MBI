# This script is used to load data from the open-access dataset contains EEG and fNIRS data from MI and MA tasks
# Data preprocess has been done with MATLAB R2024a
# Modified by John Wong, 5th March 2025

import os
import numpy as np
from scipy.io import loadmat as load
from scipy.signal import butter, iirnotch, lfilter

def read_mm_eeg(subject_id):
	eeg_root = os.path.join(r"D:\HIT\MBI\Dataset\EF-MI-MA\EEG", f"subject {subject_id:02d}", "cnt.mat")
	eeg_mrk_root = os.path.join(r"D:\HIT\MBI\Dataset\EF-MI-MA\EEG", f"subject {subject_id:02d}", "mrk.mat")

	raw_eeg = load(eeg_root)
	raw_eeg_mrk = load(eeg_mrk_root)

	eeg, labels = [], []
	for i in [0, 2, 4]:
		part_eeg = raw_eeg['cnt'][0, i]['x'][0, 0]
		fs = 200
		freq = 50
		low = 0.5
		high = 50
		half_sample = 500

		nyq = 0.5 * fs
		low = low / nyq
		high = high / nyq
		b, a = butter(5, [low, high], 'bandpass')
		part_eeg = lfilter(b, a, part_eeg, axis=0)
		w0 = freq / nyq
		b, a = iirnotch(w0, Q=30)
		part_eeg = lfilter(b, a, part_eeg, axis=0)
		part_eeg = part_eeg.astype(np.float64)

		part_mrk = raw_eeg_mrk['mrk'][0, i]['time'][0, 0].flatten()
		part_mrk = part_mrk // 5 # eeg data was downsampled to 200Hz from 1000Hz
		part_labels = raw_eeg_mrk['mrk'][0, i]['event'][0, 0][0]['desc'][0].flatten()
		n_trial = len(part_mrk)

		for i in range(n_trial-1):
			middle = (part_mrk[i] + part_mrk[i+1]) // 2
			# nirs_middle = (part_nirs_mrk[i] + part_nirs_mrk[i+1]) // 2
			trial_data = part_eeg[middle-half_sample:middle+half_sample, :30] # 30 channels
			# trial_nirs = part_nirs[nirs_middle-100:nirs_middle+100, :] # 36 channels
			eeg.append(trial_data)
			# nirs.append(trial_nirs)
		middle = (part_mrk[-1] + len(part_eeg)) // 2
		# nirs_middle = (part_nirs_mrk[-1] + len(part_nirs)) // 2
		trial_data = part_eeg[middle-half_sample:middle+half_sample, :30]
		# trial_nirs = part_nirs[nirs_middle-100:nirs_middle+100, :]
		eeg.append(trial_data)
		# nirs.append(trial_nirs)
		labels.append(part_labels)
		# nirs_labels.append(part_nirs_labels)
	eeg = np.stack(eeg, axis=0)
	# nirs = np.stack(nirs, axis=0)
	labels = np.concatenate(labels, axis=0)
	# nirs_labels = np.concatenate(nirs_labels, axis=0)
	labels[labels == 16] = 1
	labels[labels == 32] = 2 # 1 = left hand, 2 = right hand
	print('EEG', np.shape(eeg), 'Labels', np.shape(labels))
	eeg = np.transpose(eeg, (0, 2, 1))
	print('EEG.T', np.shape(eeg))
	# print(np.shape(nirs), np.shape(nirs_labels))
	# print(np.all(labels == nirs_labels))
	return eeg, labels

# forgot I can't preprocess fNIRS data in the same way I preprocess EEG 😅
def read_mm_nirs(subject_id):
	nirs_root = os.path.join(r"D:\HIT\MBI\Dataset\EF-MI-MA\fNIRS", f"subject {subject_id:02d}", "cnt.mat")
	nirs_mrk_root = os.path.join(r"D:\HIT\MBI\Dataset\EF-MI-MA\fNIRS", f"subject {subject_id:02d}", "mrk.mat")
	raw_nirs = load(nirs_root)
	raw_nirs_mrk = load(nirs_mrk_root)

	nirs = []
	fs = 10
	def bandpass_filter(data, order, low, high, fs):
		nyq = 0.5 * fs
		low = low / nyq
		high = high / nyq
		b, a = butter(order, [low, high], 'bandpass')
		return lfilter(b, a, data, axis=0)
	
	def zscore_norm(data):
		return (data - np.mean(data, axis=0)) / (np.std(data, axis=0))

	for i in [0, 2, 4]:
		part_nirs = raw_nirs['cnt'][0, i]['x'][0, 0]
		part_nirs_mrk = raw_nirs_mrk['mrk'][0, i]['time'][0, 0].flatten()
		part_nirs_mrk = (part_nirs_mrk / 100 * 0.8).astype(int)
		# part_nirs_mrk = (part_nirs_mrk * 0.08).astype(int)
		# part_nirs_labels = raw_nirs_mrk['mrk'][0, i]['event'][0, 0][0]['desc'][0].flatten()

		# The modified Beer-Lambert law has been applied in the MATLAB script
		# coef = np.array([[0.834, 3.225], [1.526, 1.131]])
		# coef_inv = np.linalg.inv(coef)
		# part_nirs = np.dot(coef_inv, part_nirs)

		HbO = part_nirs[:, :36]
		HbR = part_nirs[:, 36:]
		HbO = bandpass_filter(HbO, 5, 0.01, 0.1, fs)
		HbR = bandpass_filter(HbR, 5, 0.01, 0.1, fs)
		HbO = zscore_norm(HbO)
		HbR = zscore_norm(HbR)

		window = [-1, 25]
		window_samples = [int(fs * w) for w in window]
		HbO_list, HbR_list = [], []

		for mark in part_nirs_mrk:
			start = mark + window_samples[0]
			end = mark + window_samples[1]
			if 0 <= start < HbO.shape[0] and 0 <= end < HbO.shape[0]:
				# HbO_list.append(HbO[start:end, :])
				# HbR_list.append(HbR[start:end, :])
				HbO_segment = HbO[start:end, :]
				HbR_segment = HbR[start:end, :]
				combined = np.concatenate([HbO_segment, HbR_segment], axis=1)
				nirs.append(combined)
		
	nirs = np.stack(nirs, axis=0)
	nirs = np.transpose(nirs, (0, 2, 1))
	print('fNIRS.T', nirs.shape)

	return nirs