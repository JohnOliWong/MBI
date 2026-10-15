# This script is used to load data from BCI4_IIa dataset
# Modified by John Wong, 5th March 2025

import os
from scipy.io import loadmat as load
from scipy.signal import butter, lfilter
import numpy as np

def read_bci4_2a(mode):
	data_root = 'D:/HIT/MBI/Dataset/BCI4-IIa'
	eeg_list = []
	for subject_id in range(1, 10):
		subject_key = f'A{subject_id:02d}'
		if mode == 'train':
			file_name = f'{subject_key}T.mat'
		elif mode == 'eval':
			file_name = f'{subject_key}E.mat'
		file_root = os.path.join(data_root, file_name)
		data = load(file_root)
		subject_eeg = []
		subject_label = []
		
		all_labels = data['EVENTTYP']
		indices = np.where((all_labels == 769)|(all_labels == 770)|(all_labels == 771)|(all_labels == 772))[0]
		indices = indices[:250]
		labels = all_labels[indices]
		labels = labels.flatten().tolist()

		all_pos = data['EVENTPOS']
		pos = all_pos[indices]
		pos = pos.flatten().tolist()

		raw_eeg = data['s']
		raw_eeg = raw_eeg.T
		raw_eeg = raw_eeg.astype(float)

		# eeg signals were bandpass-filtered between 0.5Hz and 100Hz, with notch filter at 50Hz enabled
		# fs = train_data['SampleRate'][0].item() # convert fs from numpy array to integer
		# band = [1, 45]
		# f_low = band[0] / (fs / 2)
		# f_high = band[1] / (fs / 2)
		# b, a = butter(5, [f_low, f_high], 'bandpass')
		# eeg = lfilter(b, a, raw_eeg)

		slice_len = 500
		eeg = np.stack([raw_eeg[:22, position:position+slice_len] for position in pos], axis=2)
		eeg = eeg.transpose(2, 0, 1) # transpose shape to (trial, channel, sample_point)
		eeg_list.append(eeg)
	eeg_sum = np.concatenate(eeg_list, axis=0)
	return eeg_sum, labels

# Strict_TE is the MATLAB-converted version of the original BCI4_IIa dataset in .gdf format
def read_strict_te(mode):
	data_root = 'D:/HIT/MBI/Dataset/Strict_TE'
	eeg_list = []
	labels = []
	for subject_id in range(1, 10):
		subject_key = f'A{subject_id:02d}'
		if mode == 'train':
			file_name = f'{subject_key}T.mat'
		elif mode == 'eval':
			file_name = f'{subject_key}E.mat'
		file_root = os.path.join(data_root, file_name)
		data = load(file_root)
		subject_eeg = []
		subject_label = []

		s_list = list(range(4, 9))
		sub4_list = list(range(1, 6))
		struct_list = s_list
		if subject_id == 4:
			struct_list = sub4_list
		for i in struct_list:
			# each struct contains 48 trials
			trials = data['data'][0,i]['trial'][0,0].astype(int)
			n_trials = len(trials)
			X = data['data'][0,i]['X'][0,0][:,:22]
			Y = data['data'][0,i]['y'][0,0]

			for j in range(n_trials - 1):
				start_idx = int(trials[j])
				end_idx = int(trials[j + 1])
			
				trial_data = X[start_idx:end_idx, :]
				center_start = (trial_data.shape[0] - 1000) // 2
				center_end = center_start + 1000
				trial_center_data = trial_data[center_start:center_end, :]

				subject_eeg.append(trial_center_data)
				subject_label.append(Y[j])
			
			trial_data = X[int(trials[-1]):, :]  
			center_start = (trial_data.shape[0] - 1000) // 2
			center_end = center_start + 1000
			trial_center_data = trial_data[center_start:center_end, :]
			subject_eeg.append(trial_center_data)
			subject_label.append(Y[-1])

		subject_eeg = np.array(subject_eeg)
		subject_label = np.array(subject_label)
		
		print(np.shape(subject_eeg), np.shape(subject_label))
		subject_eeg = np.transpose(subject_eeg, (0, 2, 1))
		# subject_eeg.shape = (240, 22, 1000)
		subject_eeg = np.expand_dims(subject_eeg, axis=1)
		# subject_eeg.shape = (240, 1, 22, 1000)
		subject_label = np.transpose(subject_label)
		eeg_list.append(subject_eeg)
		labels.append(subject_label)
	return eeg_list, labels