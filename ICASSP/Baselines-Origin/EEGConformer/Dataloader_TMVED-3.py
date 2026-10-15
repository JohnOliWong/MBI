import pickle
import os
import numpy as np
import torch
from scipy.io import loadmat as load
from scipy.signal import butter, iirnotch, lfilter

def read_single_data(subject_id):
	# read one-subject data(EEG ,fNIRS and label)
	subject_root = f'../../Data/TMVED-3/S{subject_id}.pkl'
	print(subject_root)

	with open(subject_root, 'rb') as f:
		# data: list
		# data contains 108 samples, [(EEG), (fNIRS), label]
		# samples.shape = [(60, 1000), (40, 40), (1,)]
		data = pickle.load(f)

	eeg = np.array([i[0] for i in data])
	nirs = np.array([i[1] for i in data])
	labels = np.array([i[2] for i in data])
	labels = labels.reshape(labels.shape[0], 1)

	# initialize eeg, nirs and labels tensors using data from the first subject
	# eeg.shape = (108, 60, 1000), nirs.shape = (108, 40, 40), labels.shape = (108, )
	# trial, channel, sample points
	print(eeg.shape, nirs.shape, labels.shape)
	return eeg, nirs, labels

def read_data():
		# read multi-sujects data(EEG, fNIRS and labels)
	subject_num = 3
	samples = 108

	# eeg.shape = (3 * 108, 60, 1000), (3 * 108, 40, 40), (3 * 108, )
	eeg = torch.zeros((subject_num * samples, 60, 1000), dtype=torch.float32)
	nirs = torch.zeros((subject_num * samples, 40, 40), dtype=torch.float32)
	labels = torch.zeros((subject_num * samples,), dtype=torch.long)

	# list all subjects in the directory
	root_directory = '../../Data/TMVED-3/'
	subject_list = os.listdir(root_directory)
	# getcwd() returns the current working directory
	# subject_root = [os.path.join(os.getcwd(), 'TMVED3', i) for i in subject_list]
	subject_root = [os.path.join(root_directory, i) for i in subject_list]

	# stack data from individual subjects into the same tensor
	for index, sub in enumerate(subject_root):
		with open(sub, 'rb') as f:
			data = pickle.load(f)
		
		eeg[index * samples: (index + 1) * samples] = torch.tensor(np.array([i[0] for i in data]), dtype=torch.float32)
		nirs[index * samples: (index + 1) * samples] = torch.tensor(np.array([i[1] for i in data]), dtype=torch.float32)
		labels[index * samples: (index + 1) * samples] = torch.tensor(np.array([i[2] for i in data]), dtype=torch.long)

	print(eeg.shape, nirs.shape, labels.shape)

	return eeg, nirs, labels