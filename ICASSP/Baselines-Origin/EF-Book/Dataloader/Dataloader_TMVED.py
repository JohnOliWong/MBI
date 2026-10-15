# This script is used to load data from TMVED-3 dataset
# Modified by John Wong, 5th March 2025

import pickle
import torch

# read one-subject data(EEG ,fNIRS and label)
def read_single_data(subject_id):
	subject_root = f'D:/HIT/MBI/Dataset/TMVED-3/TMVED3/S{subject_id}.pkl'
	print(subject_root)

	with open(subject_root, 'rb') as f:
		# data: list
		# data contains 108 samples, [(EEG), (fNIRS), label]
		# samples.shape = [(60, 1000), (40, 40), (1,)]
		data = pickle.load(f)

	eeg = torch.tensor(np.array([i[0] for i in data]), dtype=torch.float32)
	nirs = torch.tensor(np.array([i[1] for i in data]), dtype=torch.float32)
	labels = torch.tensor([i[2] for i in data], dtype=torch.long)

	print('subject_id', eeg.shape, nirs.shape, labels.shape)
	return eeg, nirs, labels

# read multi-sujects data(EEG, fNIRS and labels)
def read_data(subject_num):
	samples = 108

	# eeg.shape = (3 * 108, 60, 1000), (3 * 108, 40, 40), (3 * 108, )
	eeg = torch.zeros((subject_num * samples, 60, 1000), dtype=torch.float32)
	nirs = torch.zeros((subject_num * samples, 40, 40), dtype=torch.float32)
	labels = torch.zeros((subject_num * samples,), dtype=torch.long)

	# list all subjects in the directory
	root_directory = 'D:/HIT/MBI/Dataset/TMVED-3/TMVED3'
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

	print(f'Cross-Subject data from {subject_num} subjects', eeg.shape, nirs.shape, labels.shape)

	return eeg, nirs, labels