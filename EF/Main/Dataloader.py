import torch
from torch.utils.data import Dataset
import pickle


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