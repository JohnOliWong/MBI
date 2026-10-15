import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

'''
input shape: [trial, channel, time]
'''
class DeapDataset(Dataset):
    def __init__(self, path_view1, path_view2, path_label):
        super(DeapDataset, self).__init__()
        self.path_view1 = path_view1
        self.path_view2 = path_view2
        self.path_label = path_label

        self.view1 = np.load(self.path_view1)
        self.view1 = normalize(self.view1)
        self.view1 = self.view1.reshape((720, 20, 160))
        self.view1 = torch.from_numpy(self.view1).float()

        self.view2 = np.load(self.path_view2)
        self.view2 = normalize(self.view2)
        self.view2 = self.view2.reshape((720, 20, 29))
        self.view2 = torch.from_numpy(self.view2).float()

        self.labels = np.load(self.path_label)
        self.labels[self.labels <= 5] = 0
        self.labels[self.labels > 5] = 1
        self.labels = torch.from_numpy(self.labels).int()

    def __getitem__(self, index):
        v1 = self.view1[index]
        v2 = self.view2[index]
        v = []
        v.append(v1)
        v.append(v2)
        labels = self.labels[index]
        return v, labels, index

    def __len__(self):
        return len(self.labels)

class MultimodalDataset(Dataset):
    def __init__(self, eeg, nirs, labels):
        super(MultimodalDataset, self).__init__()
        self.eeg = eeg
        self.nirs = nirs
        self.labels = labels

    def __getitem__(self, index):
        v1 = self.eeg[index]
        v2 = self.nirs[index]
        v = []
        v.append(v1)
        v.append(v2)
        labels = self.labels[index]
        return v, labels, index

    def __len__(self):
        return len(self.labels)

def normalize(x):
    scaler = StandardScaler()
    norm_x = scaler.fit_transform(x)
    return norm_x