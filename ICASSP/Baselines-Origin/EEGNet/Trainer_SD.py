from EEGNet import EEGNet as net
from Metrics import metrics

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import os
import pickle

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class Trainer:
	def __init__(self, config):
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.model = net().to(self.device, dtype=torch.float32)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

		os.makedirs('Results', exist_ok=True)
	
	def z_score(self, signal, eps=1e-6):
		mean = signal.mean(dim=(0,2), keepdim=True)
		std = signal.std(dim=(0,2), keepdim=True)
		std = torch.max(std, torch.tensor(eps, device=signal.device))
		signal = (signal - mean) / std
		return signal	
	
	def train_epoch(self, epoch, train_loader):
		epoch += 1
		self.model.train()
		total_correct, total_loss = 0, 0

		for batch_nirs, batch_labels in train_loader:
			batch_nirs = batch_nirs.to(self.device, dtype=torch.float32)
			batch_labels = batch_labels.to(self.device)
			
			outputs = self.model(batch_nirs)
			loss = self.criterion(outputs, batch_labels.long())

			self. optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			preds = torch.argmax(outputs, dim=1)
			total_correct += (preds == batch_labels).sum().item()
			total_loss += loss.item()

		train_loss = total_loss / len(train_loader)
		train_acc = total_correct / len(train_loader.dataset)
		
		return train_loss, train_acc
	
	def evaluate_epoch(self, eval_loader):
		self.model.eval()
		total_loss, total_correct = 0, 0
		all_preds, all_labels = [], []
		
		with torch.no_grad():
			for eval_nirs, eval_labels in eval_loader:
				eval_nirs = eval_nirs.to(self.device, dtype=torch.float32)
				eval_labels = eval_labels.to(self.device)
				
				outputs = self.model(eval_nirs)
				loss = self.criterion(outputs, eval_labels.long())
				
				preds = torch.argmax(outputs, dim=1)
				total_correct += (preds == eval_labels).sum().item()
				total_loss += loss.item()
				
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(eval_labels.cpu().numpy())
		
		loss = total_loss / len(eval_loader)
		acc = total_correct / len(eval_loader.dataset)
		precision = precision_score(all_labels, all_preds, zero_division=0)
		recall = recall_score(all_labels, all_preds)
		f1 = f1_score(all_labels, all_preds)
		kappa = cohen_kappa_score(all_labels, all_preds)
		
		return loss, acc, precision, recall, f1, kappa	

	def train_subject(self, subject, mode):
		if mode == 0:
			data_root = config['mi_root'] + str(subject) + '.pkl'
		elif mode == 1:
			data_root = config['ma_root'] + str(subject) + '.pkl'
		elif mode == 2:
			data_root = config['wg_root'] + str(subject) + '.pkl'
		with open(data_root, 'rb') as f:
			data = pickle.load(f)
		
		eeg = data['eeg']
		labels = data['labels']

		eeg = torch.tensor(eeg, dtype=torch.float32)
		labels = torch.tensor(labels)

		eeg = self.z_score(eeg)
		
		eeg = eeg.permute(0, 2, 1)
		eeg = eeg.unsqueeze(1)
		eeg, labels = eeg.to(self.device), labels.to(self.device)
		
		train_size = int(config['ratio'] * len(labels)) # training/testing ratio
		eval_size = len(labels) - train_size
		
		dataset = torch.utils.data.TensorDataset(eeg, labels)
		train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.config['batch_size'], shuffle=False)
		
		acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []
		
		for epoch in range(self.config['num_epochs']):
			train_loss, train_acc = self.train_epoch(epoch, train_loader)
			eval_loss, eval_acc, precision, recall, f1, kappa = self.evaluate_epoch(eval_loader)
			
			acc_list.append(eval_acc)
			precision_list.append(precision)
			recall_list.append(recall)
			f1_list.append(f1)
			kappa_list.append(kappa)
			
			print(f"Subject {subject} | Epoch {epoch+1}/{self.config['num_epochs']} | "
				f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f} | "
				f"Eval Loss: {eval_loss:.2f} | Eval Acc: {eval_acc:.2f}")
			
		metrics(subject, config['log_name'], config['log_mode'], acc_list, precision_list, recall_list, f1_list, kappa_list)
		return

# Hyperparameters
# the correlation between Q, K, V and embedding size
config = {
	'depth': 4,
	'query_size': 64,
	'key_size': 64,
	'value_size': 64,
	'emb_size': 64,
	'dict_len': 512,
	'decay': 0.99,
	'num_heads': 4,
	'expansion': 2,
	'conv_dropout': 0.3,
	'self_dropout': 0.3,
	'cross_dropout': 0.3,
	'cls_dropout': 0.5,
	'num_classes': 2,
	'batch_size': 16,
	'num_epochs': 200,
	'learning_rate': 1e-3,
	'ratio': 0.6,
	'mi_root': '../../Dataset/EF-MI-MA/EF-PKL-MI/',
	'ma_root': '../../Dataset/EF-MI-MA/EF-PKL-MA/',
	'wg_root': '../../Dataset/EF-WG/WG/',
	'log_name': 51,
	'log_mode': 1,
}

# Initialize and run trainer
trainer = Trainer(config)
for subject in range(29):
	subject += 1
	print(f"\n=== Subject {subject} ===")
	results = trainer.train_subject(subject, mode=0) # 0 = MI, 1 = MA, 2 = WG