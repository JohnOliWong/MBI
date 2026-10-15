from Dataloader import read_ef_train_si as si_data
from Dataloader import read_wg_pkl_si as wg_data

from Baselines.EEGNet import EEGNet
from Baselines.Conformer import Conformer
from Baselines.fNIRST import fNIRS_T
from Baselines.fNIRSNet import fNIRSNet
from Baselines.CAFNet import CAFNet
from Baselines.EFNet import EF_Net
from Baselines.VigilanceNet import VigilanceNet
from Baselines.TMMF import HybridTransformer
from EFBook.EF_Book_VQ import EF_Book as EF_VQ

import argparse
from Metrics import metrics, load_param
from Utils import setup_logger

import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import os
import re
from pathlib import Path
from datetime import datetime

LOG_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


class Trainer:
	def __init__(self, args):
		self.args = args
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_name = args.model
		self.dataset = args.dataset

		if self.model_name == 'EEGNet':
			if self.dataset != 'EFWG':
				self.model = EEGNet(C=30, T=4000, num_classes=args.num_class, 
									f1=8, depth=4, f2=4).to(self.device, dtype=torch.float32)
			else:
				self.model = EEGNet(C=30, T=2000, num_classes=args.num_class, 
									f1=8, depth=4, f2=4).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'Conformer':
			if self.dataset != 'EFWG':
				self.model = Conformer(emb_size=60, depth=4, 
						      		   n_classes=args.num_class).to(self.device, dtype=torch.float32)
			else:
				self.model = Conformer(emb_size=60, depth=6, 
						   			   n_classes=args.num_class).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'fNIRST':
			if self.dataset != 'EFWG':
				self.model = fNIRS_T(n_class=args.num_class, sampling_point=200, dim=64, depth=6, 
						 			 heads=8, mlp_dim=64).to(self.device, dtype=torch.float32)
			else:
				self.model = fNIRS_T(n_class=args.num_class, sampling_point=100, dim=64, depth=6, 
						 			 heads=8, mlp_dim=64).to(self.device, dtype=torch.float32)

		elif self.model_name == 'fNIRSNet':
			if self.dataset != 'EFWG':
				self.model = fNIRSNet(num_class=args.num_class, DHRConv_width=200, DWConv_height=72, 
						  			  num_DHRConv=4, num_DWConv=8).to(self.device, dtype=torch.float32)
			elif self.dataset == 'EFWG':
				self.model = fNIRSNet(num_class=args.num_class, DHRConv_width=100, DWConv_height=72, 
						  			  num_DHRConv=4, num_DWConv=8).to(self.device, dtype=torch.float32)

		elif self.model_name == 'CAFNet':
			if self.dataset != 'EFWG':
				self.model = CAFNet(eeg_dim=4000, nirs_dim=200, emb_dim=128, num_layers=8, 
									heads=1, dim_head=128, mlp_dim=64, 
									num_classes=2, dropout=0.25).to(self.device, dtype=torch.float32)
			elif self.dataset == 'EFWG':
				self.model = CAFNet(eeg_dim=2000, nirs_dim=100, emb_dim=128, num_layers=4, 
									heads=1, dim_head=128, mlp_dim=64, 
									num_classes=2, dropout=0.25).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'EFNet':
			self.model = EF_Net(num_classes=args.num_class, mode=args.dataset).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'VigilanceNet':
			if self.dataset != 'EFWG':
				self.model = VigilanceNet(hidden_size=64, num_heads=4, ffn_dim=128, 
										  attn_drop=0.25, proj_drop=0.25, feed_drop=0.25, 
										  num_classes=args.num_class, mode=self.dataset
										  ).to(self.device, dtype=torch.float32)
			elif self.dataset == 'EFWG':
				self.model = VigilanceNet(hidden_size=64, num_heads=4, ffn_dim=128, 
							  			  attn_drop=0.25, proj_drop=0.25, feed_drop=0.25, 
										  num_classes=args.num_class, mode=self.dataset
										  ).to(self.device, dtype=torch.float32)

		elif self.model_name == 'TSMMF':
			self.model = HybridTransformer(
				args.depth, args.query_size, args.key_size, args.value_size,
				args.emb_size, args.num_heads, args.expansion,
				args.conv_dropout, args.self_dropout, args.cross_dropout, args.cls_dropout,
				args.num_class, args.dataset, self.device,
			).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'EFBook':
			self.model = EF_VQ(args.dict_len, args.emb_size, args.threshold, 
					  		   args.num_class, self.dataset, self.device).to(self.device, dtype=torch.float32)

		self.criterion = nn.CrossEntropyLoss()
		weight_decay = 0.2 * args.learning_rate
		min_lr = 0.1 * args.learning_rate
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,
			mode='min',
			factor=0.8,
			patience=5,
			min_lr=min_lr,
		)

		self.logger = setup_logger('Logs/', log_filename=f'{self.model_name}_{self.dataset}_eval_{LOG_TIMESTAMP}.log')

		os.makedirs('Results', exist_ok=True)
	
	def z_score_mm(self, train_dataset, test_dataset, eps=1e-6):
		train_eeg = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
		train_eeg_mean = train_eeg.mean(dim=(0,1), keepdim=True)
		train_eeg_std = train_eeg.std(dim=(0,1),keepdim=True)
		train_eeg = (train_eeg - train_eeg_mean) / (train_eeg_std + eps)

		train_nirs = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
		train_nirs_mean = train_nirs.mean(dim=(0,1), keepdim=True)
		train_nirs_std = train_nirs.std(dim=(0,1),keepdim=True)
		train_nirs = (train_nirs - train_nirs_mean) / (train_nirs_std + eps)

		train_labels = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))])
		train_dataset = torch.utils.data.TensorDataset(train_eeg, train_nirs, train_labels)

		eval_eeg = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
		eval_eeg = (eval_eeg - train_eeg_mean) / (train_eeg_std + eps)
		eval_nirs = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
		eval_nirs = (eval_nirs - train_nirs_mean) / (train_nirs_std + eps)
		eval_labels = torch.stack([test_dataset[i][2] for i in range(len(test_dataset))])
		test_dataset = torch.utils.data.TensorDataset(eval_eeg, eval_nirs, eval_labels)

		return train_dataset, test_dataset
	
	def evaluate_epoch(self, results_root, subject, eval_loader):
		self.model.load_state_dict(torch.load(results_root + f'{subject}.pt'))
		self.model.eval()
		total_loss, total_correct = 0, 0
		all_preds, all_labels = [], []
		
		num_batch = len(eval_loader)
		with torch.no_grad():
			for i, (eval_eeg, eval_nirs, eval_labels) in enumerate(eval_loader):
				eval_eeg = eval_eeg.to(self.device, dtype=torch.float32)
				eval_nirs = eval_nirs.to(self.device, dtype=torch.float32)
				eval_labels = eval_labels.to(self.device)
				last_batch = (i == num_batch - 1)
				
				if self.model_name == 'EFBook':
					model_output = self.model(eval_eeg, eval_nirs, last_batch=last_batch)
					outputs = model_output['outputs']
					preds = torch.argmax(outputs, dim=1)
				elif self.model_name == 'CAFNet':
					preds, c_loss, kd_loss = self.model(eval_eeg, eval_nirs, eval_labels)
					preds = preds.max(1)[1]
				elif self.model_name in ['EFNet', 'VigilanceNet', 'TSMMF']:
					outputs = self.model(eval_eeg, eval_nirs)
					preds = torch.argmax(outputs, dim=1)
				elif self.model_name in ['EEGNet', 'Conformer']:
					outputs = self.model(eval_eeg)
					preds = torch.argmax(outputs, dim=1)
				elif self.model_name in ['fNIRST', 'fNIRSNet']:
					outputs = self.model(eval_nirs)
					preds = torch.argmax(outputs, dim=1)

				total_correct += (preds == eval_labels).sum().item()
				
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(eval_labels.cpu().numpy())
		
		acc = total_correct / len(eval_loader.dataset)
		precision = precision_score(all_labels, all_preds, zero_division=0)
		recall = recall_score(all_labels, all_preds, zero_division=0)
		f1 = f1_score(all_labels, all_preds, zero_division=0)
		kappa = cohen_kappa_score(all_labels, all_preds)
		
		return acc, precision, recall, f1, kappa
	
	def test_subject(self, results_root, subject, dataset):	
		dim = 4
		if self.model_name in ['CAFNet', 'VigilanceNet']:
			dim = 3

		if dataset in ('EFMI', 'EFMA'):
			train_dataset, eval_dataset = si_data(subject, dataset, dim=dim)
		elif dataset == 'EFWG':
			train_dataset, eval_dataset = wg_data(subject, dataset, dim=dim)
		
		if self.args.z_score:
			train_dataset, eval_dataset = self.z_score_mm(train_dataset, eval_dataset)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.args.batch_size, shuffle=False)
		eval_acc, precision, recall, f1, kappa = self.evaluate_epoch(results_root, subject, eval_loader)
		self.logger.info(f"Subject {subject} | Eval Acc: {eval_acc:.2f}")

		return eval_acc, precision, recall, f1, kappa

# Load parameters
checkpoint_name = ''
results_root = 'Results/' + checkpoint_name
config, seeds = load_param(results_root)
args = argparse.Namespace(**config)
results_path = Path(results_root)
logger = setup_logger('Logs/', log_filename=f'{args.model}_{args.dataset}_eval_{LOG_TIMESTAMP}.log')

subject_list = sorted(
	int(re.search(r'^(\d+)\.pt$', file.name).group(1))
	for file in results_path.glob('*.pt')
	if re.search(r'^(\d+)\.pt$', file.name)
)

acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []
for i, subject in enumerate(subject_list):
	seed = seeds[i]
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	logger.info(f"\n=== Subject {subject} ===")
	logger.info(f'Seed is {seed}')
	trainer = Trainer(args)
	eval_acc, precision, recall, f1, kappa = trainer.test_subject(results_root, subject, args.dataset)
	acc_list.append(eval_acc)
	precision_list.append(precision)
	recall_list.append(recall)
	f1_list.append(f1)
	kappa_list.append(kappa)

acc_list = [acc * 100 for acc in acc_list]
precision_list = [precision * 100 for precision in precision_list]
recall_list = [recall * 100 for recall in recall_list]
f1_list = [f1 * 100 for f1 in f1_list]
data = {
	'Acc': acc_list,
	'Precision': precision_list,
	'Recall': recall_list,
	'F1': f1_list,
	'Kappa': kappa_list,
	}
exp_name = results_root.split('/')[-2]
log_root = results_root + f'{exp_name}.xlsx'
metrics(log_root, data)