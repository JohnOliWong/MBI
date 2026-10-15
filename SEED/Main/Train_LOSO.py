from Dataloader import read_seediv as seediv_data
from Dataloader import read_seedv as seedv_data

from Baselines.EEGNet import EEGNet
from Baselines.Conformer import Conformer
from Baselines.CAFNet import CAFNet
from Baselines.SPL_Loss import SPL_Loss
from Baselines.EFNet import EF_Net
from Baselines.VigilanceNet import VigilanceNet
from Baselines.TMMF import HybridTransformer
from Baselines.EyeFormer import EyeFormer
from EFBook.EF_Book_VQ import EF_Book as EF_VQ

from Args import get_args
from Metrics import metrics, save_param
from Utils import setup_logger, log_model_structure
from Visualization import Visualization as vis
from Utils import send_yagmail
from Results_LOSO import summarize_results

import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, f1_score, cohen_kappa_score, balanced_accuracy_score, roc_auc_score
import os
import time
from datetime import datetime

LOG_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
_STRUCTURE_LOGGED = False   # LOSO 每个受试者的模型结构相同，只在第一个受试者打印一次


class Trainer:
	def __init__(self, args, exp_name):
		self.args = args
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_name = args.model
		self.dataset = args.dataset
		num_class = 4 if args.dataset == 'SEEDIV' else 5   # SEED-IV 4 类 / SEED-V 5 类
		self.logger = setup_logger('Logs/', log_filename=f'{self.model_name}_{self.dataset}_train_{LOG_TIMESTAMP}.log')

		if self.model_name == 'EEGNet':
			self.model = EEGNet(C=62, T=args.seg_len * args.K, num_classes=num_class, 
								f1=8, depth=4, f2=4).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'Conformer':
			self.model = Conformer(emb_size=60, depth=4, channels=62,
					       		   n_classes=num_class).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'CAFNet':
			self.model = CAFNet(eeg_dim=args.seg_len * args.K, eye_dim=args.K, emb_dim=64, num_layers=4, 
								heads=1, dim_head=64, mlp_dim=128, 
								num_classes=num_class, dropout=0.25).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'EFNet':
			self.model = EF_Net(num_classes=num_class, mode=args.dataset).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'VigilanceNet':
			self.model = VigilanceNet(hidden_size=64, num_heads=4, ffn_dim=128, 
									  attn_drop=0.25, proj_drop=0.25, feed_drop=0.25, 
									  num_classes=num_class, mode=args.dataset
									  ).to(self.device, dtype=torch.float32)

		elif self.model_name == 'TSMMF':
			self.model = HybridTransformer(
				args.depth, args.query_size, args.key_size, args.value_size,
				args.emb_size, args.num_heads, args.expansion,
				args.conv_dropout, args.self_dropout, args.cross_dropout, args.cls_dropout,
				num_class, args.dataset, self.device,
			).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'EFBook':
			self.model = EF_VQ(args.dict_len, args.emb_size, args.threshold, 
					  		   num_class, args.dataset, self.device).to(self.device, dtype=torch.float32)
		elif self.model_name == 'EyeFormer':
			eye_dim = 31 if args.dataset == 'SEEDIV' else 33
			self.model = EyeFormer(num_classes=num_class, eye_dim=eye_dim, seq_len=args.K,
								   hidden_size=args.emb_size, num_layers=args.depth,
								   num_heads=args.num_heads, mlp_ratio=args.expansion,
								   dropout=0.1, causal=args.causal).to(self.device, dtype=torch.float32)

		if self.model_name == 'CAFNet':
			self.criterion = SPL_Loss(batch_size=args.batch_size, alpha=args.alpha, beta=args.beta, 
							 		  spl_lambda=args.spl_lambda, spl_gamma=args.spl_gamma)
		else:
			self.criterion = nn.CrossEntropyLoss()
		
		weight_decay = 0.2 * args.learning_rate
		min_lr = 0.1 * args.learning_rate
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)

		# patience设置要满足学习率的衰减过程可以持续到训练结束
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,
			mode='min',
			factor=0.8,
			patience=5,
			min_lr=min_lr,
		)

		self.results_root = 'Results/' + exp_name
		os.makedirs(self.results_root, exist_ok=True)
	
	def z_score_mm(self, train_dataset, test_dataset, eps=1e-6):
		# 沿 batch 维计算每个 (C, T) 位置的均值/方差，并在原张量上就地标准化，避免额外拷贝
		train_eeg_mean = train_dataset.eeg.mean(dim=(0, 1), keepdim=True)
		train_eeg_std = train_dataset.eeg.std(dim=(0, 1), keepdim=True)
		train_dataset.eeg.sub_(train_eeg_mean).div_(train_eeg_std + eps)

		train_eye_mean = train_dataset.eye.mean(dim=(0, 1), keepdim=True)
		train_eye_std = train_dataset.eye.std(dim=(0, 1), keepdim=True)
		train_dataset.eye.sub_(train_eye_mean).div_(train_eye_std + eps)

		test_dataset.eeg.sub_(train_eeg_mean).div_(train_eeg_std + eps)
		test_dataset.eye.sub_(train_eye_mean).div_(train_eye_std + eps)

		train_dataset = torch.utils.data.TensorDataset(train_dataset.eeg, train_dataset.eye, train_dataset.labels)
		test_dataset = torch.utils.data.TensorDataset(test_dataset.eeg, test_dataset.eye, test_dataset.labels)

		return train_dataset, test_dataset
	
	def train_subject(self, subject, dataset):
		dim = 4
		if self.model_name in ['CAFNet', 'VigilanceNet', 'EyeFormer']:
			dim = 3

		if dataset == 'SEEDIV':
			train_dataset, eval_dataset = seediv_data(subject, dataset, dim=dim, K=self.args.K)
		elif dataset == 'SEEDV':
			train_dataset, eval_dataset = seedv_data(subject, dataset, dim=dim, K=self.args.K)
		
		if self.args.z_score:
			train_dataset, eval_dataset = self.z_score_mm(train_dataset, eval_dataset)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.args.batch_size, shuffle=False)

		# CAF-Net uses cross-entropy loss for evaluation
		cafnet_eval_criterion = nn.CrossEntropyLoss()
	
		# kaiming init
		init = True
		def init_kaiming(m):
			if type(m) == nn.Linear:
				nn.init.kaiming_normal_(m.weight.data)
		if init:
			self.model.apply(init_kaiming)

		# 26.10.06: 在数据集维度打印之后、训练结果输出之前打印模型结构
		# LOSO 每个受试者构建的模型结构相同，只打印一次
		global _STRUCTURE_LOGGED
		if not _STRUCTURE_LOGGED:
			log_model_structure(self.logger, self.model, title=f'{self.model_name} Model Structure')
			_STRUCTURE_LOGGED = True

		best_acc = 0
		# 26.10.06: 早停对所有模型生效（LOSO 无验证集，用留一测试受试者的 eval loss 监控）
		early_stop_patience = self.args.early_stop_patience
		best_eval_loss = float('inf')
		patience_counter = 0
		train_loss_list, train_acc_list = [], []
		loss_list, acc_list, precision_list, bacc_list, wf1_list, kappa_list, auc_list = [], [], [], [], [], [], [], []
		wu_lr = self.args.learning_rate / 10
		base_lr = self.args.learning_rate
		current_lr, warm_up = base_lr, self.args.warm_up
		
		if self.model_name in ['CAFNet', 'VigilanceNet']:
			num_epochs = self.args.num_epochs
		elif self.model_name in ['TSMMF', 'EFBook']:
			num_epochs = self.args.num_epochs
		else:
			num_epochs = self.args.num_epochs
		
		for epoch in range(num_epochs):
			#----------Training----------
			self.model.train()
			total_correct, total_loss = 0, 0

			# warm-up epochs
			if epoch < warm_up:
				current_lr = wu_lr
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = current_lr
			else:
				# after warm-up, let scheduler adjust the learning rate
				if epoch == warm_up:
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = base_lr
				current_lr = self.optimizer.param_groups[0]['lr']

			num_batch = len(train_loader)
			for i, (batch_eeg, batch_eye, batch_labels) in enumerate(train_loader):
				batch_eeg = batch_eeg.to(self.device, dtype=torch.float32)
				batch_eye = batch_eye.to(self.device, dtype=torch.float32)
				batch_labels = batch_labels.to(self.device)
				last_batch = (i == num_batch - 1)
				
				if self.model_name == 'EFBook':
					model_output = self.model(batch_eeg, batch_eye, last_batch=last_batch)
					outputs = model_output['outputs']
					preds = torch.argmax(outputs, dim=1)
					quan_loss = model_output['loss']
					cls_loss = self.criterion(outputs, batch_labels.long())
					loss = cls_loss + quan_loss
				elif self.model_name == 'CAFNet':
					preds, c_loss, kd_loss = self.model(batch_eeg, batch_eye, batch_labels)
					loss = self.criterion(preds, batch_labels, i, c_loss, kd_loss)
					preds = preds.max(1)[1]
				elif self.model_name in ['EFNet', 'VigilanceNet', 'TSMMF', 'EyeFormer']:
					outputs = self.model(batch_eeg, batch_eye)
					preds = torch.argmax(outputs, dim=1)
					loss = self.criterion(outputs, batch_labels.long())
				elif self.model_name in ['EEGNet', 'Conformer']:
					outputs = self.model(batch_eeg)
					preds = torch.argmax(outputs, dim=1)
					loss = self.criterion(outputs, batch_labels.long())
				
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
				total_correct += (preds == batch_labels).sum().item()
				total_loss += loss.item()

			train_loss = total_loss / len(train_loader)
			train_acc = total_correct / len(train_loader.dataset)
			train_loss_list.append(train_loss)
			train_acc_list.append(train_acc)

			#----------Evaluation----------
			self.model.eval()
			total_loss, total_correct = 0, 0
			all_preds, all_labels, all_probs = [], [], []
			
			with torch.no_grad():
				for eval_eeg, eval_eye, eval_labels in eval_loader:
					eval_eeg = eval_eeg.to(self.device, dtype=torch.float32)
					eval_eye = eval_eye.to(self.device, dtype=torch.float32)
					eval_labels = eval_labels.to(self.device)
					
					if self.model_name == 'EFBook':
						model_output = self.model(eval_eeg, eval_eye, last_batch=False)
						logits = model_output['outputs']
						preds = torch.argmax(logits, dim=1)
						quan_loss = model_output['loss']
						cls_loss = self.criterion(logits, eval_labels.long())
						loss = cls_loss + quan_loss
					elif self.model_name == 'CAFNet':
						logits, c_loss, kd_loss = self.model(eval_eeg, eval_eye, eval_labels)
						loss = cafnet_eval_criterion(logits, eval_labels) + args.alpha * c_loss + args.beta * kd_loss
						preds = logits.max(1)[1]
					elif self.model_name in ['EFNet', 'VigilanceNet', 'TSMMF', 'EyeFormer']:
						logits = self.model(eval_eeg, eval_eye)
						preds = torch.argmax(logits, dim=1)
						loss = self.criterion(logits, eval_labels.long())
					elif self.model_name in ['EEGNet', 'Conformer']:
						logits = self.model(eval_eeg)
						preds = torch.argmax(logits, dim=1)
						loss = self.criterion(logits, eval_labels.long())
					
					total_loss += loss.item()
					total_correct += (preds == eval_labels).sum().item()
					
					all_preds.extend(preds.cpu().numpy())
					all_labels.extend(eval_labels.cpu().numpy())
					all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
			
			eval_loss = total_loss / len(eval_loader)
			self.scheduler.step(eval_loss)
			eval_acc = total_correct / len(eval_loader.dataset)
			if eval_acc > best_acc:
				best_acc = eval_acc
				torch.save(self.model.state_dict(), self.results_root + f'{subject}.pt')

			# SPL: 每个 epoch 后递增阈值，让更多样本逐步纳入训练
			if self.model_name == 'CAFNet':
				self.criterion.increase_threshold()
			precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
			bacc = balanced_accuracy_score(all_labels, all_preds)
			wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
			kappa = cohen_kappa_score(all_labels, all_preds)
			try:
				auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
			except ValueError:
				auc = 0.0

			loss_list.append(eval_loss)
			acc_list.append(eval_acc)
			precision_list.append(precision)
			bacc_list.append(bacc)
			wf1_list.append(wf1)
			kappa_list.append(kappa)
			auc_list.append(auc)
			
			self.logger.info(f"Subject {subject} | Epoch {epoch+1}/{num_epochs} | LR {self.optimizer.param_groups[0]['lr']}")
			self.logger.info(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc*100:.2f} | Eval Loss: {eval_loss:.2f} | Eval Acc: {eval_acc*100:.2f}")

			# 26.10.06: 早停（监控 eval loss，即留一测试受试者的 loss）
			# 26.10.07: 判定移到本 epoch 指标记录之后，避免 train/eval 列表长度不一致
			if eval_loss < best_eval_loss:
				best_eval_loss = eval_loss
				patience_counter = 0
			else:
				patience_counter += 1
			if patience_counter >= early_stop_patience:
				self.logger.info(f'Subject {subject} early stopping at epoch {epoch+1}: eval loss not improved for {early_stop_patience} epochs')
				break

		train_acc_list = [acc * 100 for acc in train_acc_list]
		acc_list = [acc * 100 for acc in acc_list]
		precision_list = [precision * 100 for precision in precision_list]
		bacc_list = [b * 100 for b in bacc_list]
		wf1_list = [f * 100 for f in wf1_list]

		data = {
			'train loss': train_loss_list,
			'train acc': train_acc_list,
			'val loss': loss_list,
			'acc': acc_list,
			'precision': precision_list,
			'B-ACC': bacc_list,
			'W-F1': wf1_list,
			'kappa': kappa_list,
			'AUC': auc_list,
		}
		log_root = self.results_root + f'{str(subject)}.xlsx'
		metrics(log_root, data)
		return all_labels, all_preds

args = get_args()

dataset = args.dataset
logger = setup_logger('Logs/', log_filename=f'{args.model}_{dataset}_train_{LOG_TIMESTAMP}.log')
# use either a predefined string or the timestamp as the folder name to store the results
# exp_name = args.exp_name
start_time = time.time()
curr_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.localtime(start_time))
exp_name = f'{args.model}_{dataset}_{curr_time}/'

results_root = 'Results/' + exp_name

# dataset: SEEDIV, SEEDV
num_subject = 15 if dataset == 'SEEDIV' else 16
seeds, total_labels, total_preds = [], [], []
for subject in range(1, num_subject+1):
	seed = 42
	seeds.append(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	start_time = time.time()
	start_time_string = time.strftime('%b %d %Y %H:%M:%S', time.localtime(start_time))
	logger.info(f"=== Subject {subject} ===")
	logger.info(f'Started {start_time_string}')
	logger.info(f'Seed is {seed}')
	trainer = Trainer(args, exp_name)
	all_labels, all_preds = trainer.train_subject(subject, dataset)
	total_labels.append(all_labels)
	total_preds.append(all_preds)
	end_time = time.time()
	end_time_string = time.strftime('%b %d %Y %H:%M:%S', time.localtime(end_time))
	logger.info(f'Ended {end_time_string} Duration {end_time-start_time} s')

# export parameters
# np.array会将所有受试者的标签转换为(16, N)的矩阵
# 由于subject 7少了一个session，只有215个标签，会报错
params = args.__dict__
save_param(results_root, params, seeds)
total_labels = np.concatenate(total_labels)
total_preds = np.concatenate(total_preds)

# visualization
cm_visual = vis(subject, dataset, results_root)
cm_visual.plot_cm_all(total_labels, total_preds)

# send email upon the completion of training
summary, _ = summarize_results(checkpoint_names=[exp_name.rstrip('/')])
send_yagmail(exp_name.split('/')[-2], results_text=summary)