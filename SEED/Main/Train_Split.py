from Dataloader import read_seed_split

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
from Results_Split import summarize_results

import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, f1_score, cohen_kappa_score, balanced_accuracy_score, roc_auc_score
import os
import time
from datetime import datetime

# 26.10.06：1) 码本尺寸 dict_len 默认 512 -> 128（Args.py），缓解 SEED-V 过拟合与码本坍缩；
#           2) 早停扩展为对所有模型生效（监控 val loss，patience=args.early_stop_patience）
# 26.10.01做了以下改动（仅针对 EF-Book 在 SEED 上的训练，其他模型/目录不受影响）：
#   1. 码本尺寸 dict_len：1800 -> 512（Args.py --dict_len 默认值）
#   2. codebook_loss 权重：0.1 -> 0.05（EFBook/EF_Book_VQ.py forward 返回的 loss）
#   3. 分类头加 Dropout(0.2)：projector 中间层加 Dropout（EFBook/EF_Book_VQ.py Classifier，0.5 会导致训练不稳定已调低）
#   4. 加入早停：监控 val loss，patience=15（Train_Split.py run()）
#   注：曾尝试给码本加 L2 权重衰减，但码本无梯度、L2 平方惩罚项导致 loss 爆炸，已回滚。

LOG_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


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
			self.model = EF_Net(num_classes=num_class, mode=args.dataset, eeg_time=args.seg_len * args.K).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'VigilanceNet':
			self.model = VigilanceNet(hidden_size=64, num_heads=4, ffn_dim=128, 
									  attn_drop=0.25, proj_drop=0.25, feed_drop=0.25, 
									  num_classes=num_class, mode=args.dataset, eeg_time=args.seg_len * args.K, eye_time=args.K
									  ).to(self.device, dtype=torch.float32)

		elif self.model_name == 'TSMMF':
			self.model = HybridTransformer(
				args.depth, args.query_size, args.key_size, args.value_size,
				args.emb_size, args.num_heads, args.expansion,
				args.conv_dropout, args.self_dropout, args.cross_dropout, args.cls_dropout,
				num_class, args.dataset, self.device, eeg_time=args.seg_len * args.K, eye_time=args.K,
			).to(self.device, dtype=torch.float32)
		
		elif self.model_name == 'EFBook':
			self.model = EF_VQ(args.dict_len, args.emb_size, args.threshold, 
					  		   num_class, args.dataset, self.device, eeg_time=args.seg_len * args.K, eye_time=args.K).to(self.device, dtype=torch.float32)
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
		self.cafnet_eval_criterion = nn.CrossEntropyLoss()
		
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
	
	def z_score_mm(self, train_dataset, val_dataset, test_dataset, eps=1e-6):
		# 沿 batch 维计算每个 (C, T) 位置的均值/方差，train 统计量应用到 train/val/test
		train_eeg_mean = train_dataset.eeg.mean(dim=(0, 1), keepdim=True)
		train_eeg_std = train_dataset.eeg.std(dim=(0, 1), keepdim=True)
		train_eye_mean = train_dataset.eye.mean(dim=(0, 1), keepdim=True)
		train_eye_std = train_dataset.eye.std(dim=(0, 1), keepdim=True)

		for ds in (train_dataset, val_dataset, test_dataset):
			ds.eeg.sub_(train_eeg_mean).div_(train_eeg_std + eps)
			ds.eye.sub_(train_eye_mean).div_(train_eye_std + eps)

		train_dataset = torch.utils.data.TensorDataset(train_dataset.eeg, train_dataset.eye, train_dataset.labels)
		val_dataset = torch.utils.data.TensorDataset(val_dataset.eeg, val_dataset.eye, val_dataset.labels)
		test_dataset = torch.utils.data.TensorDataset(test_dataset.eeg, test_dataset.eye, test_dataset.labels)

		return train_dataset, val_dataset, test_dataset
	
	def init_kaiming(self, m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight.data)

	def forward_step(self, batch_eeg, batch_eye, batch_labels, last_batch, batch_idx=None, is_train=True):
		'''统一前向：返回 (loss, logits)。CAFNet 训练与评估的 loss 不同。'''
		if self.model_name == 'EFBook':
			model_output = self.model(batch_eeg, batch_eye, last_batch=last_batch)
			logits = model_output['outputs']
			loss = self.criterion(logits, batch_labels.long()) + model_output['loss']
		elif self.model_name == 'CAFNet':
			logits, c_loss, kd_loss = self.model(batch_eeg, batch_eye, batch_labels)
			if is_train:
				loss = self.criterion(logits, batch_labels, batch_idx, c_loss, kd_loss)
			else:
				loss = self.cafnet_eval_criterion(logits, batch_labels) + self.args.alpha * c_loss + self.args.beta * kd_loss
		elif self.model_name in ['EFNet', 'VigilanceNet', 'TSMMF', 'EyeFormer']:
			logits = self.model(batch_eeg, batch_eye)
			loss = self.criterion(logits, batch_labels.long())
		elif self.model_name in ['EEGNet', 'Conformer']:
			logits = self.model(batch_eeg)
			loss = self.criterion(logits, batch_labels.long())
		return loss, logits

	def evaluate(self, loader):
		'''在给定 loader 上评估，返回指标 dict（含 AUC，需预测概率）。'''
		self.model.eval()
		total_loss, total_correct = 0, 0
		all_preds, all_labels, all_probs = [], [], []
		with torch.no_grad():
			for batch_eeg, batch_eye, batch_labels in loader:
				batch_eeg = batch_eeg.to(self.device, dtype=torch.float32)
				batch_eye = batch_eye.to(self.device, dtype=torch.float32)
				batch_labels = batch_labels.to(self.device)
				loss, logits = self.forward_step(batch_eeg, batch_eye, batch_labels, last_batch=False, is_train=False)
				probs = torch.softmax(logits, dim=1)
				preds = torch.argmax(logits, dim=1)
				total_loss += loss.item()
				total_correct += (preds == batch_labels).sum().item()
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(batch_labels.cpu().numpy())
				all_probs.extend(probs.cpu().numpy())
		try:
			auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
		except ValueError:
			auc = 0.0
		return {
			'loss': total_loss / len(loader),
			'acc': total_correct / len(loader.dataset),
			'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
			'bacc': balanced_accuracy_score(all_labels, all_preds),
			'wf1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
			'kappa': cohen_kappa_score(all_labels, all_preds),
			'auc': auc,
			'labels': all_labels,
			'preds': all_preds,
		}

	def train_epoch(self, train_loader):
		self.model.train()
		total_correct, total_loss = 0, 0
		num_batch = len(train_loader)
		for i, (batch_eeg, batch_eye, batch_labels) in enumerate(train_loader):
			batch_eeg = batch_eeg.to(self.device, dtype=torch.float32)
			batch_eye = batch_eye.to(self.device, dtype=torch.float32)
			batch_labels = batch_labels.to(self.device)
			loss, logits = self.forward_step(batch_eeg, batch_eye, batch_labels, last_batch=(i == num_batch - 1), batch_idx=i, is_train=True)
			preds = torch.argmax(logits, dim=1)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			total_correct += (preds == batch_labels).sum().item()
			total_loss += loss.item()
		return total_loss / len(train_loader), total_correct / len(train_loader.dataset)

	def run(self, train_subs, val_subs, test_subs):
		'''单次训练：train 更新权重，val 选最优 checkpoint + 调 lr，test 只评一次。'''
		dim = 4
		if self.model_name in ['CAFNet', 'VigilanceNet', 'EyeFormer']:
			dim = 3

		train_dataset, val_dataset, test_dataset = read_seed_split(train_subs, val_subs, test_subs, self.dataset, dim=dim, K=self.args.K)

		if self.args.z_score:
			train_dataset, val_dataset, test_dataset = self.z_score_mm(train_dataset, val_dataset, test_dataset)

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

		self.model.apply(self.init_kaiming)

		# 26.10.06: 在数据集维度打印之后、训练结果输出之前打印模型结构
		log_model_structure(self.logger, self.model, title=f'{self.model_name} Model Structure')

		best_val_acc = 0
		best_path = self.results_root + 'best_model.pt'
		wu_lr = self.args.learning_rate / 10
		base_lr = self.args.learning_rate

		# 26.10.06: 早停对所有模型生效（监控 val loss），缓解验证集发散/过拟合
		early_stop_patience = self.args.early_stop_patience
		best_val_loss = float('inf')
		patience_counter = 0

		train_loss_list, train_acc_list = [], []
		val_loss_list, val_acc_list = [], []
		val_precision_list, val_bacc_list, val_wf1_list, val_kappa_list, val_auc_list = [], [], [], [], []

		for epoch in range(self.args.num_epochs):
			# warm-up lr
			if epoch < self.args.warm_up:
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = wu_lr
			elif epoch == self.args.warm_up:
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = base_lr

			train_loss, train_acc = self.train_epoch(train_loader)
			val_metrics = self.evaluate(val_loader)
			self.scheduler.step(val_metrics['loss'])

			if val_metrics['acc'] > best_val_acc:
				best_val_acc = val_metrics['acc']
				torch.save(self.model.state_dict(), best_path)

			# 26.10.06: 早停对所有模型生效（监控 val loss）
			if val_metrics['loss'] < best_val_loss:
				best_val_loss = val_metrics['loss']
				patience_counter = 0
			else:
				patience_counter += 1
			if patience_counter >= early_stop_patience:
				self.logger.info(f'Early stopping at epoch {epoch+1}: val loss not improved for {early_stop_patience} epochs')
				break

			if self.model_name == 'CAFNet':
				self.criterion.increase_threshold()

			train_loss_list.append(train_loss)
			train_acc_list.append(train_acc)
			val_loss_list.append(val_metrics['loss'])
			val_acc_list.append(val_metrics['acc'])
			val_precision_list.append(val_metrics['precision'])
			val_bacc_list.append(val_metrics['bacc'])
			val_wf1_list.append(val_metrics['wf1'])
			val_kappa_list.append(val_metrics['kappa'])
			val_auc_list.append(val_metrics['auc'])

			self.logger.info(f"Epoch {epoch+1}/{self.args.num_epochs} | LR {self.optimizer.param_groups[0]['lr']}")
			self.logger.info(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc*100:.2f} | Val Loss: {val_metrics['loss']:.2f} | Val Acc: {val_metrics['acc']*100:.2f}")

		# 训练结束：加载 val 上最优权重，在 test 上只评一次
		self.model.load_state_dict(torch.load(best_path))
		test_metrics = self.evaluate(test_loader)
		self.logger.info(f"=== TEST === Acc: {test_metrics['acc']*100:.2f} | Precision: {test_metrics['precision']*100:.2f} | B-ACC: {test_metrics['bacc']*100:.2f} | W-F1: {test_metrics['wf1']*100:.2f} | Kappa: {test_metrics['kappa']:.4f} | AUC: {test_metrics['auc']:.4f}")

		# 统一保存：前 num_epochs 行为 train/val 过程，最后一行 test（无法计算的项记 0）
		data = {
			'train loss': train_loss_list + [0.0],
			'train acc': [a * 100 for a in train_acc_list] + [0.0],
			'val loss': val_loss_list + [0.0],
			'val acc': [a * 100 for a in val_acc_list] + [0.0],
			'precision': [p * 100 for p in val_precision_list] + [test_metrics['precision'] * 100],
			'B-ACC': [b * 100 for b in val_bacc_list] + [test_metrics['bacc'] * 100],
			'W-F1': [f * 100 for f in val_wf1_list] + [test_metrics['wf1'] * 100],
			'kappa': val_kappa_list + [test_metrics['kappa']],
			'AUC': val_auc_list + [test_metrics['auc']],
		}
		metrics(self.results_root + 'results.xlsx', data)

		return test_metrics

args = get_args()

dataset = args.dataset
logger = setup_logger('Logs/', log_filename=f'{args.model}_{dataset}_train_{LOG_TIMESTAMP}.log')

start_time = time.time()
curr_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.localtime(start_time))
exp_name = f'{args.model}_{dataset}_{curr_time}/'
results_root = 'Results/' + exp_name


def parse_subs(s):
	if s is None or str(s).strip() == '':
		return None
	return [int(x) for x in str(s).strip().split(',')]


train_subs = parse_subs(args.train_subjects)
val_subs = parse_subs(args.val_subjects)
test_subs = parse_subs(args.test_subjects)

if dataset == 'SEEDIV':
	# SEED-IV 共 15 人，跳过 eye 异常的 #15，剩余 14 人按 8/3/3 划分
	if train_subs is None:
		train_subs = list(range(1, 9))
	if val_subs is None:
		val_subs = list(range(9, 12))
	if test_subs is None:
		test_subs = list(range(12, 15))
elif dataset == 'SEEDV':
	# SEED-V 共 16 人，按 10/3/3 划分
	if train_subs is None:
		train_subs = list(range(1, 11))
	if val_subs is None:
		val_subs = list(range(11, 14))
	if test_subs is None:
		test_subs = list(range(14, 17))

logger.info(f'Train subjects: {train_subs}')
logger.info(f'Val subjects:   {val_subs}')
logger.info(f'Test subjects:  {test_subs}')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

trainer = Trainer(args, exp_name)
test_metrics = trainer.run(train_subs, val_subs, test_subs)

# export parameters
save_param(results_root, args.__dict__, [seed])

# visualization（test 混淆矩阵）
cm_visual = vis('split', dataset, results_root)
cm_visual.plot_cm_all(test_metrics['labels'], test_metrics['preds'])

# send email upon the completion of training
summary = f"Test Acc={test_metrics['acc']*100:.2f} W-F1={test_metrics['wf1']*100:.2f} Kappa={test_metrics['kappa']:.4f} AUC={test_metrics['auc']:.4f}"
send_yagmail(exp_name.split('/')[-2], results_text=summary)