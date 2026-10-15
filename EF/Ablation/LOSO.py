import os

import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from Dataloader import read_ef_train_si as si_data
from Dataloader import read_wg_pkl_si as wg_data
from Metrics import metrics, save_param
from Results import summarize_results
from Utils import log_model_structure

_STRUCTURE_LOGGED = False   # LOSO 每个受试者的模型结构相同，只在第一个受试者打印一次


def z_score_mm(train_dataset, test_dataset, eps=1e-6):
	train_eeg = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
	train_eeg_mean = train_eeg.mean(dim=(0, 1), keepdim=True)
	train_eeg_std = train_eeg.std(dim=(0, 1), keepdim=True)
	train_eeg = (train_eeg - train_eeg_mean) / (train_eeg_std + eps)

	train_nirs = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
	train_nirs_mean = train_nirs.mean(dim=(0, 1), keepdim=True)
	train_nirs_std = train_nirs.std(dim=(0, 1), keepdim=True)
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


def run_loso(args, dataset, model_builder, forward_fn, exp_name, logger, device):
	"""留一受试者训练循环（EFBook 消融通用）。

	model_builder: 无参函数，每个受试者构建一个全新模型。
	forward_fn:    (model, eeg, nirs, last_batch) -> {'outputs': ..., 'loss': ...}
	"""
	results_root = 'Results/' + exp_name
	os.makedirs(results_root, exist_ok=True)

	num_subject = 29 if dataset in ('EFMI', 'EFMA') else 26
	seeds = []

	for subject in range(1, num_subject + 1):
		seed = 42
		seeds.append(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

		model = model_builder().to(device, dtype=torch.float32)

		def init_kaiming(m):
			if type(m) == nn.Linear:
				nn.init.kaiming_normal_(m.weight.data)
		model.apply(init_kaiming)

		if dataset in ('EFMI', 'EFMA'):
			train_dataset, eval_dataset = si_data(subject, dataset, dim=4)
		elif dataset == 'EFWG':
			train_dataset, eval_dataset = wg_data(subject, dataset, dim=4)

		if getattr(args, 'z_score', True):
			train_dataset, eval_dataset = z_score_mm(train_dataset, eval_dataset)

		# 26.10.06: 在数据集维度打印之后、训练结果输出之前打印模型结构
		# LOSO 每个受试者构建的模型结构相同，只打印一次
		global _STRUCTURE_LOGGED
		if not _STRUCTURE_LOGGED:
			log_model_structure(logger, model, title=f'{args.model} Model Structure')
			_STRUCTURE_LOGGED = True

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

		criterion = nn.CrossEntropyLoss()
		weight_decay = 0.2 * args.learning_rate
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min', factor=0.8, patience=10, min_lr=0.1 * args.learning_rate)

		best_acc = 0
		# 26.10.06: 早停对所有消融模型生效（LOSO 无验证集，用留一受试者的 eval loss 监控）
		early_stop_patience = getattr(args, 'early_stop_patience', 15)
		best_eval_loss = float('inf')
		patience_counter = 0
		wu_lr = args.learning_rate / 10
		base_lr = args.learning_rate
		current_lr = base_lr
		warm_up = args.warm_up

		train_loss_list, train_acc_list = [], []
		loss_list, acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], [], []

		for epoch in range(args.num_epochs):
			# ---------- Training ----------
			model.train()
			total_correct, total_loss = 0, 0

			if epoch < warm_up:
				current_lr = wu_lr
				for param_group in optimizer.param_groups:
					param_group['lr'] = current_lr
			else:
				if epoch == warm_up:
					for param_group in optimizer.param_groups:
						param_group['lr'] = base_lr
				current_lr = optimizer.param_groups[0]['lr']

			num_batch = len(train_loader)
			for i, (batch_eeg, batch_nirs, batch_labels) in enumerate(train_loader):
				batch_eeg = batch_eeg.to(device, dtype=torch.float32)
				batch_nirs = batch_nirs.to(device, dtype=torch.float32)
				batch_labels = batch_labels.to(device)
				last_batch = (i == num_batch - 1)

				model_output = forward_fn(model, batch_eeg, batch_nirs, last_batch)
				outputs = model_output['outputs']
				preds = torch.argmax(outputs, dim=1)
				quan_loss = model_output['loss']
				cls_loss = criterion(outputs, batch_labels.long())
				loss = cls_loss + quan_loss

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				total_correct += (preds == batch_labels).sum().item()
				total_loss += loss.item()

			train_loss = total_loss / len(train_loader)
			train_acc = total_correct / len(train_loader.dataset)
			train_loss_list.append(train_loss)
			train_acc_list.append(train_acc)

			# ---------- Evaluation ----------
			model.eval()
			total_loss, total_correct = 0, 0
			all_preds, all_labels = [], []
			with torch.no_grad():
				for eval_eeg, eval_nirs, eval_labels in eval_loader:
					eval_eeg = eval_eeg.to(device, dtype=torch.float32)
					eval_nirs = eval_nirs.to(device, dtype=torch.float32)
					eval_labels = eval_labels.to(device)

					model_output = forward_fn(model, eval_eeg, eval_nirs, False)
					outputs = model_output['outputs']
					preds = torch.argmax(outputs, dim=1)
					cls_loss = criterion(outputs, eval_labels.long())
					loss = cls_loss + model_output['loss']

					total_loss += loss.item()
					total_correct += (preds == eval_labels).sum().item()
					all_preds.extend(preds.cpu().numpy())
					all_labels.extend(eval_labels.cpu().numpy())

			eval_loss = total_loss / len(eval_loader)
			scheduler.step(eval_loss)
			eval_acc = total_correct / len(eval_loader.dataset)
			if eval_acc > best_acc:
				best_acc = eval_acc
				torch.save(model.state_dict(), results_root + f'{subject}.pt')

			precision = precision_score(all_labels, all_preds, zero_division=0)
			recall = recall_score(all_labels, all_preds, zero_division=0)
			f1 = f1_score(all_labels, all_preds, zero_division=0)
			kappa = cohen_kappa_score(all_labels, all_preds)

			loss_list.append(eval_loss)
			acc_list.append(eval_acc)
			precision_list.append(precision)
			recall_list.append(recall)
			f1_list.append(f1)
			kappa_list.append(kappa)

			logger.info(f"Subject {subject} | Epoch {epoch + 1}/{args.num_epochs} | LR {optimizer.param_groups[0]['lr']}")
			logger.info(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f} | Eval Loss: {eval_loss:.2f} | Eval Acc: {eval_acc:.2f}")

			# 26.10.06: 早停（监控 eval loss，即留一测试受试者的 loss）
			# 26.10.07: 判定移到本 epoch 指标记录之后，避免 train/eval 列表长度不一致
			if eval_loss < best_eval_loss:
				best_eval_loss = eval_loss
				patience_counter = 0
			else:
				patience_counter += 1
			if patience_counter >= early_stop_patience:
				logger.info(f'Subject {subject} early stopping at epoch {epoch + 1}: eval loss not improved for {early_stop_patience} epochs')
				break

		train_acc_list = [acc * 100 for acc in train_acc_list]
		acc_list = [acc * 100 for acc in acc_list]
		precision_list = [precision * 100 for precision in precision_list]
		recall_list = [recall * 100 for recall in recall_list]
		f1_list = [f1 * 100 for f1 in f1_list]

		data = {
			'Train_Loss': train_loss_list,
			'Train_Acc': train_acc_list,
			'Loss': loss_list,
			'Acc': acc_list,
			'Precision': precision_list,
			'Recall': recall_list,
			'F1': f1_list,
			'Kappa': kappa_list,
		}
		metrics(results_root + f'{subject}.xlsx', data)

	save_param(results_root, args.__dict__, seeds)

	# 26.10.07: 训练完成后汇总 LOSO 各受试者指标，作为邮件正文
	summary, _ = summarize_results(checkpoint_names=[exp_name.rstrip('/')])
	return summary