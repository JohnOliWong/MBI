import json
import os

import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from Dataloader import read_seed_split
from Metrics import metrics, save_param
from Utils import log_model_structure


def z_score_mm(train_dataset, val_dataset, test_dataset, eps=1e-6):
	"""按 train 统计量对 train/val/test 就地标准化（沿 batch 维计算每个 (C, T) 位置的均值/方差）。"""
	train_eeg_mean = train_dataset.eeg.mean(dim=(0, 1), keepdim=True)
	train_eeg_std = train_dataset.eeg.std(dim=(0, 1), keepdim=True)
	train_eye_mean = train_dataset.eye.mean(dim=(0, 1), keepdim=True)
	train_eye_std = train_dataset.eye.std(dim=(0, 1), keepdim=True)

	for ds in (train_dataset, val_dataset, test_dataset):
		ds.eeg.sub_(train_eeg_mean).div_(train_eeg_std + eps)
		ds.eye.sub_(train_eye_mean).div_(train_eye_std + eps)

	return train_dataset, val_dataset, test_dataset


def get_default_subs(dataset):
	"""与 Main/Train_Split.py 一致的 holdout 划分。"""
	if dataset == 'SEEDIV':
		# SEED-IV 共 15 人，跳过 eye 异常的 #15，剩余 14 人按 8/3/3 划分
		return list(range(1, 9)), list(range(9, 12)), list(range(12, 15))
	# SEED-V 共 16 人，按 10/3/3 划分
	return list(range(1, 11)), list(range(11, 14)), list(range(14, 17))


def parse_subs(s):
	if s is None or str(s).strip() == '':
		return None
	return [int(x) for x in str(s).strip().split(',')]


def evaluate(model, loader, forward_fn, criterion, device):
	"""在 loader 上评估，返回 (loss, acc, preds, labels)。"""
	model.eval()
	total_loss, total_correct = 0, 0
	all_preds, all_labels = [], []
	with torch.no_grad():
		for batch_eeg, batch_eye, batch_labels in loader:
			batch_eeg = batch_eeg.to(device, dtype=torch.float32)
			batch_eye = batch_eye.to(device, dtype=torch.float32)
			batch_labels = batch_labels.to(device)

			model_output = forward_fn(model, batch_eeg, batch_eye, False)
			outputs = model_output['outputs']
			preds = torch.argmax(outputs, dim=1)
			loss = criterion(outputs, batch_labels.long()) + model_output['loss']

			total_loss += loss.item()
			total_correct += (preds == batch_labels).sum().item()
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(batch_labels.cpu().numpy())

	return total_loss / len(loader), total_correct / len(loader.dataset), all_preds, all_labels


def run_split(args, dataset, model_builder, forward_fn, exp_name, logger, device):
	"""Split（holdout）训练循环（SEED EFBook 消融通用，26.10.06 由 LOSO 改为 Split）。

	model_builder: 无参函数，构建全新模型。
	forward_fn:    (model, eeg, eye, last_batch) -> {'outputs': ..., 'loss': ...}
	划分与 Main/Train_Split.py 一致：SEED-IV 8/3/3（跳过 #15），SEED-V 10/3/3，
	可用 --train_subjects/--val_subjects/--test_subjects 覆盖。
	"""
	results_root = 'Results/' + exp_name
	os.makedirs(results_root, exist_ok=True)

	train_subs = parse_subs(getattr(args, 'train_subjects', None))
	val_subs = parse_subs(getattr(args, 'val_subjects', None))
	test_subs = parse_subs(getattr(args, 'test_subjects', None))
	default_train, default_val, default_test = get_default_subs(dataset)
	train_subs = train_subs if train_subs is not None else default_train
	val_subs = val_subs if val_subs is not None else default_val
	test_subs = test_subs if test_subs is not None else default_test
	logger.info(f'Train subjects: {train_subs}')
	logger.info(f'Val subjects:   {val_subs}')
	logger.info(f'Test subjects:  {test_subs}')

	seed = 42
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	model = model_builder().to(device, dtype=torch.float32)

	def init_kaiming(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight.data)
	model.apply(init_kaiming)

	train_dataset, val_dataset, test_dataset = read_seed_split(
		train_subs, val_subs, test_subs, dataset, dim=4, K=args.K)
	if getattr(args, 'z_score', True):
		train_dataset, val_dataset, test_dataset = z_score_mm(train_dataset, val_dataset, test_dataset)

	# 26.10.06: 在数据集维度打印之后、训练结果输出之前打印模型结构
	log_model_structure(logger, model, title=f'{args.model} Model Structure')

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	criterion = nn.CrossEntropyLoss()
	weight_decay = 0.2 * args.learning_rate
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.8, patience=10, min_lr=0.1 * args.learning_rate)

	best_val_acc = 0
	best_path = results_root + 'best_model.pt'
	# 26.10.06: 早停对所有消融模型生效（Split 模式，监控 val loss）
	early_stop_patience = getattr(args, 'early_stop_patience', 15)
	best_val_loss = float('inf')
	patience_counter = 0
	wu_lr = args.learning_rate / 10
	base_lr = args.learning_rate
	current_lr = base_lr
	warm_up = args.warm_up

	train_loss_list, train_acc_list = [], []
	val_loss_list, val_acc_list = [], []
	precision_list, recall_list, f1_list, kappa_list = [], [], [], []

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
		for i, (batch_eeg, batch_eye, batch_labels) in enumerate(train_loader):
			batch_eeg = batch_eeg.to(device, dtype=torch.float32)
			batch_eye = batch_eye.to(device, dtype=torch.float32)
			batch_labels = batch_labels.to(device)
			last_batch = (i == num_batch - 1)

			model_output = forward_fn(model, batch_eeg, batch_eye, last_batch)
			outputs = model_output['outputs']
			preds = torch.argmax(outputs, dim=1)
			loss = criterion(outputs, batch_labels.long()) + model_output['loss']

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_correct += (preds == batch_labels).sum().item()
			total_loss += loss.item()

		train_loss = total_loss / len(train_loader)
		train_acc = total_correct / len(train_loader.dataset)

		# ---------- Evaluation ----------
		val_loss, val_acc, all_preds, all_labels = evaluate(model, val_loader, forward_fn, criterion, device)
		scheduler.step(val_loss)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), best_path)

		# 26.10.06: 早停（监控 val loss）
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			patience_counter = 0
		else:
			patience_counter += 1
		if patience_counter >= early_stop_patience:
			logger.info(f'Early stopping at epoch {epoch + 1}: val loss not improved for {early_stop_patience} epochs')
			break

		precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
		recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
		f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
		kappa = cohen_kappa_score(all_labels, all_preds)

		train_loss_list.append(train_loss)
		train_acc_list.append(train_acc)
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)
		precision_list.append(precision)
		recall_list.append(recall)
		f1_list.append(f1)
		kappa_list.append(kappa)

		logger.info(f"Epoch {epoch + 1}/{args.num_epochs} | LR {optimizer.param_groups[0]['lr']}")
		logger.info(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f} | Val Loss: {val_loss:.2f} | Val Acc: {val_acc:.2f}")

	# ---------- Test（加载 val 上最优权重，test 只评一次）----------
	model.load_state_dict(torch.load(best_path, map_location=device))
	test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, forward_fn, criterion, device)
	test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
	test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
	test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
	test_kappa = cohen_kappa_score(test_labels, test_preds)
	logger.info(f"=== TEST === Acc: {test_acc*100:.2f} | Precision: {test_precision*100:.2f} | Recall: {test_recall*100:.2f} | F1: {test_f1*100:.2f} | Kappa: {test_kappa:.4f}")

	# epoch-wise val 指标（Results.py 聚合用）
	data = {
		'Train_Loss': train_loss_list,
		'Train_Acc': train_acc_list,
		'Loss': val_loss_list,
		'Acc': val_acc_list,
		'Precision': precision_list,
		'Recall': recall_list,
		'F1': f1_list,
		'Kappa': kappa_list,
	}
	metrics(results_root + 'results.xlsx', data)

	test_summary = {
		'Loss': float(test_loss),
		'Acc': float(test_acc),
		'Precision': float(test_precision),
		'Recall': float(test_recall),
		'F1': float(test_f1),
		'Kappa': float(test_kappa),
	}
	with open(results_root + 'Test_Metrics.json', 'w', encoding='utf-8') as f:
		json.dump(test_summary, f, indent=2)

	# Params.json：记录实际使用的划分
	params = dict(vars(args))
	params['train_subjects'] = train_subs
	params['val_subjects'] = val_subs
	params['test_subjects'] = test_subs
	save_param(results_root, params, [seed])

	return test_summary
