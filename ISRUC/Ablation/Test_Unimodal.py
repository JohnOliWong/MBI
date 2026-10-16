import argparse
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from Dataloader import read_seed_split
from EFBook.EF_Book_VQ import EF_Book
from Metrics import load_param
from Utils import setup_logger


# 仿照 EF/Main/Test_Unimodal.py：加载在双模态数据下训练得到的 Split 权重
# （Results/<checkpoint>/best_model.pt），只输入 EEG 或眼动单模态信号，
# 在 holdout 验证/测试受试者上测试；缺失模态用全零张量占位。
#
# # 只测 EEG
# python Test_Unimodal.py --checkpoint EFBook_SEEDV_OT_xxx --modality eeg
#
# # 只测眼动
# python Test_Unimodal.py --checkpoint EFBook_SEEDV_OT_xxx --modality eye
#
# # 两种都测（默认）
# python Test_Unimodal.py --checkpoint EFBook_SEEDV_OT_xxx


def z_score_eval(train_dataset, eval_dataset, eps=1e-6):
	"""用 train 统计量标准化 eval（与训练时 z_score 一致，就地修改 eval）。"""
	train_eeg_mean = train_dataset.eeg.mean(dim=(0, 1), keepdim=True)
	train_eeg_std = train_dataset.eeg.std(dim=(0, 1), keepdim=True)
	train_eye_mean = train_dataset.eye.mean(dim=(0, 1), keepdim=True)
	train_eye_std = train_dataset.eye.std(dim=(0, 1), keepdim=True)

	eval_dataset.eeg.sub_(train_eeg_mean).div_(train_eeg_std + eps)
	eval_dataset.eye.sub_(train_eye_mean).div_(train_eye_std + eps)

	return eval_dataset


def eval_metrics(model, loader, modality, device):
	"""在 loader 上以单模态输入（另一模态置零）评估，返回 (acc, precision, recall, f1, kappa)。"""
	all_preds, all_labels = [], []
	with torch.no_grad():
		for batch_eeg, batch_eye, batch_labels in loader:
			batch_eeg = batch_eeg.to(device, dtype=torch.float32)
			batch_eye = batch_eye.to(device, dtype=torch.float32)
			batch_labels = batch_labels.to(device)

			if modality == 'eeg':
				batch_eye = torch.zeros_like(batch_eye)
			elif modality == 'eye':
				batch_eeg = torch.zeros_like(batch_eeg)

			model_output = model(batch_eeg, batch_eye, last_batch=False)
			outputs = model_output['outputs']
			preds = torch.argmax(outputs, dim=1)

			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(batch_labels.cpu().numpy())

	acc = np.mean(np.array(all_preds) == np.array(all_labels))
	precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
	recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
	f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
	kappa = cohen_kappa_score(all_labels, all_preds)
	return acc, precision, recall, f1, kappa


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=str, required=True, help='Results/ 下的实验目录名（Split 结果，含 Params.json 与 best_model.pt）')
	parser.add_argument('--modality', type=str, default='both', choices=['eeg', 'eye', 'both'])
	parser.add_argument('--device', type=str, default='cuda')
	args = parser.parse_args()

	results_root = os.path.join('Results', args.checkpoint)
	config, seeds = load_param(results_root)
	cfg = argparse.Namespace(**config)

	dataset = getattr(cfg, 'dataset', None)
	if dataset not in ('SEEDIV', 'SEEDV'):
		raise SystemExit(f'Params.json 中 dataset 异常: {dataset}')
	num_class = 4 if dataset == 'SEEDIV' else 5

	train_subs = getattr(cfg, 'train_subjects', None)
	val_subs = getattr(cfg, 'val_subjects', None)
	test_subs = getattr(cfg, 'test_subjects', None)
	if not train_subs or not test_subs:
		raise SystemExit('Params.json 缺少 train/val/test 受试者划分（旧 LOSO checkpoint 不支持本脚本）')

	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	logger = setup_logger('Logs/', log_filename=f'TestUnimodal_{dataset}_{args.modality}_{ts}.log')

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	logger.info(f'Device: {device}')
	logger.info(f'Train: {train_subs} | Val: {val_subs} | Test: {test_subs}')

	# 与训练一致的输入尺寸（旧 checkpoint 未存 seg_len/K 时回退到 Split 默认 800*1/1）
	seg_len = int(getattr(cfg, 'seg_len', 800))
	K = int(getattr(cfg, 'K', 1))
	eeg_time = seg_len * K
	eye_time = K
	logger.info(f'eeg_time={eeg_time} eye_time={eye_time} dict_len={cfg.dict_len}')

	model = EF_Book(cfg.dict_len, cfg.emb_size, cfg.threshold, num_class, dataset, device,
					eeg_time=eeg_time, eye_time=eye_time)
	model = model.to(device, dtype=torch.float32)
	model.load_state_dict(torch.load(os.path.join(results_root, 'best_model.pt'), map_location=device))
	model.eval()

	train_dataset, val_dataset, test_dataset = read_seed_split(
		train_subs, val_subs, test_subs, dataset, dim=4, K=K)
	if getattr(cfg, 'z_score', True):
		val_dataset = z_score_eval(train_dataset, val_dataset)
		test_dataset = z_score_eval(train_dataset, test_dataset)

	batch_size = int(getattr(cfg, 'batch_size', 32))
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	modalities = ['eeg', 'eye'] if args.modality == 'both' else [args.modality]
	for split_name, loader in (('Val', val_loader), ('Test', test_loader)):
		for mod in modalities:
			acc, precision, recall, f1, kappa = eval_metrics(model, loader, mod, device)
			logger.info(f"[{split_name} | {mod}] Acc {acc*100:.2f} Precision {precision*100:.2f} Recall {recall*100:.2f} F1 {f1*100:.2f} Kappa {kappa:.4f}")
