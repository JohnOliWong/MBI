import argparse
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from Dataloader import read_ef_train_si as si_data
from Dataloader import read_wg_pkl_si as wg_data
from EFBook.EF_Book_VQ import EF_Book
from Metrics import load_param
from Utils import setup_logger


# # 只测 EEG
# python Ablation.py --checkpoint EFBook_EFWG_xxx --modality eeg

# # 只测 fNIRS
# python Ablation.py --checkpoint EFBook_EFWG_xxx --modality nirs

# # 两种都测（默认）
# python Ablation.py --checkpoint EFBook_EFWG_xxx

# 加载在多模态数据下训练的模型权重，使用EEG/fNIRS单模态数据测试
# 例如只输入EEG，则使用全零张量为fNIRS模态占位

def z_score_mm(train_dataset, test_dataset, eps=1e-6):
	train_eeg = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
	train_eeg_mean = train_eeg.mean(dim=(0,1), keepdim=True)
	train_eeg_std = train_eeg.std(dim=(0,1),keepdim=True)
	train_nirs = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
	train_nirs_mean = train_nirs.mean(dim=(0,1), keepdim=True)
	train_nirs_std = train_nirs.std(dim=(0,1),keepdim=True)

	eval_eeg = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
	eval_eeg = (eval_eeg - train_eeg_mean) / (train_eeg_std + eps)
	eval_nirs = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
	eval_nirs = (eval_nirs - train_nirs_mean) / (train_nirs_std + eps)
	eval_labels = torch.stack([test_dataset[i][2] for i in range(len(test_dataset))])
	test_dataset = torch.utils.data.TensorDataset(eval_eeg, eval_nirs, eval_labels)

	return test_dataset


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=str, required=True, help='Results/ 下的实验目录名')
	parser.add_argument('--modality', type=str, default='both', choices=['eeg', 'nirs', 'both'])
	parser.add_argument('--device', type=str, default='cuda')
	args = parser.parse_args()

	results_root = os.path.join('Results', args.checkpoint)
	config, seeds = load_param(results_root)
	cfg = argparse.Namespace(**config)

	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	logger = setup_logger('Logs/', log_filename=f'Ablation_{cfg.dataset}_{ts}.log')

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	logger.info(f'Device: {device}')

	model = EF_Book(cfg.dict_len, cfg.emb_size, cfg.threshold, cfg.num_class, cfg.dataset, device)
	model = model.to(device, dtype=torch.float32)

	results_path = Path(results_root)
	subject_list = sorted(
		int(re.search(r'^(\d+)\.pt$', f.name).group(1))
		for f in results_path.glob('*.pt')
		if re.search(r'^(\d+)\.pt$', f.name)
	)
	logger.info(f'Subjects: {subject_list}')

	modalities = ['eeg', 'nirs'] if args.modality == 'both' else [args.modality]
	summary = {m: {k: [] for k in ['Acc', 'Precision', 'Recall', 'F1', 'Kappa']} for m in modalities}

	for subject in subject_list:
		if cfg.dataset in ('EFMI', 'EFMA'):
			train_dataset, eval_dataset = si_data(subject, cfg.dataset, dim=4)
		elif cfg.dataset == 'EFWG':
			train_dataset, eval_dataset = wg_data(subject, cfg.dataset, dim=4)

		if getattr(cfg, 'z_score', True):
			eval_dataset = z_score_mm(train_dataset, eval_dataset)

		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False)

		model.load_state_dict(torch.load(os.path.join(results_root, f'{subject}.pt'), map_location=device))
		model.eval()

		for mod in modalities:
			all_preds, all_labels = [], []
			with torch.no_grad():
				for batch_eeg, batch_nirs, batch_labels in eval_loader:
					batch_eeg = batch_eeg.to(device, dtype=torch.float32)
					batch_nirs = batch_nirs.to(device, dtype=torch.float32)
					batch_labels = batch_labels.to(device)

					if mod == 'eeg':
						batch_nirs = torch.zeros_like(batch_nirs)
					elif mod == 'nirs':
						batch_eeg = torch.zeros_like(batch_eeg)

					model_output = model(batch_eeg, batch_nirs, last_batch=False)
					outputs = model_output['outputs']
					preds = torch.argmax(outputs, dim=1)

					all_preds.extend(preds.cpu().numpy())
					all_labels.extend(batch_labels.cpu().numpy())

			acc = np.mean(np.array(all_preds) == np.array(all_labels))
			precision = precision_score(all_labels, all_preds, zero_division=0)
			recall = recall_score(all_labels, all_preds, zero_division=0)
			f1 = f1_score(all_labels, all_preds, zero_division=0)
			kappa = cohen_kappa_score(all_labels, all_preds)

			summary[mod]['Acc'].append(acc)
			summary[mod]['Precision'].append(precision)
			summary[mod]['Recall'].append(recall)
			summary[mod]['F1'].append(f1)
			summary[mod]['Kappa'].append(kappa)

			logger.info(f"Subject {subject:2d} | {mod:4s} | Acc {acc:.4f} Precision {precision:.4f} Recall {recall:.4f} F1 {f1:.4f} Kappa {kappa:.4f}")

	for mod in modalities:
		logger.info(f"[{mod}] mean over {len(summary[mod]['Acc'])} subjects | "
					f"Acc {np.mean(summary[mod]['Acc'])*100:.2f} "
					f"Precision {np.mean(summary[mod]['Precision'])*100:.2f} "
					f"Recall {np.mean(summary[mod]['Recall'])*100:.2f} "
					f"F1 {np.mean(summary[mod]['F1'])*100:.2f} "
					f"Kappa {np.mean(summary[mod]['Kappa']):.2f}")