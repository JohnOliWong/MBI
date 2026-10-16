import time
from datetime import datetime

import torch

from Utils import get_args
from Utils import send_yagmail
from Utils import setup_logger
from EF_Book_Unimodal import EF_Book_Unimodal
from Split import run_split


# 单模态消融入口（从训练到测试都只用一个模态）：
#   python Ablation_Modality.py --dataset SEEDV  --modality eeg    # EEG
#   python Ablation_Modality.py --dataset SEEDV  --modality eye    # 眼动
#   python Ablation_Modality.py --dataset SEEDIV --modality eeg    # EEG
#   python Ablation_Modality.py --dataset SEEDIV --modality eye    # 眼动
# 日志名自动为 EFBook_<dataset>_EEG_train_<ts>.log / EFBook_<dataset>_EYE_train_<ts>.log。

TYPE_MAP = {'eeg': 'EEG', 'eye': 'EYE'}

if __name__ == '__main__':
	args = get_args()
	dataset = args.dataset
	modality = args.modality
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	num_class = 4 if dataset == 'SEEDIV' else 5

	ablation_type = TYPE_MAP[modality]

	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	log_filename = f'EFBook_{dataset}_{ablation_type}_train_{ts}.log'
	logger = setup_logger('Logs/', log_filename=log_filename)

	curr_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.localtime(time.time()))
	exp_name = f'EFBook_{dataset}_{ablation_type}_{curr_time}/'

	def model_builder():
		return EF_Book_Unimodal(
			args.dict_len, args.emb_size, args.threshold, num_class, dataset, device,
			modality=modality,
			eeg_time=args.seg_len * args.K, eye_time=args.K,
		)

	if modality == 'eeg':
		def forward_fn(model, eeg, eye, last_batch):
			return model(eeg, last_batch)
	else:
		def forward_fn(model, eeg, eye, last_batch):
			return model(eye, last_batch)

	logger.info(f'Ablation Type: {ablation_type} | Dataset: {dataset} | Modality: {modality}')
	run_split(args, dataset, model_builder, forward_fn, exp_name, logger, device)
	send_yagmail(log_filename[:-len('.log')])