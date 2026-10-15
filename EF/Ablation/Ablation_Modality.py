import time
from datetime import datetime

import torch

from Utils import get_args
from Utils import send_yagmail
from Utils import setup_logger
from EF_Book_Unimodal import EF_Book_Unimodal
from LOSO import run_loso


# 单模态消融入口（从训练到测试都只用一个模态）：
#   python Ablation_Modality.py --dataset EFMI --modality eeg    # EEG
#   python Ablation_Modality.py --dataset EFMI --modality nirs   # fNIRS
#   python Ablation_Modality.py --dataset EFWG --modality eeg    # EEG
#   python Ablation_Modality.py --dataset EFWG --modality nirs   # fNIRS
# 日志名自动为 EFBook_<dataset>_EEG_train_<ts>.log / EFBook_<dataset>_fNIRS_train_<ts>.log。

TYPE_MAP = {'eeg': 'EEG', 'nirs': 'fNIRS'}

if __name__ == '__main__':
	args = get_args()
	dataset = args.dataset
	modality = args.modality
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	ablation_type = TYPE_MAP[modality]

	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	log_filename = f'EFBook_{dataset}_{ablation_type}_train_{ts}.log'
	logger = setup_logger('Logs/', log_filename=log_filename)

	curr_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.localtime(time.time()))
	exp_name = f'EFBook_{dataset}_{ablation_type}_{curr_time}/'

	def model_builder():
		return EF_Book_Unimodal(
			args.dict_len, args.emb_size, args.threshold, args.num_class, dataset, device,
			modality=modality,
		)

	if modality == 'eeg':
		def forward_fn(model, eeg, nirs, last_batch):
			return model(eeg, last_batch)
	else:
		def forward_fn(model, eeg, nirs, last_batch):
			return model(nirs, last_batch)

	logger.info(f'Ablation Type: {ablation_type} | Dataset: {dataset} | Modality: {modality}')
	summary = run_loso(args, dataset, model_builder, forward_fn, exp_name, logger, device)
	send_yagmail(log_filename[:-len('.log')], results_text=summary)