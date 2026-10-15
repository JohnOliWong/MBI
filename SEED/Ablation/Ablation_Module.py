import time
from datetime import datetime

import torch

from Utils import get_args
from Utils import setup_logger
from EF_Book_Module import EF_Book_Module
from Split import run_split


# 模块消融入口（OT / DECOUPLE / PC / SC）：
#   python Ablation_Module.py --dataset SEEDV --mode OT          # 去掉 ot_loss
#   python Ablation_Module.py --dataset SEEDV --mode DECOUPLE    # 去掉 disentangling_loss
#   python Ablation_Module.py --dataset SEEDIV --mode PC         # 去掉 private codebook
#   python Ablation_Module.py --dataset SEEDIV --mode SC         # 去掉 shared codebook
# 每次恰好传入一个 --mode，日志名自动为 EFBook_<dataset>_<MODE>_train_<ts>.log。

MODE_MAP = {
	'OT': 'use_ot_loss',
	'DECOUPLE': 'use_disentangling',
	'PC': 'use_private_codebook',
	'SC': 'use_shared_codebook',
}


def parse_flags(mode):
	"""根据 --mode 返回四个开关（默认全 True，只关闭对应模块）。"""
	if mode not in MODE_MAP:
		raise SystemExit(f'--mode 必须是 {list(MODE_MAP)} 之一，当前是 {mode!r}')
	flags = {
		'use_ot_loss': True,
		'use_disentangling': True,
		'use_private_codebook': True,
		'use_shared_codebook': True,
	}
	flags[MODE_MAP[mode]] = False
	return flags

if __name__ == '__main__':
	args = get_args()
	dataset = args.dataset
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	num_class = 4 if dataset == 'SEEDIV' else 5

	ablation_type = args.mode
	flags = parse_flags(ablation_type)

	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	log_filename = f'EFBook_{dataset}_{ablation_type}_train_{ts}.log'
	logger = setup_logger('Logs/', log_filename=log_filename)

	curr_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.localtime(time.time()))
	exp_name = f'EFBook_{dataset}_{ablation_type}_{curr_time}/'

	def model_builder():
		return EF_Book_Module(
			args.dict_len, args.emb_size, args.threshold, num_class, dataset, device,
			eeg_time=args.seg_len * args.K, eye_time=args.K,
			use_ot_loss=flags['use_ot_loss'],
			use_disentangling=flags['use_disentangling'],
			use_private_codebook=flags['use_private_codebook'],
			use_shared_codebook=flags['use_shared_codebook'],
		)

	def forward_fn(model, eeg, eye, last_batch):
		return model(eeg, eye, last_batch)

	logger.info(f'Ablation Type: {ablation_type} | Dataset: {dataset}')
	logger.info(f'Flags: {flags}')
	run_split(args, dataset, model_builder, forward_fn, exp_name, logger, device)