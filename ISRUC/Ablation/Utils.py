import os
from collections import Counter

# 26.10.06: 本目录维护 EFBook 组件的独立副本（Ablation/EFBook/），不再依赖 Main。
# 数据读取仍通过 Dataloader.py 以文件路径方式桥接 Main/Dataloader.py。

from argparse import ArgumentParser
import logging
import datetime

import yagmail
from loguru import logger
from torch import nn


def get_args():
	parser = ArgumentParser()

	parser.add_argument('--model', type=str, default='EFBook')
	parser.add_argument('--dataset', type=str, default='SEEDV', choices=['SEEDIV', 'SEEDV'])
	parser.add_argument('--train_subjects', type=str, default=None, help='逗号分隔的训练受试者 id（Split 模式；默认按数据集内置 8/3/3 或 10/3/3 划分）')
	parser.add_argument('--val_subjects', type=str, default=None, help='逗号分隔的验证受试者 id（Split 模式）')
	parser.add_argument('--test_subjects', type=str, default=None, help='逗号分隔的测试受试者 id（Split 模式）')
	parser.add_argument('--seg_len', type=int, default=800, help='一个 4s 段的采样点数（200Hz * 4s）')
	parser.add_argument('--K', type=int, default=1, help='每个样本包含的 4s 段数（样本时长 = 4s * K），需与 PKL-K{K} 目录匹配')
	parser.add_argument('--dict_len', type=int, default=128, help='EF-Book 码本尺寸（26.10.06 512->128 缓解过拟合+码本坍缩，备用 256）')
	parser.add_argument('--emb_size', type=int, default=64)
	parser.add_argument('--threshold', type=int, default=60, help='死码字替换窗口（batch 数）。D=128 保持 60；若回退 D=256 应改为 120')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--warm_up', type=int, default=5)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--early_stop_patience', type=int, default=15, help='早停耐心值（val loss 连续未改善的 epoch 数），Split 模式对所有消融模型生效')
	parser.add_argument('--learning_rate', type=float, default=5e-4)
	parser.add_argument('--z_score', type=bool, default=True)

	parser.add_argument('--depth', type=int, default=4)
	parser.add_argument('--query_size', type=int, default=64)
	parser.add_argument('--key_size', type=int, default=64)
	parser.add_argument('--value_size', type=int, default=64)
	parser.add_argument('--decay', type=float, default=0.99)
	parser.add_argument('--num_heads', type=int, default=4)
	parser.add_argument('--expansion', type=int, default=2)
	parser.add_argument('--conv_dropout', type=float, default=0.2)
	parser.add_argument('--self_dropout', type=float, default=0.2)
	parser.add_argument('--cross_dropout', type=float, default=0.2)
	parser.add_argument('--cls_dropout', type=float, default=0.3)

	# 消融实验：模块消融用 --mode OT/DECOUPLE/PC/SC，单模态用 --modality eeg/eye
	parser.add_argument('--mode', type=str, default='OT', choices=['OT', 'DECOUPLE', 'PC', 'SC'],
	                    help='模块消融：OT=去掉ot_loss, DECOUPLE=去掉disentangling, PC=去掉private codebook, SC=去掉shared codebook')
	parser.add_argument('--modality', type=str, default='eeg', choices=['eeg', 'eye'])

	args = parser.parse_args()
	return args


def send_yagmail(exp_name, results_text=None):
	smtp_servers = ['smtp.qq.com']
	emails = ['johnoliwong@foxmail.com']
	auth_codes = ['dtfzabqcbegehgbb']
	sender_id = 0
	receiver_id = 0
	smtp_server = smtp_servers[sender_id]
	sender = emails[sender_id]
	receiver = emails[receiver_id]
	auth_code = auth_codes[sender_id]
	yagmail_server = yagmail.SMTP(user=sender, password=auth_code, host=smtp_server)
	email_title = ["Training Completed"]
	body = f"Experiment: {exp_name}\nStatus: Completed"
	if results_text:
		body += "\n=== Results Summary ===\n" + results_text
	email_content = [body]
	file = None
	if os.path.exists('nohup.out'):
		file = 'nohup.out'
	elif os.path.exists(f'{exp_name}.log'):
		file = f'{exp_name}.log'
	yagmail_server.send(to=receiver, subject=email_title, contents=email_content, attachments=file)
	yagmail_server.close()
	logger.info("Email Sent!")


def setup_logger(log_root, timestamp=None, log_filename=None, logger_name="MMTrainer"):
	os.makedirs(log_root, exist_ok=True)
	timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	log_filename = log_filename or f"train_{timestamp}.log"
	log_file = os.path.join(log_root, log_filename)

	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)

	if logger.handlers:
		return logger

	fh = logging.FileHandler(log_file)
	fh.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	fh.setFormatter(fmt)
	ch.setFormatter(fmt)

	logger.addHandler(fh)
	logger.addHandler(ch)
	logger.info(f"Logging to {log_file}")
	return logger

# 打印模型结构
def _kernel_tup(x):
	"""把 kernel/stride/padding 统一转成 tuple 形式便于打印。"""
	return x if isinstance(x, (tuple, list)) else (x,)

def model_structure_text(model, title='Model Structure'):
	"""生成模型结构摘要文本：层类型统计、卷积层明细（kernel/stride/padding/通道）、码本尺寸、参数量。
	通过 model.named_modules() 遍历实现，无需第三方包。"""
	lines = [f'==================== {title} ====================']

	# 1. 层类型统计
	type_counts = Counter(type(m).__name__ for m in model.modules())
	lines.append('Layer Count: ' + ', '.join(f'{k}={v}' for k, v in sorted(type_counts.items())))

	# 2. 卷积层明细
	conv_layers = [(name, m) for name, m in model.named_modules()
	               if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
	lines.append(f'---- Convolution Layers ({len(conv_layers)}) ----')
	if conv_layers:
		for idx, (name, m) in enumerate(conv_layers, 1):
			lines.append(
				f'  {idx:2d}. {name:<45s} {type(m).__name__:<7s} '
				f'in={m.in_channels:<4d} out={m.out_channels:<4d} '
				f'kernel={_kernel_tup(m.kernel_size)} stride={_kernel_tup(m.stride)} '
				f'padding={_kernel_tup(m.padding)} groups={m.groups}'
			)
	else:
		lines.append('  (no convolution layers)')

	# 3. 码本统计（EF-Book 的 Quantization 模块，含码本尺寸）
	codebooks = []
	for name, m in model.named_modules():
		if hasattr(m, 'dict_len'):
			cb = getattr(m, 'codebook', None)
			shape = tuple(cb.shape) if cb is not None else None
			codebooks.append((name, int(m.dict_len), shape))
	if codebooks:
		lines.append(f'---- Codebooks ({len(codebooks)}) ----')
		for name, dlen, shape in codebooks:
			lines.append(f'  {name:<45s} size={dlen:<5d} shape={shape}')

	# 4. 参数量
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	lines.append(f'Total Parameters: {total:,} (trainable: {trainable:,})')
	lines.append('=' * 52)
	return '\n'.join(lines)

def log_model_structure(logger, model, title='Model Structure'):
	"""把模型结构摘要逐行写入 logger（放在数据集维度打印与训练结果之间）。"""
	for line in model_structure_text(model, title).splitlines():
		logger.info(line)