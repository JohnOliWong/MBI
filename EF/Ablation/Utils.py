import os
import sys

# Ablation/ 下的入口脚本运行时，把上一级目录 MBI/EF 加入 sys.path，
# 以便 import EFBook / Dataloader / Metrics 等上级模块
# 加入系统目录后，从EF或者Ablation目录都可以运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
import yagmail
from loguru import logger
import smtplib
import ssl
from email.mime.text import MIMEText
from email.header import Header
import os
import datetime
import logging
from collections import Counter

from torch import nn


def get_args():
	parser = ArgumentParser()

	parser.add_argument('--num_class', type=int, default=2)
	parser.add_argument('--model', type=str, default='EFBook', help='EEGNet, Conformer, fNIRST, fNIRSNet, CAFNet, EFNet, VigilanceNet, TSMMF, EFBook')
	parser.add_argument('--dataset', type=str, default='EFMI', choices=['EFMI', 'EFMA', 'EFWG'])
	parser.add_argument('--mi_root', type=str, default='data_main/junlin//BFM/Datasets/EF-MI/')
	parser.add_argument('--ma_root', type=str, default='data_main/junlin/BFM/Datasets/EF-MA/')
	parser.add_argument('--wg_root', type=str, default='data_main/junlin/BFM/Datasets/EF-WG/WG/')
	parser.add_argument('--dict_len', type=int, default=1800)
	parser.add_argument('--emb_size', type=int, default=64)
	parser.add_argument('--threshold', type=int, default=60)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--warm_up', type=int, default=10)
	parser.add_argument('--num_epochs', type=int, default=50)
	parser.add_argument('--early_stop_patience', type=int, default=15, help='早停耐心值（eval loss 连续未改善的 epoch 数），LOSO 对所有消融模型生效')
	parser.add_argument('--learning_rate', type=float, default=5e-4)
	parser.add_argument('--z_score', type=bool, default=True)

	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--beta', type=float, default=0.4)
	parser.add_argument('--spl_lambda', type=float, default=1.0)
	parser.add_argument('--spl_gamma', type=float, default=1.15)

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

	# 消融实验：模块消融用 --mode OT/DECOUPLE/PC/SC，单模态用 --modality eeg/nirs
	parser.add_argument('--mode', type=str, default='OT', choices=['OT', 'DECOUPLE', 'PC', 'SC'],
	                    help='模块消融：OT=去掉ot_loss, DECOUPLE=去掉disentangling, PC=去掉private codebook, SC=去掉shared codebook')
	parser.add_argument('--modality', type=str, default='eeg', choices=['eeg', 'nirs'])

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


def integrity(results_dir=None, expected_epochs=50):
    """检查 Results/ 下各 checkpoint 的完整性。

    1) 受试者是否齐全：EFMI/EFMA 期望 29 个，EFWG 期望 26 个
    2) 每个受试者 xlsx 的 epoch 行数是否等于 expected_epochs
    3) 列出完全缺失的 (dataset, mode) 组合
    """
    import pandas as pd

    MODES = ['OT', 'DECOUPLE', 'PC', 'SC', 'EEG', 'fNIRS']

    if results_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'Results')

    if not os.path.isdir(results_dir):
        print(f'[integrity] 目录不存在: {results_dir}')
        return

    print('=' * 90)
    print(f'完整性检查：{results_dir}（期望 {expected_epochs} epochs）')
    print('=' * 90)

    checkpoints = sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d != 'Temp'
    )
    if not checkpoints:
        print('未找到任何 checkpoint 目录。')
        return

    present_combos = set()
    for name in checkpoints:
        root = os.path.join(results_dir, name)
        parts = name.split('_')
        dataset = parts[1] if len(parts) > 1 else '?'
        mode = parts[2] if len(parts) > 2 else '?'
        present_combos.add((dataset, mode))
        expected_subjects = 29 if dataset in ('EFMI', 'EFMA') else 26

        subjects = []
        for f in os.listdir(root):
            if f.endswith('.xlsx'):
                stem = f[:-5]
                if stem.isdigit():
                    subjects.append(int(stem))
        subjects = sorted(set(subjects))
        missing = [s for s in range(1, expected_subjects + 1) if s not in subjects]

        epoch_problems = []
        for s in subjects:
            path = os.path.join(root, f'{s}.xlsx')
            try:
                df = pd.read_excel(path)
                n = len(df)
            except Exception:
                epoch_problems.append(f's{s}=读取失败')
                continue
            if n != expected_epochs:
                epoch_problems.append(f's{s}={n}')

        ok = (not missing) and (not epoch_problems)
        print(f'\n[{"OK" if ok else "异常"}] {name}')
        print(f'  dataset={dataset} mode={mode}  受试者 {len(subjects)}/{expected_subjects}')
        if missing:
            print(f'  缺失受试者: {missing}')
        if epoch_problems:
            print(f'  epoch 异常: {epoch_problems}')
        else:
            print(f'  epoch 检查: {len(subjects)} 个受试者全部为 {expected_epochs} epochs')

    # 完全缺失的 (dataset, mode) 组合
    datasets = sorted({d for d, _ in present_combos})
    missing_combos = [
        (d, m) for d in datasets for m in MODES
        if (d, m) not in present_combos
    ]
    print('\n' + '=' * 90)
    if missing_combos:
        print('完全缺失的 (dataset, mode) 组合:')
        for d, m in missing_combos:
            print(f'  - {d} / {m}')
    else:
        print('没有完全缺失的 (dataset, mode) 组合。')
    print('=' * 90)
    print('完整性检查结束')
    print('=' * 90)