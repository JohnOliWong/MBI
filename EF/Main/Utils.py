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

def send_mail():
	smtp_servers = ['smtp.qq.com']
	smtp_ports = [465, 587]
	emails = ['johnoliwong@foxmail.com']
	auth_codes = ['dtfzabqcbegehgbb']
	sender_id = 1
	receiver_id = 1
	port_id = 0
	smtp_server = smtp_servers[sender_id]
	smtp_port = smtp_ports[port_id]
	sender = emails[sender_id]
	receiver = emails[receiver_id]
	auth_code = auth_codes[sender_id]
	subject = 'Training Completed'
	content = 'Training on the remote server is completed'

	message = MIMEText(content, 'plain', 'utf-8')
	message['From'] = Header(sender, 'utf-8')
	message['To'] = Header(receiver, 'utf-8')
	message['Subject'] = Header(subject, 'utf-8')
	context = ssl.create_default_context()

	try:
		with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
			server.starttls(context=context) # Foxmail's server doesn't support STARTTLS extension
			server.login(sender, auth_code)
			server.sendmail(sender, receiver, message.as_string())
			server.quit()
			print('Email Sent')
	except Exception as e:
		print(f'Error: {str(e)}')
		if "Authentication failed" in str(e):
			print("Authentication Failed")
		elif "Connection refused" in str(e):
			print("Connection Refused")
		elif "SSL" in str(e) or "TLS" in str(e):
			print("SSL/TLS Error")

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