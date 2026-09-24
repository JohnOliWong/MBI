import yagmail
from loguru import logger
import smtplib
import ssl
from email.mime.text import MIMEText
from email.header import Header
import os
import datetime
import logging


def send_yagmail(exp_name):
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
	email_content = [f"The training process of experiment {exp_name} on the remote server is completed"]
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