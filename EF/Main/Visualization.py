import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Visualization():
	def __init__(self, subject, mode, results_root):
		self.subject = str(subject)
		self.mode = mode
		self.title = ('MI' if mode == 'EFMI' else 'MA' if mode == 'EFMA' else 'WG')
		self.results_root = results_root
		self.curve_path = results_root + f'Curve_{self.subject}_{self.title}.png'
		self.cm_path = results_root + f'CM_{self.subject}_{self.title}.png'

	def _get_title(self):
		return self.title
	
	def _get_save_path(self):
		return self.curve_path, self.cm_path

	def plot_curve(self, train_loss, train_acc, eval_loss, eval_acc):
		fig, ax1 = plt.subplots(figsize=(12, 7))
		plt.title(self.title, fontsize=16)
		
		train_loss = np.array(train_loss)
		train_acc = np.array(train_acc)
		eval_loss = np.array(eval_loss)
		eval_acc = np.array(eval_acc)
		epochs = np.arange(1, len(train_loss)+1, dtype=np.int32)
		
		ax1.set_xlabel('Epoch')
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.set_ylabel('Loss', color='tab:blue')
		line1 = ax1.plot(epochs, train_loss, 'b-o', label='Training_loss')
		line2 = ax1.plot(epochs, eval_loss, 'b--s', label='Testing_Loss')
		ax1.tick_params(axis='y', labelcolor='tab:blue')
		
		ax2 = ax1.twinx()
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.set_ylabel('Accuracy (%)', color='tab:red')
		line3 = ax2.plot(epochs, train_acc, 'r-^', label='Training_Acc')
		line4 = ax2.plot(epochs, eval_acc, 'r--D', label='Testing_Acc')
		ax2.tick_params(axis='y', labelcolor='tab:red')
		ax2.set_ylim(0, 100) 
		
		lines = line1 + line2 + line3 + line4
		labels = [l.get_label() for l in lines]
		ax1.legend(lines, labels, loc='upper center', ncol=2, frameon=True, shadow=True)
		ax1.grid(True, linestyle='--', alpha=0.7)
		
		plt.tight_layout()
		plt.savefig(self.curve_path, dpi=300, bbox_inches='tight')
	
	def plot_cm(self, all_labels, all_preds):
		cm = confusion_matrix(all_labels, all_preds)
		cm = cm / cm.sum(axis=1, keepdims=True)
		labels = ['Task 1', 'Task 2']
		if self.mode == 'EFMI':
			labels = ['Left', 'Right']
		elif self.mode == 'EFMA':
			labels = ['MA', 'BL']
		elif self.mode == 'EFWG':
			labels = ['WG', 'BL']
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
		disp.plot(cmap=plt.cm.Blues)
		plt.title(self.title)
		plt.savefig(self.cm_path, dpi=300, bbox_inches='tight')
		plt.close()
	
	def plot_cm_all(self, all_labels, all_preds):
		cm = confusion_matrix(all_labels, all_preds)
		cm = cm / cm.sum(axis=1, keepdims=True)
		labels = ['Task 1', 'Task 2']
		if self.mode == 'EFMI':
			labels = ['Left', 'Right']
		elif self.mode == 'EFMA':
			labels = ['MA', 'BL']
		elif self.mode == 'EFWG':
			labels = ['WG', 'BL']
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
		disp.plot(cmap=plt.cm.Blues)
		plt.title(self.title)
		cm_save_path = self.results_root + f'CM_Overall_{self.title}.png'
		plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
		plt.close()
		
	def plot_all(self, train_loss, train_acc, eval_loss, eval_acc, all_labels, all_preds):
		self.plot_curve(train_loss, train_acc, eval_loss, eval_acc)
		self.plot_cm(all_labels, all_preds)