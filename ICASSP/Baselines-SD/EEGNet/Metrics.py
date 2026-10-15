import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import os

def metrics(subject, log_name, metric_epoch, acc_list, precision_list, recall_list, f1_list, kappa_list):
		mean_acc, std_acc = np.mean(acc_list[-metric_epoch:]), np.std(acc_list[-metric_epoch:])
		mean_precision, std_precision = np.mean(precision_list[-metric_epoch:]), np.std(precision_list[-metric_epoch:])
		mean_recall, std_recall = np.mean(recall_list[-metric_epoch:]), np.std(recall_list[-metric_epoch:])
		mean_f1, std_f1 = np.mean(f1_list[-metric_epoch:]), np.std(f1_list[-metric_epoch:])
		mean_kappa = np.mean(kappa_list[-metric_epoch:])

		mean_acc, std_acc = mean_acc * 100, std_acc * 100
		mean_precision, std_precision = mean_precision * 100, std_precision * 100
		mean_recall, std_recall = mean_recall * 100, std_recall * 100
		mean_f1, std_f1 = mean_f1 * 100, std_f1 * 100
	
		results_dir = 'Results/'
		log_name = str(log_name)
		log_root = results_dir + log_name
		if not os.path.exists(log_root):
			os.makedirs(log_root)
		log_excel = log_root + f'/{log_name}.xlsx'
		new_row = pd.DataFrame([[mean_acc, std_acc, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1, mean_kappa]], 
							columns=['Accuracy', 'Std_Accuracy', 'Precision', 'Std_Precision', 'Recall', 'Std_Recall', 'F1', 'Std_F1', 'Kappa'])
		new_row = new_row.round(2)
		if not os.path.exists(log_excel):
			new_row.to_excel(log_excel, index=False, engine='openpyxl')
		else:
			with pd.ExcelWriter(log_excel, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
				if 'Sheet1' in writer.sheets:
					startrow = writer.sheets['Sheet1'].max_row
				else:
					startrow = 0
				new_row.to_excel(writer, index=False, header=False, startrow=startrow)

		print(f'Subject:{subject} Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n')