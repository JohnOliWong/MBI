import pandas as pd
import os
import json


def metrics(log_root, data):
	new_row = pd.DataFrame(data).round(2)
	if not os.path.exists(log_root):
		new_row.to_excel(log_root, index=False, engine='openpyxl')
	else:
		with pd.ExcelWriter(log_root, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
			if 'Sheet1' in writer.sheets:
				startrow = writer.sheets['Sheet1'].max_row
			else:
				startrow = 0
			new_row.to_excel(writer, index=False, header=False, startrow=startrow)

def save_param(results_root, params, seeds):
	log_root = results_root + 'Params.json'
	if not os.path.exists(log_root):
		os.makedirs(os.path.dirname(log_root), exist_ok=True)
	data = {
		'results_root': results_root,
		'params': params,
		'seeds': seeds,
	}
	with open(log_root, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)

def load_param(results_root):
	log_root = results_root + 'Params.json'
	with open(log_root, 'r', encoding='utf-8') as f:
		data = json.load(f)
	params = data['params']
	seeds = data['seeds']
	return params, seeds
