from argparse import ArgumentParser

# 消融实验需要使用Ablation/Utils.py中的get_args()函数
def get_args():
	parser = ArgumentParser()

	parser.add_argument('--num_class', type=int, default=2)
	parser.add_argument('--model', type=str, default='EFBook', help='EEGNet, Conformer, fNIRST, fNIRSNet, CAFNet, EFNet, VigilanceNet, TSMMF, EFBook')
	parser.add_argument('--dataset', type=str, default='EFMA', choices=['EFMI', 'EFMA', 'EFWG'])
	parser.add_argument('--mi_root', type=str, default='data_main/junlin//BFM/Datasets/EF-MI/')
	parser.add_argument('--ma_root', type=str, default='data_main/junlin/BFM/Datasets/EF-MA/')
	parser.add_argument('--wg_root', type=str, default='data_main/junlin/BFM/Datasets/EF-WG/WG/')
	parser.add_argument('--dict_len', type=int, default=128)
	parser.add_argument('--emb_size', type=int, default=64)
	parser.add_argument('--threshold', type=int, default=60)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--warm_up', type=int, default=10)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--early_stop_patience', type=int, default=15, help='早停耐心值（eval loss 连续未改善的 epoch 数），Train.py 对所有模型生效')
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

	args = parser.parse_args()
	return args