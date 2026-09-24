from argparse import ArgumentParser

def get_args():
	parser = ArgumentParser()

	parser.add_argument('--num_class', type=int, default=2)
	parser.add_argument('--model', type=str, default='Conformer', help='EEGNet, Conformer, fNIRS-T, fNIRS-Net, CAF-Net, EF-Net, Vigilance-Net, TSMMF, EF-Book')
	parser.add_argument('--dataset', type=str, default='EFWG', choices=['EFMI', 'EFMA', 'EFWG'])
	parser.add_argument('--mi_root', type=str, default='data_main/junlin//BFM/Datasets/EF-MI/')
	parser.add_argument('--ma_root', type=str, default='data_main/junlin/BFM/Datasets/EF-MA/')
	parser.add_argument('--wg_root', type=str, default='data_main/junlin/BFM/Datasets/EF-WG/WG/')
	parser.add_argument('--dict_len', type=int, default=1800)
	parser.add_argument('--emb_size', type=int, default=64)
	parser.add_argument('--threshold', type=int, default=60)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--warm_up', type=int, default=10)
	parser.add_argument('--num_epochs', type=int, default=50)
	parser.add_argument('--learning_rate', type=float, default=5e-4)
	parser.add_argument('--mode', type=int, default=0)
	parser.add_argument('--exp_name', type=str, default='8100/')
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
	parser.add_argument('--conv_dropout', type=float, default=0.4)
	parser.add_argument('--self_dropout', type=float, default=0.4)
	parser.add_argument('--cross_dropout', type=float, default=0.4)
	parser.add_argument('--cls_dropout', type=float, default=0.5)

	args = parser.parse_args()
	return args