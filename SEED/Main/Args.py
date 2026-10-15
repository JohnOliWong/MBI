from argparse import ArgumentParser

def get_args():
	parser = ArgumentParser()

	parser.add_argument('--model', type=str, default='EFBook', help='EEGNet, Conformer, EyeFormer, CAFNet, EFNet, VigilanceNet, TSMMF, EFBook')
	parser.add_argument('--dataset', type=str, default='SEEDV', choices=['SEEDIV', 'SEEDV'])
	parser.add_argument('--train_subjects', type=str, default=None, help='逗号分隔的训练受试者 id，如 1,2,3,4,5,6,7,8（默认按数据集内置划分）')
	parser.add_argument('--val_subjects', type=str, default=None, help='逗号分隔的验证受试者 id')
	parser.add_argument('--test_subjects', type=str, default=None, help='逗号分隔的测试受试者 id')
	parser.add_argument('--seediv_root', type=str, default='/data_main/junlin/BFM/Datasets/SEED-IV/')
	parser.add_argument('--seedv_root', type=str, default='/data_main/junlin/BFM/Datasets/SEED-V/')
	parser.add_argument('--seg_len', type=int, default=800, help='一个 4s 段的采样点数（200Hz * 4s）')
	parser.add_argument('--K', type=int, default=1, help='每个样本包含的 4s 段数（样本时长 = 4s * K），需与 PKL-K{K} 目录匹配')
	parser.add_argument('--dict_len', type=int, default=256, help='EF-Book 码本尺寸（26.10.01 1800->512 缓解过拟合；26.10.06 512->128 缓解过拟合+码本坍缩，备用 256）')
	parser.add_argument('--emb_size', type=int, default=64)
	parser.add_argument('--threshold', type=int, default=60, help='死码字替换窗口（batch 数）。D=128 时 60 batch 内每码字期望命中 60*32/128≈15 次，保持 60；若回退 D=256 应改为 120')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--warm_up', type=int, default=5)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--early_stop_patience', type=int, default=15, help='早停耐心值（val/eval loss 连续未改善的 epoch 数），Train_Split 与 Train_LOSO 均对所有模型生效')
	parser.add_argument('--learning_rate', type=float, default=5e-4)
	parser.add_argument('--z_score', type=bool, default=True)
	parser.add_argument('--causal', type=bool, default=True, help='EyeFormer: True=自回归注意力(原版核心), False=双向注意力+CLS')

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