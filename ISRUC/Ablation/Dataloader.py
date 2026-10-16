'''数据读取：复用 Main/Dataloader.py 中 SEED-IV/SEED-V 的 Split 读取函数
（26.10.06 由 LOSO 改为 Split，新增 read_seed_split）。

用文件路径方式加载 Main/Dataloader.py，避免与"本文件也叫 Dataloader"发生模块名冲突
（本目录脚本 import Dataloader 时会先命中本文件）。
'''
import importlib.util
import os

_MAIN_DL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Main', 'Dataloader.py')
_spec = importlib.util.spec_from_file_location('_seed_main_dataloader', _MAIN_DL)
_dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dl)

read_seediv = _dl.read_seediv
read_seedv = _dl.read_seedv
read_seed_pkls = _dl.read_seed_pkls
read_seed_split = _dl.read_seed_split
SeedDataset = _dl.SeedDataset