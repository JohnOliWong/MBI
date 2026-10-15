import argparse
import sys
from typing import Optional, Dict, Any

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt: Optional[argparse.Namespace] = None

    def initialize(self):

        self.parser.add_argument('--task', type=str, default='Arousal', help='the type of classification task')
        self.parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training [default: 40]')
        self.parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train [default: 200]')
        self.parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
        self.parser.add_argument('--alpha', type=float, default=0.5, help='alpha value of loss funciton')
        self.parser.add_argument('--beta', type=float, default=0.4, help='beta value of loss funciton')
        self.parser.add_argument('--spl_lambda', type=float, default=1, help='lambda value of spl process')
        self.parser.add_argument('--spl_gamma', type=float, default=1.15, help='gamma value of spl process')

        self.initialized = True

    def parse(self, args_list: Optional[list] = None) -> argparse.Namespace:
            """Parse arguments with Jupyter notebook compatibility"""
            if not self.initialized:
                self.initialize()
                
            # Handle Jupyter notebook case
            if 'ipykernel' in sys.modules and args_list is None:
                self.opt = argparse.Namespace(
                    task='Arousal',
                    mode=1,
                    batch_size=16,
                    epochs=100,
                    lr=1e-3,
                    alpha=0.5,
                    beta=0.4,
                    spl_lambda=1.0,
                    spl_gamma=1.15,
                    log_name='71A',
                    log_mode=1,
                )
            else:
                # Normal command line parsing
                self.opt = self.parser.parse_args(args=args_list)
                
            return self.opt