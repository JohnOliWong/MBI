from Dataloader import read_isruc_split, read_isruc_dual_split, read_isruc_modal_split, DEFAULT_TRAIN, DEFAULT_VAL, DEFAULT_TEST, N_CH, EPOCH_SAMPLES, N_CLASS

from Baselines.EEGNet import EEGNet
from Baselines.Conformer import Conformer
from Baselines.EFNet import EF_Net
from Baselines.VigilanceNet import VigilanceNet
from Baselines.TMMF import HybridTransformer
from Baselines.MyoCNN import MyoCNN
from EFBook.EF_Book_VQ import EF_Book as EF_VQ

from Args import get_args
from Metrics import metrics, save_param
from Utils import setup_logger, log_model_structure, send_yagmail

import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, f1_score, cohen_kappa_score, balanced_accuracy_score, roc_auc_score
import os
from datetime import datetime

LOG_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


class Trainer:
    def __init__(self, args, exp_name):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = args.model
        num_class = args.num_classes  # 5 类：W/N1/N2/N3/REM
        # 双模态模型（EEG + EMG）标记
        self.is_dual = self.model_name in ['EFNet', 'VigilanceNet', 'TSMMF']
        # EFBook：可变模态数（eeg / eeg+emg / eeg+emg+eog）
        self.is_efbook = self.model_name == 'EFBook'
        self.modalities = [m.strip() for m in args.modalities.split(',')]
        # MyoCNN：EMG-only 单模态基线（3 通道）
        self.is_emg = self.model_name == 'MyoCNN'
        # VigilanceNet 用 3D 输入 [B,ch,time]（Conv1d），其余用 4D [B,1,ch,time]（Conv2d）
        self.unsqueeze_input = self.model_name != 'VigilanceNet'

        self.logger = setup_logger('Logs/', log_filename=f'{self.model_name}_ISRUC_train_{LOG_TIMESTAMP}.log')

        if self.model_name == 'EEGNet':
            # ISRUC：单 epoch 分类器，C=11 通道（EEG6+EOG2+EMG3），T=6000（30s@200Hz），5 类
            self.model = EEGNet(C=N_CH, T=EPOCH_SAMPLES, num_classes=num_class,
                                f1=8, depth=4, f2=4).to(self.device, dtype=torch.float32)

        elif self.model_name == 'Conformer':
            # ISRUC：单 epoch 分类器，11 通道 @ 200Hz，5 类；emb_size 需为 num_heads(6) 的整数倍，故用 60
            self.model = Conformer(emb_size=60, depth=args.depth, n_classes=num_class,
                                   channels=N_CH).to(self.device, dtype=torch.float32)

        elif self.model_name == 'MyoCNN':
            # ISRUC：EMG-only 单模态基线（3 通道下颌 EMG，30s epoch，5 类）
            self.model = MyoCNN(in_channels=3, time=EPOCH_SAMPLES, num_classes=num_class).to(self.device, dtype=torch.float32)

        elif self.model_name == 'EFNet':
            # ISRUC：双模态 EEG(6ch) + EMG(3ch)，单 epoch 分类，5 类
            self.model = EF_Net(num_classes=num_class, mode='ISRUC',
                                eeg_time=EPOCH_SAMPLES).to(self.device, dtype=torch.float32)

        elif self.model_name == 'VigilanceNet':
            # ISRUC：双模态 EEG(6ch) + EMG(3ch)，3D 输入 [B,ch,time]，单 epoch 分类
            self.model = VigilanceNet(hidden_size=64, num_heads=4, ffn_dim=128,
                                      attn_drop=0.25, proj_drop=0.25, feed_drop=0.25,
                                      num_classes=num_class, mode='ISRUC',
                                      eeg_time=EPOCH_SAMPLES, emg_time=EPOCH_SAMPLES
                                      ).to(self.device, dtype=torch.float32)

        elif self.model_name == 'TSMMF':
            # ISRUC：双模态 EEG(6ch) + EMG(3ch)，4D 输入 [B,1,ch,time]，单 epoch 分类
            self.model = HybridTransformer(
                args.depth, args.query_size, args.key_size, args.value_size,
                args.emb_size, args.num_heads, args.expansion,
                args.conv_dropout, args.self_dropout, args.cross_dropout, args.cls_dropout,
                num_class, 'ISRUC', self.device,
                eeg_time=EPOCH_SAMPLES, emg_time=EPOCH_SAMPLES,
            ).to(self.device, dtype=torch.float32)

        elif self.model_name == 'EFBook':
            # ISRUC：可变模态（eeg / eeg+emg / eeg+emg+eog）+ 码本
            self.model = EF_VQ(args.dict_len, args.emb_size, args.threshold,
                               num_class, 'ISRUC', self.device,
                               modalities=self.modalities, eeg_time=EPOCH_SAMPLES).to(self.device, dtype=torch.float32)

        else:
            raise ValueError(f'unsupported model: {self.model_name}')

        self.criterion = nn.CrossEntropyLoss()

        weight_decay = 0.2 * args.learning_rate
        min_lr = 0.1 * args.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, min_lr=min_lr)

        self.results_root = 'Results/' + exp_name
        os.makedirs(self.results_root, exist_ok=True)

    def init_kaiming(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight.data)

    def forward_step(self, batch, last_batch=False):
        '''统一前向：返回 (loss, logits, labels)。处理单模态 / 双模态 / 可变模态（EFBook）。'''
        if self.is_efbook:
            *modals, labels = batch
            inputs = {name: m.to(self.device, dtype=torch.float32).unsqueeze(1)
                      for name, m in zip(self.modalities, modals)}
            labels = labels.to(self.device)
            out = self.model(inputs, last_batch=last_batch)
            logits = out['outputs']
            loss = self.criterion(logits, labels) + out['loss']
            return loss, logits, labels
        elif self.is_dual:
            eeg, emg, labels = batch
            eeg = eeg.to(self.device, dtype=torch.float32)
            emg = emg.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device)
            if self.unsqueeze_input:
                eeg = eeg.unsqueeze(1)   # [B,1,6,6000]
                emg = emg.unsqueeze(1)   # [B,1,3,6000]
            logits = self.model(eeg, emg)
            loss = self.criterion(logits, labels)
            return loss, logits, labels
        else:
            eeg, labels = batch
            eeg = eeg.to(self.device, dtype=torch.float32).unsqueeze(1)   # [B,1,11,6000]
            labels = labels.to(self.device)
            logits = self.model(eeg)
            loss = self.criterion(logits, labels)
            return loss, logits, labels

    def evaluate(self, loader):
        self.model.eval()
        total_loss, total_correct = 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                loss, logits, batch_labels = self.forward_step(batch, last_batch=False)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                total_loss += loss.item()
                total_correct += (preds == batch_labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.0
        return {
            'loss': total_loss / len(loader),
            'acc': total_correct / len(loader.dataset),
            'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'bacc': balanced_accuracy_score(all_labels, all_preds),
            'wf1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'kappa': cohen_kappa_score(all_labels, all_preds),
            'auc': auc,
            'labels': all_labels,
            'preds': all_preds,
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_correct, total_loss = 0, 0
        num_batch = len(train_loader)
        for i, batch in enumerate(train_loader):
            loss, logits, batch_labels = self.forward_step(batch, last_batch=(i == num_batch - 1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_labels).sum().item()
            total_loss += loss.item()
        return total_loss / len(train_loader), total_correct / len(train_loader.dataset)

    def run(self, train_subs, val_subs, test_subs):
        # 单 epoch 模型（EEGNet/Conformer）把 20 个 epoch 展开为独立样本
        flatten_epochs = self.model_name in ['EEGNet', 'Conformer']

        if self.is_emg:
            # EMG-only 单模态基线（3 通道下颌 EMG）
            train_dataset, val_dataset, test_dataset = read_isruc_modal_split(
                ['emg'], train_subs, val_subs, test_subs, flatten_epochs=True)
        elif self.is_efbook:
            # EFBook：可变模态，单 epoch 分类
            train_dataset, val_dataset, test_dataset = read_isruc_modal_split(
                self.modalities, train_subs, val_subs, test_subs, flatten_epochs=True)
        elif self.is_dual:
            # 双模态（EEG+EMG），单 epoch 分类
            train_dataset, val_dataset, test_dataset = read_isruc_dual_split(
                train_subs, val_subs, test_subs, flatten_epochs=True)
        else:
            train_dataset, val_dataset, test_dataset = read_isruc_split(
                train_subs, val_subs, test_subs, flatten_epochs=flatten_epochs)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        log_model_structure(self.logger, self.model, title=f'{self.model_name} Model Structure')
        self.model.apply(self.init_kaiming)

        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        best_path = self.results_root + 'best_model.pt'
        wu_lr = self.args.learning_rate / 10
        base_lr = self.args.learning_rate

        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []
        val_precision_list, val_bacc_list, val_wf1_list, val_kappa_list, val_auc_list = [], [], [], [], []

        for epoch in range(self.args.num_epochs):
            if epoch < self.args.warm_up:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = wu_lr
            elif epoch == self.args.warm_up:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = base_lr

            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_metrics['loss'])

            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                torch.save(self.model.state_dict(), best_path)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.args.early_stop_patience:
                self.logger.info(f'Early stopping at epoch {epoch+1}')
                break

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_metrics['loss'])
            val_acc_list.append(val_metrics['acc'])
            val_precision_list.append(val_metrics['precision'])
            val_bacc_list.append(val_metrics['bacc'])
            val_wf1_list.append(val_metrics['wf1'])
            val_kappa_list.append(val_metrics['kappa'])
            val_auc_list.append(val_metrics['auc'])

            self.logger.info(f"Epoch {epoch+1}/{self.args.num_epochs} | LR {self.optimizer.param_groups[0]['lr']:.6f}")
            self.logger.info(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc*100:.2f} | Val Loss: {val_metrics['loss']:.2f} | Val Acc: {val_metrics['acc']*100:.2f}")

        self.model.load_state_dict(torch.load(best_path))
        test_metrics = self.evaluate(test_loader)
        self.logger.info(f"=== TEST === Acc: {test_metrics['acc']*100:.2f} | B-ACC: {test_metrics['bacc']*100:.2f} | W-F1: {test_metrics['wf1']*100:.2f} | Kappa: {test_metrics['kappa']:.4f} | AUC: {test_metrics['auc']:.4f}")

        data = {
            'train loss': train_loss_list + [0.0],
            'train acc': [a * 100 for a in train_acc_list] + [0.0],
            'val loss': val_loss_list + [0.0],
            'val acc': [a * 100 for a in val_acc_list] + [0.0],
            'precision': [p * 100 for p in val_precision_list] + [test_metrics['precision'] * 100],
            'B-ACC': [b * 100 for b in val_bacc_list] + [test_metrics['bacc'] * 100],
            'W-F1': [f * 100 for f in val_wf1_list] + [test_metrics['wf1'] * 100],
            'kappa': val_kappa_list + [test_metrics['kappa']],
            'AUC': val_auc_list + [test_metrics['auc']],
        }
        metrics(self.results_root + 'results.xlsx', data)
        return test_metrics


def parse_subs(s):
    if s is None or str(s).strip() == '':
        return None
    return [int(x) for x in str(s).strip().split(',')]


if __name__ == '__main__':
    args = get_args()
    train_subs = parse_subs(args.train_subjects) or DEFAULT_TRAIN
    val_subs = parse_subs(args.val_subjects) or DEFAULT_VAL
    test_subs = parse_subs(args.test_subjects) or DEFAULT_TEST

    exp_name = f'{args.model}_ISRUC_{LOG_TIMESTAMP}/'
    logger = setup_logger('Logs/', log_filename=f'{args.model}_ISRUC_train_{LOG_TIMESTAMP}.log')
    logger.info(f'Train subjects ({len(train_subs)}): {train_subs}')
    logger.info(f'Val subjects ({len(val_subs)}): {val_subs}')
    logger.info(f'Test subjects ({len(test_subs)}): {test_subs}')

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trainer = Trainer(args, exp_name)
    test_metrics = trainer.run(train_subs, val_subs, test_subs)
    save_param('Results/' + exp_name, args.__dict__, [seed])
    logger.info(f'Final TEST: Acc={test_metrics["acc"]*100:.2f} W-F1={test_metrics["wf1"]*100:.2f} Kappa={test_metrics["kappa"]:.4f} AUC={test_metrics["auc"]:.4f}')

    # 每个模型训练完成后发一封邮件（含测试指标）
    exp_label = args.model + (f' ({args.modalities})' if args.modalities else '')
    results_text = (
        f"Test Acc={test_metrics['acc']*100:.2f}% | "
        f"B-ACC={test_metrics['bacc']*100:.2f}% | "
        f"W-F1={test_metrics['wf1']*100:.2f}% | "
        f"Kappa={test_metrics['kappa']:.4f} | "
        f"AUC={test_metrics['auc']:.4f}"
    )
    try:
        send_yagmail(exp_label, results_text)
        logger.info('Email notification sent.')
    except Exception as e:
        logger.warning(f'Email notification failed: {e}')