import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

from params import Config
from utils import SetSeeds
from net import net
from dataset import DeapDataset, MultimodalDataset
from Dataloader_Excel import read_excel_eeg, read_excel_nirs
from train_validate import train
from loss_function import MySpl_loss
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('Results'):
        os.mkdir('Results')

    for subject in range(29):
        subject += 1
        seed = np.random.randint(1, 2025)
        print('seed is {}'.format(seed))
        print('training on:', device)
        SetSeeds(seed)

        args = Config().parse()

        mode = 0
        eeg, labels = read_excel_eeg(subject, mode)
        nirs, _ = read_excel_nirs(subject, mode)
        dataset = MultimodalDataset(eeg, nirs, labels)
        # dataset = DeapDataset(path_view1 = args.path_eeg, path_view2 = args.path_face, path_label = args.path_label)

        K = 5
        KF = KFold(n_splits=K, shuffle=False) # 80/20 for training/testing

        acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []

        fold = 1
        for train_idx, test_idx in KF.split(dataset):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
            train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
            eval_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)
        
            model = net(eeg_dim=160, nirs_dim=29, 
                        hidden_size=256, num_layers=1, 
                        dim=256, heads=1, dim_head=256, 
                        mlp_dim=512, num_classes=2, dropout=0.5).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = MySpl_loss(n_samples=len(train_loader.dataset), batch_size=args.batch_size, alpha=args.alpha, beta=args.beta, spl_lambda=args.spl_lambda, spl_gamma=args.spl_gamma)
            criterions = [criterion1, criterion2]

            acc, best_acc, precision, recall, f1, kappa, _, _ = train(model, criterions, optimizer, scheduler, train_loader, eval_loader, device, args.epochs, args.alpha, args.beta, fold, args.task)

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            kappa_list.append(kappa)

            fold += 1
        
        mean_acc, std_acc = np.mean(acc_list[-50:]), np.std(acc_list[-50:])
        mean_precision, std_precision = np.mean(precision_list[-50:]), np.std(precision_list[-50:])
        mean_recall, std_recall = np.mean(recall_list[-50:]), np.std(recall_list[-50:])
        mean_f1, std_f1 = np.mean(f1_list[-50:]), np.std(f1_list[-50:])
        mean_kappa = np.mean(kappa_list[-50:])

        mean_acc, std_acc = mean_acc * 100, std_acc * 100
        mean_precision, std_precision = mean_precision * 100, std_precision * 100
        mean_recall, std_recall = mean_recall * 100, std_recall * 100
        mean_f1, std_f1 = mean_f1 * 100, std_f1 * 100

        log_path = f'Results/log_s{subject:2d}.txt'
        with open(log_path, 'a') as log_file:
            log_file.write(f'Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n')
            log_file.write(f'Precision: {mean_precision:.2f} ± {std_precision:.2f}\n')
            log_file.write(f'Recall: {mean_recall:.2f} ± {std_recall:.2f}\n')
            log_file.write(f'F1: {mean_f1:.2f} ± {std_f1:2f}\n')
            log_file.write(f'Kappa: {mean_kappa:.2f}\n')
            log_file.write('\n')

        log_excel = f'Results/Log.xlsx'
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