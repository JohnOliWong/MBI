import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
import pandas as pd

from Dataloader.Dataloader_WG import read_wg_data_single
from TemporalHybridTransformer import HybridTransformer as TMMF
from SpatialHybridTransformer import HybridTransformer as SMMF
from TSMMF import HybridTransformer as TSMMF

# Hyperparameters
depth = 4
depth_series = [4, 4] # dedicated for TSMMF
query_size = 64
key_size = 64
value_size = 64
emb_size = 64
num_heads = 4
expansion = 2
conv_dropout = 0.3
self_dropout = 0.3
cross_dropout = 0.3
cls_dropout = 0.5
num_classes = 2 # number of paradigm
batch_size = 16
num_epochs = 200
learning_rate = 0.001
model_flag = 0 # 0, 1, 2 = Temporal, Spatial, Both
eeg_root = f'D:/HIT/MBI/Dataset/EF-WG/Raw/'
nirs_root = f'D:/HIT/MBI/Dataset/EF-WG/Raw/'
baseline = False

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories for results
os.makedirs("Results", exist_ok=True)

# Initialize model inside the fold loop to prevent data leak
if not model_flag:
    model = TMMF(
    depth, query_size, key_size, value_size, emb_size, num_heads, expansion,
    conv_dropout, self_dropout, cross_dropout, cls_dropout, num_classes, device).to(device)
elif model_flag == 1:
    model = SMMF(
    depth, query_size, key_size, value_size, emb_size, num_heads, expansion,
    conv_dropout, self_dropout, cross_dropout, cls_dropout, num_classes, device).to(device)
else:
    model = TSMMF(
    depth_series, query_size, key_size, value_size, emb_size, num_heads, expansion,
    conv_dropout, self_dropout, cross_dropout, cls_dropout, num_classes, device).to(device)

# model was converted to torch.float64 to avoid overflow ealier
model = model.to(torch.float64)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def evaluate_model(eval_loader, model):
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for eval_eeg, eval_nirs, eval_labels in eval_loader:
            eval_eeg, eval_nirs, eval_labels = eval_eeg.to(device), eval_nirs.to(device), eval_labels.to(device)
            eval_eeg = eval_eeg.to(torch.float64)
            eval_nirs = eval_nirs.to(torch.float64)

            outputs = model(eval_eeg, eval_nirs)
            loss = criterion(outputs, eval_labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == eval_labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(eval_labels.cpu().numpy())

    loss = total_loss / len(eval_loader)
    acc = total_correct / len(eval_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)

    print(f"Evaluation Loss: {loss:.2f} | Evaluation Acc: {acc:.2f}")
    return acc, precision, recall, f1, kappa

for subject_id in range(26):
    subject_id += 1
    eeg, nirs, labels = read_wg_data_single(subject_id, eeg_root, nirs_root, baseline)
    eeg, nirs, labels = eeg.to(device), nirs.to(device), labels.to(device)

    train_size = int(0.6 * len(eeg))
    eval_size = len(eeg) - train_size

    # create a tensor dataset
    dataset = torch.utils.data.TensorDataset(eeg, nirs, labels)

    # randomly split the data into training and evaluation sets
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_correct, total_loss = 0, 0

        for batch_eeg, batch_nirs, batch_labels in train_loader:
            batch_eeg = batch_eeg.to(device, dtype=torch.float64)
            batch_nirs = batch_nirs.to(device, dtype=torch.float64)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_eeg, batch_nirs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == batch_labels).sum().item()

        # len(train_loader) is the number of batches
        # len(train_dataset) is the number of trials
        loss = total_loss / len(train_loader)
        acc = total_correct / len(train_loader.dataset)
        print(f"Epoch {epoch} Training Loss: {loss:.2f} | Training Acc: {acc:.2f}")

        acc, precision, recall, f1, kappa = evaluate_model(eval_loader, model)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        kappa_list.append(kappa)

    mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
    mean_precision, std_precision = np.mean(precision_list), np.std(precision_list)
    mean_recall, std_recall = np.mean(recall_list), np.std(recall_list)
    mean_f1, std_f1 = np.mean(f1_list), np.std(f1_list)
    mean_kappa = np.mean(kappa_list)

    mean_acc, std_acc = mean_acc*100, std_acc*100
    mean_precision, std_precision = mean_precision*100, std_precision*100
    mean_recall, std_recall = mean_recall*100, std_recall*100
    mean_f1, std_f1 = mean_f1*100, std_f1*100

    log_path = f'Results/log_s{subject_id:2d}.txt'
    log_excel = 'Results/Log.xlsx'
    with open(log_path, 'a') as log_file:
        log_file.write(f'Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n')
        log_file.write(f'Precision: {mean_precision:.2f} ± {std_precision:.2f}\n')
        log_file.write(f'Recall: {mean_recall:.2f} ± {std_recall:.2f}\n')
        log_file.write(f'F1: {mean_f1:.2f} ± {std_f1:2f}\n')
        log_file.write(f'Kappa: {mean_kappa:.2f}\n')
        log_file.write('\n')

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

    print(f'Subject:{subject_id} Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n')

print('Training and evaluation completed\n')