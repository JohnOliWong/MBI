import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, cohen_kappa_score

def train(model, criterions, optimizer, scheduler, train_loader, eval_loader, device, num_epoch, alpha, beta, task, init=False):
	def init_kaiming(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight.data)

	if init:
		model.apply(init_kaiming)

	model.to(device)
	
	train_acces, train_losses, valid_acces, valid_losses = [], [], [], []
	acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []

	for epoch in range(num_epoch):
		start = time.time()
		model.train()
		train_epoch_loss = 0
		valid_epoch_loss = 0

		# train
		predict_train, y_true_train = None, None
		test_i = 0
		for batch_eeg, batch_nirs, batch_labels in train_loader:
			batch_eeg = batch_eeg.to(device, dtype=torch.float64)
			batch_nirs = batch_nirs.to(device, dtype=torch.float64)
			batch_labels = batch_labels.to(device)

			predict_train, c_loss, kd_loss = model(batch_eeg, batch_nirs, batch_labels)

			loss = criterions[0](predict_train, batch_labels.long()) + alpha * c_loss + beta * kd_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			train_epoch_loss += loss.item()

			predict_train = predict_train.max(1)[1]

			if test_i == 0:
				pre = predict_train
				y_true_train = batch_labels
			else:
				pre = torch.hstack((pre, predict_train))
				y_true_train = torch.hstack((y_true_train, batch_labels))

			test_i += 1
			
			train_acc = accuracy_score(y_true_train.cpu(), pre.cpu())
			
		scheduler.step()
		# criterions[1].increase_threshold()
		
		# evaluation
		best_acc = 0.0
		predict, y_true = None, None
		model.eval()

		with torch.no_grad():
			eval_i = 0
			for eval_eeg, eval_nirs, eval_labels in eval_loader:
			# for i, (v, labels, index) in enumerate(eval_loader):
				eval_eeg = eval_eeg.to(device, dtype=torch.float64)
				eval_nirs = eval_nirs.to(device, dtype=torch.float64)
				eval_labels = eval_labels.to(device)

				predict, c_loss, kd_loss = model(eval_eeg, eval_nirs, eval_labels)

				loss = criterions[0](predict, eval_labels.long()) + alpha * c_loss + beta * kd_loss
				
				valid_epoch_loss += loss.item()

				predict = predict.max(1)[1]

				if eval_i == 0:
					pre = predict
					y_true = eval_labels
				else:
					pre = torch.hstack((pre, predict))
					y_true = torch.hstack((y_true, eval_labels))
					
				eval_i += 1

			valid_acc = accuracy_score(y_true.cpu(), pre.cpu())
			precision = precision_score(y_true.cpu(), pre.cpu(), zero_division=0)
			recall = recall_score(y_true.cpu(), pre.cpu())
			f1 = f1_score(y_true.cpu(), pre.cpu())
			# precision, recall, f1, _ = precision_recall_fscore_support(y_true.cpu(), pre.cpu(), average='weighted')
			kappa = cohen_kappa_score(y_true.cpu(), pre.cpu())
		
			if valid_acc > best_acc:
				best_acc = valid_acc
				# torch.save(model.state_dict(), 'Results/' + 'Fold_' + str(fold) + '_Best.pth')
			
			train_epoch_loss = train_epoch_loss / len(train_loader)
			valid_epoch_loss = valid_epoch_loss / len(eval_loader)

			end = time.time() - start

		acc_list.append(valid_acc)
		precision_list.append(precision)
		recall_list.append(recall)
		f1_list.append(f1)
		kappa_list.append(kappa)
		
		train_acces.append(train_acc)
		train_losses.append(train_epoch_loss)
		valid_acces.append(valid_acc)
		valid_losses.append(valid_epoch_loss)
		 
		# print("< Fold{} {:.0f}% {}/{} {:.3f}s >".format(fold, (epoch + 1) / num_epoch * 100, epoch + 1, num_epoch, end), end="")
		# print("{:.0f}% {}/{} {:.3f}s >".format((epoch + 1) / num_epoch * 100, epoch + 1, num_epoch, end), end="")
		print('train_loss =', '{:.4f}'.format(train_epoch_loss), end=" ")
		print('train_acc =', '{:.4f}'.format(train_acc), end=" ")
		print('valid_loss =', '{:.4f}'.format(valid_epoch_loss), end=" ")
		print('valid_acc =', '{:.4f}'.format(valid_acc))
	
	return valid_acc, best_acc, acc_list[-50:], precision_list[-50:], recall_list[-50:], f1_list[-50:], kappa_list[-50:], train_losses, valid_losses