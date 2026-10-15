import torch
import torch.nn as nn
import torch.nn.functional as F

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)

dict_len = 512
emb_size = 64
cont_features = torch.randn(32, 64)
codebook = nn.Parameter(torch.randn(dict_len, emb_size))
# print(codebook)
N = torch.ones(dict_len)
m = codebook.clone()
status = torch.zeros(dict_len)

distances = torch.cdist(cont_features, codebook, p=2)
# print(distances.data.shape)

quan_idx = torch.argmin(distances, dim=-1)
# print(quan_idx)
# print(quan_idx.data.shape)

quan_token = torch.zeros(32, 64)
quan_token = codebook[quan_idx]
# print(quan_token)
# print(quan_token.data.shape)

quan_token = quan_token + (quan_token - cont_features).detach()
codebook_loss = F.mse_loss(quan_token.detach(), cont_features)
print(codebook_loss)

one_hot = F.one_hot(quan_idx, num_classes=len(codebook)).float()
print(one_hot)
print(one_hot.data.shape)

counts = one_hot.sum(dim=[0,1])
matched = (counts > 0)
print(counts.data.shape)
print(matched)

x = torch.randn(16, 200, 64)
x_pool = x.permute(0, 2, 1)  # [16, 64, 200]
pool = nn.AvgPool1d(kernel_size=4, stride=4)
output = pool(x_pool)  # [16, 64, 50]

output = output.permute(0, 2, 1)  # [16, 50, 64]
projection = nn.Linear(64, 16)
output = projection(output)

print("Input: ", x.shape)
print("Output: ", output.shape)