import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Tuple

# If you have any questions, please contact us at shulmt@shu.edu.cn

# pearson correlation
def pearson_r(eeg, fnirs):
	mx = torch.mean(eeg, dim=1, keepdim=True)
	my = torch.mean(fnirs, dim=1, keepdim=True)
	xm, ym = eeg - mx, fnirs - my
	r_num = torch.mean(xm * ym, dim=1)
	r_den = torch.std(xm, dim=1) * torch.std(ym, dim=1) + 1e-6
	plcc = r_num / r_den
	plcc = torch.abs(plcc)
	plcc_meanbatch = torch.mean(plcc)
	return plcc_meanbatch

# positional embedding
class PosEmbedding(nn.Module):
	def __init__(self, shape: Tuple[int, int]):
		super(PosEmbedding, self).__init__()
		self.pos_embedding = nn.Parameter(torch.Tensor(1, *shape))
		nn.init.kaiming_uniform_(self.pos_embedding)

	def forward(self, inputs: Tensor) -> Tensor:
		return inputs + self.pos_embedding

def get_angles(pos: np.ndarray, i: int, d_model: int) -> np.ndarray:
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position: int, d_model: int) -> Tensor:
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.FloatTensor(pos_encoding)

# eeg-guided attention
class EFAttention(nn.Module):
    def __init__(self, emb_size: int, d_model: int, heads: int, drop: float):
        super(EFAttention, self).__init__()
        self.emb_size = emb_size
        self.d_model = d_model
        self.heads = heads
        self.drop = drop
        
        self.q_flat = nn.Flatten()
        self.q_proj = nn.Linear(16*16*32, emb_size)  # Adjusted based on eeg2 shape
        
        self.fusion_proj = nn.Linear(16*16*32, emb_size)  # Adjusted based on eeg2 shape
        
        self.k_flat = nn.Sequential(
            nn.Flatten(start_dim=2),  # Flatten spatial dimensions
            nn.Unflatten(1, (11, -1))  # Reshape to (11, -1)
        )
        self.k_proj = nn.Linear(16*16*32, emb_size)  # Adjusted based on fnirs2 shape
        self.pos = PosEmbedding((11, 16*16*32))  # Adjusted based on fnirs2 shape
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, dropout=drop)
        self.ef_plcc_tracker = 0
        self.ef_plcc_count = 0
        
    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        eeg, fnirs = inputs
        
        q_eeg = self.q_flat(eeg)
        fusion_output = self.fusion_proj(q_eeg)
        
        q_eeg = self.q_proj(q_eeg)
        q_eeg = q_eeg.unsqueeze(1)  # Add sequence dimension
        
        k_fnirs = self.k_flat(fnirs)
        k_fnirs = self.pos(k_fnirs)
        k_fnirs = self.k_proj(k_fnirs)
        
        # MultiheadAttention expects (seq_len, batch, embed_dim)
        q_eeg = q_eeg.transpose(0, 1)
        k_fnirs = k_fnirs.transpose(0, 1)
        
		# eeg tokens are Q, knirs tokens are both K and V
        # query is like a compass, eligible K and V will be matched to the Q
        fnirs_weighted, attention_weights = self.multihead_attn(
            q_eeg, k_fnirs, k_fnirs, need_weights=True
        )
        
        attention_weights = torch.mean(attention_weights, dim=(1, 2))
        q_eeg = torch.mean(q_eeg, dim=0)
        fnirs_weighted = torch.mean(fnirs_weighted, dim=0)
        
        ef_loss = pearson_r(q_eeg, fnirs_weighted)
        self.ef_plcc_tracker += ef_loss.item()
        self.ef_plcc_count += 1
        
        # Return to original dimension order
        fnirs_weighted = fnirs_weighted.unsqueeze(0)
        
        return fusion_output, fnirs_weighted, attention_weights

# compute the mean of input fnirs along the second to last dimension -> tap_fnirs
class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
        
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.mean(inputs, dim=-2, keepdim=True)

# fnirs-guided attention
class FGA(nn.Module):
    def __init__(self, tem_kernel_size: int, fga_loss_name: str):
        super(FGA, self).__init__()
        self.tem_kernel_size = tem_kernel_size
        self.fga_loss_name = fga_loss_name
        
        self.channel_pooling = nn.Conv3d(
            in_channels=1, out_channels=1,
            kernel_size=(3, 3, tem_kernel_size),
            stride=(1, 1, 1), padding='same'
        )
        self.tap_fnirs = GAP()
        self.residual_para = nn.Parameter(torch.zeros(1))
        self.eeg_flatten = nn.Flatten()
        self.fnirs_flatten = nn.Flatten()
        self.fusion_flatten = nn.Flatten()
        
        self.fga_loss_tracker = 0
        self.fga_loss_count = 0
        
    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        eeg_fusion, eeg, fnirs = inputs
        
        # Add channel dimension for Conv3d
        fnirs_attention = fnirs.unsqueeze(1)  # [batch, 1, 11, 16, 16, 30, 2]
        fnirs_attention = fnirs_attention.permute(0, 1, 6, 2, 3, 4, 5)  # [batch, 1, 2, 11, 16, 16, 30]
        # Process each channel separately
        fnirs_attention = torch.stack([
            self.channel_pooling(fnirs_attention[:, :, i]) for i in range(2)
        ], dim=2)
        fnirs_attention = fnirs_attention.permute(0, 2, 3, 4, 5, 6, 1)  # Restore original dim order
        
        fnirs_attention_map = self.tap_fnirs(fnirs_attention)
        fnirs_attention_map = torch.mean(fnirs_attention_map, dim=1)
        fnirs_attention_map_norm = torch.sigmoid(fnirs_attention_map)
        
        eeg_fusion_guided = eeg_fusion * fnirs_attention_map_norm
        
        residual_para_norm = torch.sigmoid(self.residual_para)
        eeg_add = residual_para_norm * eeg + (1 - residual_para_norm) * eeg_fusion
        
        fga_feature = eeg_fusion_guided + eeg_add
        
        eeg_plcc = torch.mean(eeg, dim=(-1, -2))
        eeg_plcc = self.eeg_flatten(eeg_plcc)
        fnirs_attention_map_norm_plcc = self.fnirs_flatten(fnirs_attention_map_norm)
        
        fga_loss = pearson_r(eeg_plcc, fnirs_attention_map_norm_plcc)
        self.fga_loss_tracker += fga_loss.item()
        self.fga_loss_count += 1
        
        return fga_feature

# 3D convolution
class ConvBlock(nn.Module):
    def __init__(self, eeg_filter, eeg_size, eeg_stride,
                fnirs_filter, fnirs_size, fnirs_stride,
                eegfusion_filter, eegfusion_size, eegfusion_stride,
                tem_kernel_size, fga_loss_name, padding):
        super(ConvBlock, self).__init__()
        
        # 计算需要的padding值来保持尺寸
        def calculate_padding(input_size, kernel_size, stride):
            if padding == 'same':
                total_padding = (input_size * stride - input_size + kernel_size - stride) 
                return total_padding // 2
            return 0
        
        # EEG pathway
        eeg_padding = (
            calculate_padding(16, eeg_size[0], eeg_stride[0]),
            calculate_padding(16, eeg_size[1], eeg_stride[1]),
            calculate_padding(600, eeg_size[2], eeg_stride[2])
        )
        self.eeg_conv = nn.Conv3d(1, eeg_filter, kernel_size=eeg_size, 
                                 stride=eeg_stride, padding=eeg_padding)
        self.eeg_bn = nn.BatchNorm3d(eeg_filter)
        
        # fNIRS pathway (输入通道数为2)
        fnirs_padding = (
            calculate_padding(16, fnirs_size[0], fnirs_stride[0]),
            calculate_padding(16, fnirs_size[1], fnirs_stride[1]),
            calculate_padding(30, fnirs_size[2], fnirs_stride[2])
        )
        self.fnirs_conv = nn.Conv3d(2, fnirs_filter, kernel_size=fnirs_size, 
                                   stride=fnirs_stride, padding=fnirs_padding)
        self.fnirs_bn = nn.BatchNorm3d(fnirs_filter)
        
        # Fusion pathway
        eegfusion_padding = (
            calculate_padding(16, eegfusion_size[0], eegfusion_stride[0]),
            calculate_padding(16, eegfusion_size[1], eegfusion_stride[1]),
            calculate_padding(600, eegfusion_size[2], eegfusion_stride[2])
        )
        self.eegfusion_conv = nn.Conv3d(1, eegfusion_filter, kernel_size=eegfusion_size, 
                                       stride=eegfusion_stride, padding=eegfusion_padding)
        self.eegfusion_bn = nn.BatchNorm3d(eegfusion_filter)
        
        self.fga = FGA(tem_kernel_size, fga_loss_name)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, inputs):
        eegfusion, eeg, fnirs = inputs
        
        # EEG processing [B, 1, 16, 16, 600]
        eeg_feature = self.eeg_conv(eeg.permute(0, 4, 1, 2, 3))  # [B, C, D, H, W]
        eeg_feature = self.eeg_bn(eeg_feature)
        eeg_feature = F.elu(eeg_feature)
        eeg_feature = self.dropout(eeg_feature)
        eeg_feature = eeg_feature.permute(0, 2, 3, 4, 1)  # Back to [B, D, H, W, C]
        
        # fNIRS processing [B, 11, 16, 16, 30, 2]
        # 首先合并前两个维度 [B*11, 16, 16, 30, 2]
        original_shape = fnirs.shape
        fnirs = fnirs.reshape(-1, *fnirs.shape[2:])
        fnirs_feature = self.fnirs_conv(fnirs.permute(0, 4, 1, 2, 3))  # [B*11, C, D, H, W]
        fnirs_feature = self.fnirs_bn(fnirs_feature)
        fnirs_feature = F.elu(fnirs_feature)
        fnirs_feature = self.dropout(fnirs_feature)
        fnirs_feature = fnirs_feature.permute(0, 2, 3, 4, 1)  # Back to [B*11, D, H, W, C]
        # 恢复原始batch和11的维度
        fnirs_feature = fnirs_feature.reshape(original_shape[0], original_shape[1], 
                                            *fnirs_feature.shape[1:])
        
        # Fusion processing [B, 1, 16, 16, 600]
        eegfusion_feature = self.eegfusion_conv(eegfusion.permute(0, 4, 1, 2, 3))
        eegfusion_feature = self.eegfusion_bn(eegfusion_feature)
        eegfusion_feature = F.elu(eegfusion_feature)
        eegfusion_feature = self.dropout(eegfusion_feature)
        eegfusion_feature = eegfusion_feature.permute(0, 2, 3, 4, 1)
        
        eegfusion_fga = self.fga((eegfusion_feature, eeg_feature, fnirs_feature))
        
        return eegfusion_fga, eeg_feature, fnirs_feature
        
# class ConvBlock(nn.Module):
#     def __init__(self, eeg_filter: int, eeg_size: Tuple[int, int, int], eeg_stride: Tuple[int, int, int],
#                 fnirs_filter: int, fnirs_size: Tuple[int, int, int], fnirs_stride: Tuple[int, int, int],
#                 eegfusion_filter: int, eegfusion_size: Tuple[int, int, int], eegfusion_stride: Tuple[int, int, int],
#                 tem_kernel_size: int, fga_loss_name: str, padding: str):
#         super(ConvBlock, self).__init__()
        
#         # EEG pathway
#         self.eeg_conv = nn.Conv3d(1, eeg_filter, kernel_size=eeg_size, stride=eeg_stride, padding=padding)
#         self.eeg_bn = nn.BatchNorm3d(eeg_filter)
        
#         # fNIRS pathway
#         self.fnirs_conv = nn.Conv3d(2, fnirs_filter, kernel_size=fnirs_size, stride=fnirs_stride, padding=padding)
#         self.fnirs_bn = nn.BatchNorm3d(fnirs_filter)
        
#         # Fusion pathway
#         self.eegfusion_conv = nn.Conv3d(1, eegfusion_filter, kernel_size=eegfusion_size, stride=eegfusion_stride, padding=padding)
#         self.eegfusion_bn = nn.BatchNorm3d(eegfusion_filter)
        
#         self.fga = FGA(tem_kernel_size, fga_loss_name)
#         self.dropout = nn.Dropout(0.5)
        
#     def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
#         eegfusion, eeg, fnirs = inputs
        
#         # EEG processing
#         eeg_feature = self.eeg_conv(eeg.permute(0, 4, 1, 2, 3))  # [B, C, D, H, W]
#         eeg_feature = self.eeg_bn(eeg_feature)
#         eeg_feature = F.elu(eeg_feature)
#         eeg_feature = self.dropout(eeg_feature)
#         eeg_feature = eeg_feature.permute(0, 2, 3, 4, 1)  # Back to [B, D, H, W, C]
        
#         # fNIRS processing
#         fnirs_feature = self.fnirs_conv(fnirs.permute(0, 5, 1, 2, 3, 4))  # [B, C, D, H, W, T]
#         fnirs_feature = self.fnirs_bn(fnirs_feature)
#         fnirs_feature = F.elu(fnirs_feature)
#         fnirs_feature = self.dropout(fnirs_feature)
#         fnirs_feature = fnirs_feature.permute(0, 2, 3, 4, 5, 1)  # Back to [B, D, H, W, T, C]
        
#         # Fusion processing
#         eegfusion_feature = self.eegfusion_conv(eegfusion.permute(0, 4, 1, 2, 3))
#         eegfusion_feature = self.eegfusion_bn(eegfusion_feature)
#         eegfusion_feature = F.elu(eegfusion_feature)
#         eegfusion_feature = self.dropout(eegfusion_feature)
#         eegfusion_feature = eegfusion_feature.permute(0, 2, 3, 4, 1)
        
#         eegfusion_fga = self.fga((eegfusion_feature, eeg_feature, fnirs_feature))
        
#         return eegfusion_fga, eeg_feature, fnirs_feature

class PredictionWeightLayer(nn.Module):
    def __init__(self):
        super(PredictionWeightLayer, self).__init__()
        self.p_weight = nn.Parameter(torch.zeros(2, 1))
        
    def forward(self, inputs: Tensor) -> Tensor:
        p_weights_softmax = F.softmax(self.p_weight, dim=0)
        p_weights_softmax = p_weights_softmax.unsqueeze(0)
        the_prediction = inputs * p_weights_softmax
        the_prediction = torch.sum(the_prediction, dim=1)
        return the_prediction

class STA_Net(nn.Module):
    def __init__(self):
        super(STA_Net, self).__init__()
        
        # First conv block
        self.conv_block1 = ConvBlock(
            eeg_filter=16, eeg_size=(2, 2, 13), eeg_stride=(2, 2, 6),
            fnirs_filter=16, fnirs_size=(2, 2, 5), fnirs_stride=(2, 2, 2),
            eegfusion_filter=16, eegfusion_size=(2, 2, 13), eegfusion_stride=(2, 2, 6),
            tem_kernel_size=5, fga_loss_name='fgsa1_plcc', padding='same'
        )
        
        # Second conv block
        self.conv_block2 = ConvBlock(
            eeg_filter=32, eeg_size=(2, 2, 5), eeg_stride=(2, 2, 2),
            fnirs_filter=32, fnirs_size=(2, 2, 3), fnirs_stride=(2, 2, 2),
            eegfusion_filter=32, eegfusion_size=(2, 2, 5), eegfusion_stride=(2, 2, 2),
            tem_kernel_size=3, fga_loss_name='fgsa2_plcc', padding='same'
        )
        
        # Attention module
        self.ef_attention = EFAttention(emb_size=256, d_model=256, heads=8, drop=0.5)
        
        # Fully connected layers
        self.eegfusion_fc1 = nn.Linear(256, 256)
        self.eegfusion_fc2 = nn.Linear(256, 2)
        
        self.fnirs_fc1 = nn.Linear(256, 256)
        self.fnirs_fc2 = nn.Linear(256, 2)
        
        self.eeg_fc1 = nn.Linear(32*8*8*1, 256)  # Adjusted based on eeg2 shape
        self.eeg_fc2 = nn.Linear(256, 2)
        
        # Weight layers
        self.eegfusion_p_weight = nn.Linear(256, 1)
        self.fnirs_p_weight = nn.Linear(256, 1)
        self.pred_weight = PredictionWeightLayer()
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, eeg_input: Tensor, fnirs_input: Tensor) -> Tuple[Tensor, Tensor]:
        # Input shapes:
        # eeg_input: [batch, 16, 16, 600, 1]
        # fnirs_input: [batch, 11, 16, 16, 30, 2]
        
        # First conv block
        eegfusion1, eeg1, fnirs1 = self.conv_block1((eeg_input, eeg_input, fnirs_input))
        eegfusion1 = self.dropout(eegfusion1)
        eeg1 = self.dropout(eeg1)
        fnirs1 = self.dropout(fnirs1)
        
        # Second conv block
        eegfusion2, eeg2, fnirs2 = self.conv_block2((eegfusion1, eeg1, fnirs1))
        eegfusion2 = torch.mean(eegfusion2, dim=-2, keepdim=True)  # GAP
        eeg2 = torch.mean(eeg2, dim=-2, keepdim=True)
        fnirs2 = torch.mean(fnirs2, dim=-2, keepdim=True)
        
        eegfusion2 = self.dropout(eegfusion2)
        eeg2 = self.dropout(eeg2)
        fnirs2 = self.dropout(fnirs2)
        
        # Attention module
        eegfusion_feature, fnirs_feature, _ = self.ef_attention((eegfusion2, fnirs2))
        
        # Prepare features for prediction
        eegfusion_feature_pweight = F.elu(eegfusion_feature)
        fnirs_feature_pweight = F.elu(fnirs_feature)
        
        eegfusion_feature_pweight = self.eegfusion_fc1(eegfusion_feature_pweight)
        eegfusion_feature_pweight = F.elu(eegfusion_feature_pweight)
        
        fnirs_feature_pweight = self.fnirs_fc1(fnirs_feature_pweight)
        fnirs_feature_pweight = F.elu(fnirs_feature_pweight)
        
        eeg_feature = eeg2.flatten(start_dim=1)
        eeg_feature = self.eeg_fc1(eeg_feature)
        eeg_feature = F.elu(eeg_feature)
        
        # Predictions
        # fusion_pred = eegfusion_pred + fnirs_pred
        # eeg_pred = fc(eeg_feature)
        eegfusion_pred = self.eegfusion_fc2(eegfusion_feature_pweight)
        fnirs_pred = self.fnirs_fc2(fnirs_feature_pweight)
        eeg_pred = self.eeg_fc2(eeg_feature)
        
        eeg_pred = F.softmax(eeg_pred, dim=-1)
        eegfusion_pred = F.softmax(eegfusion_pred, dim=-1)
        fnirs_pred = F.softmax(fnirs_pred, dim=-1)
        
        eegfusion_pred = eegfusion_pred.unsqueeze(1)
        fnirs_pred = fnirs_pred.unsqueeze(1)
        
        fusion_pred = torch.cat([eegfusion_pred, fnirs_pred], dim=1)
        
        # Weight calculation
        fnirs_p_weight = self.fnirs_p_weight(fnirs_feature_pweight)
        eegfusion_p_weight = self.eegfusion_p_weight(eegfusion_feature_pweight)
        
        p_weight = torch.cat([eegfusion_p_weight, fnirs_p_weight], dim=-1)
        p_weight = F.softmax(p_weight, dim=-1)
        p_weight = p_weight.unsqueeze(-1)
        
        fusion_pred = fusion_pred * p_weight
        fusion_pred = torch.sum(fusion_pred, dim=1)
        
        return fusion_pred, eeg_pred

def sta_net():
    return STA_Net()