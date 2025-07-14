import torch.nn as nn
from data_loader import dataloader_generator
import torch
import torch.optim as optim
import os
import logging
import time

class EMGMLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 128]):
        super(EMGMLPEncoder, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 除了最后一层，都加激活函数和归一化
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(dims[i + 1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(-1, C)  # 展平为 [batch_size * num_points, num_channels]
        x = self.mlp(x)    # 编码为 [batch_size * num_points, output_dim]
        x = x.view(B, N, -1)  # 恢复为 [batch_size, num_points, output_dim]
       
        return x

# 定义 LSTM 模型



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig_mlp = EMGMLPEncoder(input_dim=6, output_dim=64, hidden_dims=[64, 128])
        self.ref_mlp = EMGMLPEncoder(input_dim=51, output_dim=64, hidden_dims=[64, 128])

    def forward(self, x, ref):
        x = self.sig_mlp(x)
        ref = self.ref_mlp(ref.view(ref.size(0),ref.size(1), -1))

        x = torch.cat([x, ref], dim=-1)
        # print (x.shape)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 只取最后时间步的输出
        return output



class LSTMModel_IMU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel_IMU, self).__init__()
        self.lstm = nn.LSTM(192, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig_mlp = EMGMLPEncoder(input_dim=6, output_dim=64, hidden_dims=[64, 128])
        self.ref_mlp = EMGMLPEncoder(input_dim=51, output_dim=64, hidden_dims=[64, 128])
        self.imu_mlp = EMGMLPEncoder(input_dim=7, output_dim=64, hidden_dims=[64, 128])

    def forward(self, x, ref,imu):
        x = self.sig_mlp(x)
        ref = self.ref_mlp(ref.view(ref.size(0),ref.size(1), -1))
        # print(imu.shape)
        imu = self.imu_mlp(imu)
        x = torch.cat([x, ref,imu], dim=-1)
        # print (x.shape)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 只取最后时间步的输出
        return output


class SignalEncoder(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(SignalEncoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.Dropout(0.1),
            nn.Conv1d(16, output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.Dropout(0.1),
            # nn.AdaptiveAvgPool1d(1)  # 将输出压缩为固定维度
        )

    def forward(self, x):
        x = self.cnn(x)  # CNN提取特征
        # x = x.view(x.size(0), -1)  # Flatten
        # x = self.fc(x)  # 输出编码结果
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.bn2(self.conv2(x))
        x = self.dropout2(x)
        return F.relu(x + residual)
    
    
class NeuroPose(nn.Module):
    def __init__(self):
        super(NeuroPose, self).__init__()

        # Encoder for signal x (shape: [bs, 200, 21])
        self.encoder_ref = SignalEncoder(51, 32)
        self.encoder_x = SignalEncoder(6, 32)
        self.encoder_imu = SignalEncoder(7, 32)

        # Encoder for signal ref (shape: [bs, 200, 6])
        # self.encoder_x = nn.Sequential(
        #     nn.Conv1d(6, 32, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
            
        #     nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        # )

        # Encoder for signal imu (shape: [bs, 200, 7])
        # self.encoder_imu = nn.Sequential(
        #     nn.Conv1d(7, 32, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
            
        #     nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        # )
        
        # ResNet
        self.resnet = nn.Sequential(
            ResidualBlock(96),
            ResidualBlock(96),
            # ResidualBlock(192),
            # ResidualBlock(192),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Final regression layer to predict 21 outputs
        self.regressor = nn.Linear(32, 21)
        
    def forward(self, x, ref, imu):
        # Encode each signal separately
        x_encoded = self.encoder_x(x.permute(0, 2, 1))  # Change shape to [bs, 6, 200]
        ref_encoded = self.encoder_ref(ref.view(ref.size(0),ref.size(1), -1).permute(0, 2, 1))  # Change shape to [bs, 21, 200]
        imu_encoded = self.encoder_imu(imu.permute(0, 2, 1))  # Change shape to [bs, 7, 200]
        
        # print(x_encoded.shape)
        # print(ref_encoded.shape)
        # print(imu_encoded.shape)
        # Concatenate the encoded signals along the channel dimension
        x = torch.cat([x_encoded, ref_encoded, imu_encoded], dim=1)  # Shape: [bs, 256, time_steps]
        
        # Pass through ResNet and Decoder
        x = self.resnet(x)
        x = self.decoder(x)
        x = x.squeeze(-1)    # Remove the last dimension: [bs, 32]
        x = self.regressor(x)  # Output shape: [bs, 21]
        
        return x