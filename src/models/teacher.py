
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import Any

# =====================================================================
# 1-1. Teacher model(Transformer) definition
# =====================================================================
class PurePyTorchTransformer(nn.Module):
    '''
        Pure PyTorch Transformer for sleep stage classification
        Input shape: [Batch, 7, 3000] (7 segments of 30s each, sampled at 100Hz)
        Output shape: [Batch, 5] (5 sleep stages)
    '''
    def __init__(self, num_classes : float = None , embedding_dim=256, nhead=4, num_layers=2, dropout=0.5):
        super(PurePyTorchTransformer, self).__init__()

        self.input_projection = nn.Linear(3000, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim) # 입력 직후 LayerNorm 추가 (오버플로우 방지)

        self.pos_embedding = nn.Parameter(torch.randn(1, 7, embedding_dim) * 0.01) # Positional Embedding (소량으로 작게 초기화)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 2, # feedforward 차원을 축소하여 스파이크 방지
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x): # x: [Batch, 7, 3000]
        x = self.input_projection(x) # [Batch, 7, 128]
        x = self.input_norm(x)
        x = x + self.pos_embedding

        x = self.transformer_encoder(x) # [Batch, 7, 128]
        x = self.final_norm(x)

        x = x.mean(dim=1) # [Batch, 128] (Global Average Pooling)
        return self.fc(x) # [Batch, 5]

class CNNTransformerTeacher(nn.Module):
    """
    [직관적이고 수렴이 빠른 Single-Stream Hybrid Teacher]
    - 입력: 7개 채널 전체 [Batch, 7, 3000]
    - 1D CNN: 7개 채널의 미세 파형 및 채널간 상호작용 피처 압축 -> [Batch, 128, 187]
    - Transformer: 압축된 187개 시간 토큰의 맥락 관계 인코딩 -> [Batch, 187, 128]
    """
    def __init__(self, in_channels=7, teacher_embedding_dim: int = None, teacher_nhead: int = None, teacher_num_layers: int = None, teacher_dropout: float = None, num_classes: float = None):
        super().__init__()
        
        # 1. 1D CNN Feature Extractor (7개 채널 전체 수용)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=25, stride=2, padding=12),
            nn.GroupNorm(num_groups=1, num_channels=32), nn.ELU(alpha=1.0), # nn.BatchNorm1d(teacher_embedding_dim)
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(num_groups=1, num_channels=64), nn.ELU(alpha=1.0), # nn.BatchNorm1d(teacher_embedding_dim)
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=teacher_embedding_dim, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=1, num_channels=teacher_embedding_dim), nn.ELU(alpha=1.0) # nn.BatchNorm1d(teacher_embedding_dim)
        ) # 출력: [Batch, 128, 187]
        
        # 2. Positional Embedding
        self.seq_len = 187
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, teacher_embedding_dim) * 0.01)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=teacher_embedding_dim, # 128
            nhead=teacher_nhead,            # 8
            dim_feedforward=teacher_embedding_dim * 2,
            dropout=teacher_dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=teacher_num_layers)

        self.final_norm = nn.LayerNorm(teacher_embedding_dim)
        self.fc = nn.Sequential(
            nn.Dropout(teacher_dropout),
            nn.Linear(teacher_embedding_dim, num_classes)
        )

    def forward(self, x): # x shape: [Batch, 7, 3000]
        # 1. CNN 특징 추출 -> [Batch, 128, 187]
        feat = self.feature_extractor(x)
        
        # 2. Transformer 차원 변환 [Batch, 187, 128] 및 Positional Embedding 추가
        feat = feat.transpose(1, 2) + self.pos_embedding
        
        # 3. Transformer 인코딩 & Global Average Pooling
        out = self.transformer_encoder(feat)
        out = self.final_norm(out).mean(dim=1)
        return self.fc(out)

