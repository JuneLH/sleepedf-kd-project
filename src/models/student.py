
import os
import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# Student model(CNN) definition for 2ch sleep EDF Data
# =====================================================================

class SleepStudentCNN(nn.Module):
    '''
        Student CNN model for sleep stage classification
        Input shape: [Batch, 2, 3000] (2 channels of 30s each, sampled at 100Hz)
        Output shape: [Batch, 5] (5 sleep stages)
    '''

    def __init__(self, in_channels=2, num_classes: int = 5):
        super(SleepStudentCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            # CNN layer 1: 대형 커널(kernel_size=50), 거시적 파형 포착
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=50, stride=2, padding=24), # [Batch, 2, 3000] -> [Batch, 32, 1500]
            nn.GroupNorm(num_groups=1, num_channels=32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=2, stride=2), # [Batch, 32, 1500] -> [Batch, 32, 750]

            # CNN layer 2: 중형 커널(kernel_size=15), 미세 파형(Spindle, K-Complex 피크) 포착
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, stride=2, padding=14, dilation=2), # [Batch, 32, 750] -> [Batch, 64, 375]
            nn.GroupNorm(num_groups=1, num_channels=64),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=2, stride=2), # [Batch, 64, 375] -> [Batch, 64, 187]

            # CNN layer 3: 소형 커널(kernel_size=5), 미시적 파형 포착
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2), # [Batch, 64, 187] -> [Batch, 128, 187]
            nn.GroupNorm(num_groups=1, num_channels=128), 
            nn.ELU(alpha=1.0),

            nn.AdaptiveAvgPool1d(16) # [Batch, 128, 187] -> [Batch, 128, 16]
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 128), # 128*16 [Batch, 2048] feature -> [Batch, 128] hidden layer
            nn.ELU(alpha=1.0),
            nn.Dropout(0.2),          # 과적합 방지를 위해 무작위 dropout
            nn.Linear(128, num_classes) # 최종 수면 단계(5개) 분류 로짓 mapping. [Batch, 5]
        )

        self._init_weights()  # 가중치 초기화

    def forward(self, x): # 입력 x : [batch_size, 2, 3000]
        x = self.feature_extractor(x)  # 1. 1D CNN 특징 추출 -> 차원 결과: [batch_size, 128, 16]
        x = x.view(x.size(0), -1)      # 2. 1차원 벡터로 평탄화(Flatten) -> 차원 결과: [batch_size, 2048]
        x = self.classifier(x)         # 3. 전결합층 통과 -> 최종 분류 로짓 반환: [batch_size, 5]
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)