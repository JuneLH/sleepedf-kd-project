import sys
import os
from dotenv import load_dotenv
load_dotenv(override=True) # read local .env file

# LIB_PATH가 환경변수로 등록되어 있지 않다면 sys.path에 추가
LIB_PATH = os.getenv("YOUR_LIB_PATH")
if LIB_PATH in sys.path:
    sys.path.remove(LIB_PATH)
sys.path.insert(0, LIB_PATH)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Class Weights (클래스 불균형 보정 가중치) 자동 계산
# =====================================================================

def compute_class_weights(
    train_labels, 
    num_classes=5, 
    device='cpu', 
    method='sqrt', 
    max_clip=10.0, 
    eps=1e-5
):
    """
    Train Dataset 내 클래스별 샘플 수를 기반으로 손실 가중치(Class Weights)를 계산합니다.

    Args:
        train_labels: Training labels tensor
        num_classes (int): # of sleep stages (기본값: 5)
        device (str): 텐서 할당 디바이스 ('cpu' 또는 'cuda')
        method (str): 'sqrt' (제곱근 완화 가중치, 권장) 또는 'inverse' (단순 역수 가중치)
        eps (float): epsilon to prevent division-by-zero (기본값: 1e-5)
    """
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.from_numpy(train_labels)

    # 클래스별 빈도수 고속 계산
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    print(f"Train Sleep Stage Dataset 클래스별 빈도수: {class_counts.long().tolist()}")

    total_samples = class_counts.sum()

    if method == 'inverse':
        # 1-1. Pure Inverse Weight 계산
        class_weights = 1.0 / (class_counts + eps)
    elif method == 'sqrt':
        # 1-2. Square Root Inverse Weight 계산
        class_weights = 1.0 / (torch.sqrt(class_counts) + eps)
    else:
        raise ValueError(f"지원하지 않는 method입니다: {method}. 'inverse' 또는 'sqrt'를 사용하세요.")

    # 2. 다수 클래스(최대 샘플) 기준 정규화 (가장 흔한 클래스의 가중치를 1.0으로 고정)
    class_weights = class_weights / class_weights.min()

    # 3. Upper Bound Clipping (상한선 제한)
    if max_clip is not None:
        class_weights = torch.clamp(class_weights, min=1.0, max=max_clip)

    class_weights = class_weights.to(device)

    print(f"계산된 Class Weights ({method.upper()} 가중치): {class_weights.round(decimals=3).tolist()}")
    return class_weights

# ==============================================================================
# 2. DISTILLATION LOSS FUNCTION
# ==============================================================================
class KDLoss(nn.Module):
    """Teacher/Vanilla/KD 통합 손실 함수."""
    def __init__(
            self, 
            alpha: float = 0.35, 
            temperature: float = 3.0,
            ce_weight: torch.Tensor | None = None,  # Class Weights 인자
            kl_weight: torch.Tensor | None = None
        
        ) -> None:
            super().__init__()
            self.alpha: float = alpha
            self.temperature: float = temperature
            self.register_buffer("ce_weight", ce_weight) # Class Weights Tensor
            self.register_buffer("kl_weight", kl_weight) # KL Divergence Weights Tensor
            # CrossEntropyLoss에 weight 텐서를 전달 (None이면 일반 CE 동작)
            self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=0.02)
            self.kl_div = nn.KLDivLoss(reduction="none")
            
    def forward(
        self, 
        student_logits: torch.Tensor, 
        targets: torch.Tensor, 
        teacher_logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        loss_ce = self.ce_loss(student_logits, targets)
        
        # Teacher가 없거나 Alpha가 0인 경우 -> 일반 CrossEntropy 동작 (Teacher/Vanilla 전용)
        if teacher_logits is None or self.alpha == 0.0:
            return loss_ce
        
        # Knowledge Distillation 동작 (KD Student 전용)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)

        # [Batch, Classes] 형태의 개별 손실 산출
        kl_raw = self.kl_div(soft_prob, soft_targets) # (B, C)
        kl_per_sample = kl_raw.sum(dim=1) # 클래스 축 합산 -> (B,)

        # Class Weight가 지정되어 있다면 배치 내 정답 클래스 기반 가중치 부여
        if self.kl_weight is not None:
            kl_weight = self.kl_weight[targets] # (B,)
            # 가중치 평균 계산 및 Temperature^2 스케일링
            loss_kd = (kl_per_sample * kl_weight).mean() * (self.temperature ** 2)
        else:
            loss_kd = kl_per_sample.mean() * (self.temperature ** 2)

        # ③ 최종 결합 손실
        return (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd