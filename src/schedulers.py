import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 3,
    start_factor: float = 0.01,
    min_lr: float = 1e-6
) -> SequentialLR:
    """
        처음 warmup_epochs 동안 LR을 Linear하게 올린 후, 
        남은 epochs 동안 Cosine Annealing을 적용하는 SequentialLR을 생성합니다.
    """
    # 1. 처음 3 Epoch: Warmup (0.1배 -> 1.0배)
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=start_factor, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # 2. 나머지 Epoch: Cosine Annealing
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=max(1, epochs - warmup_epochs), 
        eta_min=min_lr
    )
    
    # 3. 두 스케줄러 연결 (SequentialLR)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler

