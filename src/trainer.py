import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import Any
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from tqdm.notebook import tqdm
from config import Config

import logging
logger = logging.getLogger(__name__)

from tqdm.notebook import tqdm

# ==============================================================================
# 3. UNIFIED TRAINER (Teacher, Vanilla, KD 모두 적용)
# ==============================================================================
class UnifiedTrainer:
    """Teacher, Vanilla Student, KD Student 모델 학습을 모두 담당하는 통합 트레이너."""
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        train_criterion: nn.Module | None = None,
        eval_criterion: nn.Module | None = None,  # Evaluation 전용 Loss (미지정 시 train_criterion 사용)
        teacher_model: nn.Module | None = None,
        scheduler: Any | None = None
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.train_criterion: nn.Module = train_criterion
        # eval_criterion이 지정되지 않은 경우 기본 train_criterion 사용 (Vanilla/Teacher용)
        self.eval_criterion: nn.Module = eval_criterion if eval_criterion is not None else train_criterion
        self.device: torch.device = device
        self.teacher_model: nn.Module | None = teacher_model
        self.scheduler: Any | None = scheduler
        
        if self.teacher_model is not None:
            self.teacher_model.to(device)
            self.teacher_model.eval()

        self.history: dict[str, list[float]] = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [], 
            'val_f1': [], 'lr': []
        }

    def _prepare_inputs(self, signals: torch.Tensor, is_student: bool = False) -> torch.Tensor:
        """Student 모델일 경우 지정된 2채널만 슬라이싱, Teacher일 경우 전체 채널 사용."""
        if is_student and signals.size(1) > Config.STUDENT_IN_CHANNELS:
            return signals[:, :Config.STUDENT_IN_CHANNELS, :]
        return signals

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int, is_student: bool = False) -> tuple[float, float]:
        self.model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        skipped_batches = 0
        total_batches = len(train_loader)

        pbar = tqdm(
            train_loader, 
            mininterval=1.0,
            desc=f"Epoch [{epoch+1:02d}/{total_epochs:02d}] Train", 
            leave=False,
            dynamic_ncols=True
        )

        for signals, targets in pbar:
            signals, targets = signals.to(self.device), targets.to(self.device)
            
            # Data Augmentation (Teacher/Student 동시 적용을 위해 inputs 생성 전 signals에 수행)
            if self.model.training:
                noise = torch.randn_like(signals) * (signals.std(dim=-1, keepdim=True) * 0.01)
                scale = torch.empty(signals.size(0), 1, 1, device=self.device).uniform_(0.95, 1.05)
                shift_steps = torch.randint(-10, 11, (1,)).item()
                
                signals = torch.roll((signals + noise) * scale, shifts=shift_steps, dims=-1)

            inputs = self._prepare_inputs(signals, is_student=is_student)

            self.optimizer.zero_grad()

            # Teacher Logit 추출 (KD Mode일 때만 실행)
            teacher_logits: torch.Tensor | None = None
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = self.teacher_model(signals)  # Teacher는 Full Channels 사용

            logits = self.model(inputs)
            
            # 훈련 손실 함수 계산 (KD Loss 또는 기본 CE Loss)
            if self.teacher_model is not None:
                loss = self.train_criterion(logits, targets, teacher_logits)
            else:
                loss = self.train_criterion(logits, targets)
            
            # NaN Loss 예외 처리
            if torch.isnan(loss):
                logger.error("❌ NaN Loss 감지! 해당 배치의 기울기 업데이트를 스킵합니다.")
                self.optimizer.zero_grad()
                skipped_batches += 1
                continue

            loss.backward()

            # Gradient Clipping 및 Extreme Gradient Spike 방지
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning("⚠️ NaN/Inf Gradient 감지! 해당 배치 업데이트를 스킵합니다.")
                self.optimizer.zero_grad()
                skipped_batches += 1
            else:
                self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            current_loss = running_loss / max(total, 1)
            current_acc = 100.0 * correct / max(total, 1)
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})

        avg_loss = running_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        logger.info(f"📊 Training Summary: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, Skipped Batches={skipped_batches}/{total_batches}")
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, is_student: bool = False) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        all_preds, all_targets = [], []

        pbar = tqdm(val_loader, mininterval=1.0, desc="    Validating", leave=False, dynamic_ncols=True)

        for signals, targets in pbar:
            signals, targets = signals.to(self.device), targets.to(self.device)
            inputs = self._prepare_inputs(signals, is_student=is_student)

            logits = self.model(inputs)

            if logits.abs().max() > 20.0:
                tqdm.write(f"🚨 [Logit Explode Alert] Min: {logits.min().item():.2f}, Max: {logits.max().item():.2f}")
            
            # 검증 과정에서는 오직 정답 라벨 기준의 eval_criterion 사용
            loss = self.eval_criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            current_loss = running_loss / max(total, 1)
            pbar.set_postfix({'val_loss': f"{current_loss:.4f}"})

        avg_loss = running_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        return avg_loss, accuracy, macro_f1, np.array(all_targets), np.array(all_preds)

    def fit(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int, 
        model_name: str, 
        is_student: bool = False,
        save_dir: str = "."
    ) -> None:
        logger.info(f"🚀 Starting Training [{model_name}] for {epochs} Epochs...")
        best_f1: float = 0.0
        best_path: str = ""  # UnboundLocalError 방지를 위한 초기화

        epoch_pbar = tqdm(range(epochs), desc=f"Overall [{model_name}]", unit="epoch", dynamic_ncols=True)

        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch(train_loader, epoch, epochs, is_student=is_student)
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader, is_student=is_student)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            epoch_pbar.set_postfix({
                'T_Loss': f"{train_loss:.4f}",
                'V_Loss': f"{val_loss:.4f}",
                'V_Acc': f"{val_acc:.2f}%",
                'V_F1': f"{val_f1:.4f}"
            })

            # Best Model 체크포인트 저장
            if val_f1 > best_f1:
                best_f1 = val_f1

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"best_{model_name}.pth")
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_f1': val_f1,
                        'history': self.history
                    }
                    torch.save(checkpoint, save_path)
                    logger.info(f"💾 Best Model Saved -> {save_path} (Val F1: {val_f1:.4f})")
                    best_path = save_path

        # 저장된 가중치 안전 로드
        if best_path and os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ [{model_name}] Best Model (Epoch {checkpoint['epoch']}, Val F1: {checkpoint['val_f1']:.4f}) 가중치 자동 복원 완료!")