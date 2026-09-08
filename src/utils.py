import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import classification_report


def get_model_size_mb(model: nn.Module) -> float:
    """PyTorch 모델의 파라미터 및 버퍼 용량을 계산하여 MB 단위(float)로 반환합니다."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(size_all_mb, 2)


def save_metrics_to_csv(
    summary_csv_path: str,
    history_csv_path: str,
    cfg_obj: object,
    teacher_size_mb: float,
    student_size_mb: float,
    histories: dict[str, dict[str, list[float]]],
    preds: dict[str, list[int]],
    true_labels: list[int]
) -> None:
    """
    실험 결과(클래스별 리포트 및 에폭별 학습 추이)를 2개의 독립된 CSV 파일로 분리하여 저장합니다.

    Args:
        summary_csv_path (str): 요약 성적표 저장 CSV 경로 (예: sleepedf_trial_08_summary.csv)
        history_csv_path (str): 에폭별 시계열 저장 CSV 경로 (예: sleepedf_trial_08_history.csv)
        cfg_obj (object): Config 객체 또는 json dict
        teacher_size_mb (float): Teacher 모델 용량 (MB)
        student_size_mb (float): Student 모델 용량 (MB)
        histories (dict): 3개 모델(teacher, vanilla, kd)의 epoch별 history 딕셔너리
        preds (dict): 3개 모델의 Validation 예측 라벨 리스트
        true_labels (list[int]): Validation Set의 실제 정답 라벨 리스트
    """
    # 💡 Config 속성 획득 헬퍼 함수
    def _get_cfg(key: str, default=None):
        if isinstance(cfg_obj, dict):
            return cfg_obj.get(key, default)
        return getattr(cfg_obj, key, default)

    classes = _get_cfg("CLASS_NAMES", ["Wake", "N1", "N2", "N3", "REM"])
    trial_num = _get_cfg("TRIAL_NUM", 0)
    alpha = _get_cfg("KD_ALPHA", 0.0)
    temp = _get_cfg("KD_TEMPERATURE", 1.0)
    epochs = _get_cfg("STUDENT_EPOCHS", 0)
    batch_size = _get_cfg("BATCH_SIZE", 0)

    # 1. SUMMARY CSV: 클래스별 분류 성적 (Precision, Recall, Support)
    reports = {
        'teacher': classification_report(true_labels, preds.get('teacher', []), target_names=classes, output_dict=True, zero_division=0),
        'vanilla': classification_report(true_labels, preds.get('vanilla', []), target_names=classes, output_dict=True, zero_division=0),
        'kd': classification_report(true_labels, preds.get('kd', []), target_names=classes, output_dict=True, zero_division=0)
    }

    summary_rows = []
    for cls in classes:
        summary_rows.append({
            "trial_num": trial_num,
            "alpha": alpha,
            "temperature": temp,
            "epochs": epochs,
            "batch_size": batch_size,
            "teacher_size_mb": teacher_size_mb,
            "student_size_mb": student_size_mb,
            "class_name": cls,
            "teacher_precision": reports['teacher'][cls]["precision"],
            "teacher_recall": reports['teacher'][cls]["recall"],
            "teacher_f1": reports['teacher'][cls]["f1-score"],
            "teacher_support": reports['teacher'][cls]["support"],
            "vanilla_precision": reports['vanilla'][cls]["precision"],
            "vanilla_recall": reports['vanilla'][cls]["recall"],
            "vanilla_f1": reports['vanilla'][cls]["f1-score"],
            "vanilla_support": reports['vanilla'][cls]["support"],
            "kd_precision": reports['kd'][cls]["precision"],
            "kd_recall": reports['kd'][cls]["recall"],
            "kd_f1": reports['kd'][cls]["f1-score"],
            "kd_support": reports['kd'][cls]["support"]
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"📊 [저장 완료] Summary CSV ➡️ {summary_csv_path}")

    # 2. HISTORY CSV: Epoch별 Loss / Acc / F1 추이
    t_hist = histories.get('teacher', {})
    v_hist = histories.get('vanilla', {})
    k_hist = histories.get('kd', {})

    # 각 모델별 가장 긴 metric 리스트 길이 탐색
    max_eps = max([
        len(t_hist.get('val_loss', [])), len(v_hist.get('val_loss', [])), len(k_hist.get('val_loss', [])),
        len(t_hist.get('val_acc', [])),  len(v_hist.get('val_acc', [])),  len(k_hist.get('val_acc', [])),
        len(t_hist.get('val_f1', [])),   len(v_hist.get('val_f1', [])),   len(k_hist.get('val_f1', []))
    ] + [0])

    history_rows = []
    for ep_idx in range(max_eps):
        def _get_ep_val(hist_dict, metric_key):
            lst = hist_dict.get(metric_key, [])
            if ep_idx < len(lst) and lst[ep_idx] is not None:
                return float(lst[ep_idx])
            return None # 에폭 범위를 벗어나면 None(CSV상 빈칸)으로 저장

        history_rows.append({
            "trial_num": trial_num,
            "epoch": ep_idx + 1,
            "teacher_val_loss": _get_ep_val(t_hist, 'val_loss'),
            "teacher_val_acc": _get_ep_val(t_hist, 'val_acc'),
            "teacher_val_f1": _get_ep_val(t_hist, 'val_f1'),
            "vanilla_val_loss": _get_ep_val(v_hist, 'val_loss'),
            "vanilla_val_acc": _get_ep_val(v_hist, 'val_acc'),
            "vanilla_val_f1": _get_ep_val(v_hist, 'val_f1'),
            "kd_val_loss": _get_ep_val(k_hist, 'val_loss'),
            "kd_val_acc": _get_ep_val(k_hist, 'val_acc'),
            "kd_val_f1": _get_ep_val(k_hist, 'val_f1')
        })

    df_history = pd.DataFrame(history_rows)
    df_history.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
    print(f"📈 [저장 완료] History CSV ➡️ {history_csv_path}")


def visualize_from_csv(
    summary_csv_path: str,
    history_csv_path: str = None,
    save_fig: bool = True
) -> None:
    """
    분리된 Summary CSV 및 History CSV 데이터를 읽어와 종합 시각화 대시보드를 생성합니다.
    """
    # history_csv_path 미지정 시 자동으로 경로 매칭 추정
    if history_csv_path is None:
        if "summary" in summary_csv_path:
            history_csv_path = summary_csv_path.replace("summary", "history")
        else:
            history_csv_path = summary_csv_path.replace(".csv", "_history.csv")

    if not os.path.exists(summary_csv_path):
        print(f"❌ [오류] Summary CSV 파일이 없습니다: {summary_csv_path}")
        return
    if not os.path.exists(history_csv_path):
        print(f"❌ [오류] History CSV 파일이 없습니다: {history_csv_path}")
        return

    df_summary = pd.read_csv(summary_csv_path)
    df_history = pd.read_csv(history_csv_path)

    # Meta 정보 추출
    first_row = df_summary.iloc[0]
    trial_num = int(first_row["trial_num"])
    alpha = first_row["alpha"]
    temp = first_row["temperature"]
    teacher_mb = float(first_row["teacher_size_mb"])
    student_mb = float(first_row["student_size_mb"])

    def _get_last_valid_value(df, col_name):
        if col_name in df.columns:
            valid_series = df[col_name].dropna()
            if not valid_series.empty:
                return float(valid_series.iloc[-1])
        return 0.0

    # 최종 Epoch Accuracy 추출
    final_teacher_acc = _get_last_valid_value(df_history, "teacher_val_acc")
    final_vanilla_acc = _get_last_valid_value(df_history, "vanilla_val_acc")
    final_kd_acc = _get_last_valid_value(df_history, "kd_val_acc")

    fig, axes = plt.subplots(2, 1, figsize=(7, 11))
    fig.suptitle(
        f"Sleep-EDF Knowledge Distillation Analysis",
        fontsize=15, fontweight="bold"
    )

    # [Graph 1] Class-wise Recall Comparison
    ax1 = axes[0]

    classes = df_summary["class_name"].tolist()
    x_cls = np.arange(len(classes))
    w_cls = 0.25

    teacher_recalls = df_summary["teacher_recall"].fillna(0).values * 100
    vanilla_recalls = df_summary["vanilla_recall"].fillna(0).values * 100
    kd_recalls = df_summary["kd_recall"].fillna(0).values * 100

    ax1.bar(x_cls - w_cls, teacher_recalls, w_cls, label="Teacher Recall", color="#C44E52", alpha=0.8)
    ax1.bar(x_cls, vanilla_recalls, w_cls, label="Vanilla Recall", color="#DD8452")
    ax1.bar(x_cls + w_cls, kd_recalls, w_cls, label="KD Student Recall (Ours)", color="#4C72B0",edgecolor="#1D3557", linewidth=1.8, zorder=3)

    ax1.set_title("1. Class-wise Recall Comparison", fontweight="bold")
    ax1.set_xticks(x_cls)
    ax1.set_xticklabels(classes, rotation=15)
    ax1.set_ylabel("Recall (%)")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # [Graph 2] Epoch-wise Loss Curves
    ax2 = axes[1]
    for m_key, m_label, color, ls, mk, lw, ms, zo in [
        ('teacher', 'Teacher', '#C44E52', '--', '^', 1.2, 3, 2),
        ('vanilla', 'Vanilla', '#DD8452', '-', 'o', 1.2, 3, 3),
        ('kd', 'KD Student (Ours)', '#4C72B0', '-', 's', 2.5, 5, 5)
    ]:
        col = f"{m_key}_val_f1"
        if col in df_history.columns:
            valid_df = df_history[["epoch", col]].dropna()
            if not valid_df.empty:
                ax2.plot(valid_df["epoch"], valid_df[col], label=f"{m_label} F1", color=color, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, zorder=zo, alpha=0.9)

    ax2.set_title("2. F1 Score Curves per Epoch", fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_fig:
        fig_save_path = summary_csv_path.replace(".csv", "_dashboard.png")
        plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ [시각화 완료] 대시보드 저장 ➡️ {fig_save_path}")

    plt.show()