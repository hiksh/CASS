"""
CASS — ML Evaluator  (AutoGluon TabularPredictor + PyTorch CNN / LSTM)

exports/ 디렉토리의 비교군 CSV를 읽어
XGBoost · RandomForest · Logistic Regression · CNN · LSTM 5개 모델을
훈련·평가하고 binary F1 score를
results/{dataset}/ml/ 에 저장합니다.

GPU 정책:
  - XGBoost / RandomForest / LogisticReg : AutoGluon CPU (num_gpus=0)
  - CNN / LSTM                           : PyTorch CUDA (GPU_MEMORY_FRACTION=0.2)

실행:
  python main.py --dataset cicids2018 --export --ml
  python main.py --dataset unsw_nb15  --export --ml
  python main.py --dataset cicids2018 --ml          # export가 이미 존재하는 경우
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ── GPU 설정 ──────────────────────────────────────────────────────────────────

GPU_MEMORY_FRACTION = 0.2   # A100 0.2 할당 비율
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_gpu() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        except RuntimeError:
            pass  # 이미 다른 컨텍스트에서 설정된 경우 무시
        print(f"  [GPU] {torch.cuda.get_device_name(0)}  "
              f"메모리 할당: {GPU_MEMORY_FRACTION * 100:.0f}%")
    else:
        print("  [GPU] CUDA 없음 — CPU 사용")


# ── AutoGluon 모델 정의 ────────────────────────────────────────────────────────

# (출력 이름, AutoGluon 모델 키)
AG_MODELS = [
    ("XGBoost",      "XGB"),
    ("RandomForest", "RF"),
    ("LogisticReg",  "LR"),
]


def _fit_ag_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ag_key: str,
    save_dir: Path,
) -> float:
    """
    AutoGluon TabularPredictor로 단일 모델 타입을 훈련·평가합니다.

    - num_bag_folds=0, num_stack_levels=0 → 단일 모델 (앙상블 없음)
    - num_gpus=0 → CPU 전용 (0.2 GPU 할당을 CNN/LSTM에 온전히 양보)

    Returns:
        binary F1 score (float)
    """
    from autogluon.tabular import TabularPredictor

    save_dir.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label="attack_flag",
        eval_metric="f1",
        path=str(save_dir),
        verbosity=0,
    )
    predictor.fit(
        train_data=train_df,
        hyperparameters={ag_key: {}},
        num_bag_folds=0,
        num_stack_levels=0,
        num_gpus=0,
        verbosity=0,
    )

    y_true = test_df["attack_flag"].values
    y_pred = predictor.predict(test_df).values
    return float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))


# ── PyTorch 모델 정의 ──────────────────────────────────────────────────────────

class _CNN1D(nn.Module):
    """
    1-D CNN NIDS 분류기.
    피처 벡터 (batch, n_features) 를 1-D 시계열 (batch, 1, n_features) 로 처리합니다.
    """

    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),   # (B, 64, F)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # (B, 128, F)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                        # (B, 128, 1)
            nn.Flatten(),                                   # (B, 128)
            nn.Dropout(dropout),
            nn.Linear(128, 1),                             # (B, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → unsqueeze → (B, 1, F)
        return self.net(x.unsqueeze(1)).squeeze(1)


class _LSTMNet(nn.Module):
    """
    LSTM NIDS 분류기.
    각 피처를 단일 타임스텝으로 취급: (batch, n_features, 1).
    마지막 타임스텝의 hidden state → Linear → logit.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → (B, F, 1)
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(self.drop(out[:, -1, :])).squeeze(1)


# ── PyTorch 훈련 루프 ──────────────────────────────────────────────────────────

def _train_torch(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 5,
) -> nn.Module:
    """
    BCEWithLogitsLoss + Adam + 조기종료(val F1 기준, patience=5).
    최고 val-F1 가중치를 복원하여 반환합니다.
    """
    model = model.to(DEVICE)

    Xt = torch.FloatTensor(X_tr).to(DEVICE)
    yt = torch.FloatTensor(y_tr).to(DEVICE)
    Xv = torch.FloatTensor(X_vl).to(DEVICE)

    loader    = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1, best_state, wait = 0.0, None, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(Xv)) >= 0.5).cpu().numpy().astype(int)

        val_f1 = f1_score(y_vl, preds, average="binary", zero_division=0)
        if val_f1 > best_f1:
            best_f1  = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _eval_torch(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(DEVICE))
        preds  = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
    return float(f1_score(y_test, preds, average="binary", zero_division=0))


# ── 비교군 단위 평가 ───────────────────────────────────────────────────────────

def _run_group(
    group_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ag_base_dir: Path,
) -> dict[str, float]:
    """
    비교군 하나에 대해 5개 모델 전부 훈련·평가합니다.

    Args:
        group_name  : 비교군 이름 (예: "cass", "anova")
        train_df    : train_*.csv DataFrame (피처 + attack_flag + attack_step)
        test_df     : test_*.csv  DataFrame (동일 구조)
        ag_base_dir : AutoGluon 모델 저장 루트

    Returns:
        {모델이름: binary_f1} 딕셔너리
    """
    feat_cols = [c for c in train_df.columns if c not in ("attack_flag", "attack_step")]

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["attack_flag"].values.astype(np.int32)
    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df["attack_flag"].values.astype(np.int32)
    n_feat  = X_train.shape[1]

    # AutoGluon에 넘길 DataFrame: attack_step 제거, attack_flag 유지
    ag_train = train_df.drop(columns=["attack_step"], errors="ignore")
    ag_test  = test_df.drop(columns=["attack_step"],  errors="ignore")

    scores: dict[str, float] = {}

    # ── AutoGluon (XGBoost / RandomForest / LogisticReg) ─────────────────────
    for display, ag_key in AG_MODELS:
        try:
            f1 = _fit_ag_model(
                ag_train, ag_test, ag_key,
                ag_base_dir / f"{group_name}_{ag_key}",
            )
            scores[display] = f1
            print(f"    {display:<15}: F1 = {f1:.4f}")
        except Exception as exc:
            print(f"    {display:<15}: 실패 — {exc}")
            scores[display] = float("nan")

    # ── Train / Val 분리 (90 / 10) ─────────────────────────────────────────
    rng   = np.random.RandomState(42)
    idx   = rng.permutation(len(X_train))
    n_val = max(int(len(X_train) * 0.1), 1)
    X_tr, y_tr = X_train[idx[n_val:]], y_train[idx[n_val:]]
    X_vl, y_vl = X_train[idx[:n_val]], y_train[idx[:n_val]]

    # ── CNN ───────────────────────────────────────────────────────────────────
    try:
        cnn = _train_torch(_CNN1D(n_feat), X_tr, y_tr, X_vl, y_vl)
        f1  = _eval_torch(cnn, X_test, y_test)
        scores["CNN"] = f1
        print(f"    {'CNN':<15}: F1 = {f1:.4f}")
        del cnn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"    {'CNN':<15}: 실패 — {exc}")
        scores["CNN"] = float("nan")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    try:
        lstm = _train_torch(_LSTMNet(n_feat), X_tr, y_tr, X_vl, y_vl)
        f1   = _eval_torch(lstm, X_test, y_test)
        scores["LSTM"] = f1
        print(f"    {'LSTM':<15}: F1 = {f1:.4f}")
        del lstm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"    {'LSTM':<15}: 실패 — {exc}")
        scores["LSTM"] = float("nan")

    return scores


# ── 결과 시각화 ────────────────────────────────────────────────────────────────

MODEL_ORDER = ["XGBoost", "RandomForest", "LogisticReg", "CNN", "LSTM"]


def _plot_heatmap(
    results_df: pd.DataFrame,
    save_path: Path,
    dataset_name: str,
) -> None:
    """그룹 × 모델 F1 히트맵."""
    n_g, n_m = results_df.shape
    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.6), max(3, n_g * 0.8)))

    sns.heatmap(
        results_df.astype(float),
        annot=True, fmt=".4f",
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Binary F1"},
    )
    title = "ML Evaluation — Binary F1"
    if dataset_name:
        title += f"  [{dataset_name.upper()}]"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model",         fontsize=11)
    ax.set_ylabel("Feature Group", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  히트맵 저장: {save_path}")


def _plot_bar(
    results_df: pd.DataFrame,
    save_path: Path,
    dataset_name: str,
) -> None:
    """그룹별 모델 F1 점수 그룹 막대 차트."""
    n_groups = len(results_df)
    n_models = len(results_df.columns)
    bar_w    = 0.8 / n_models
    x        = np.arange(n_groups)
    cmap     = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 5))

    for i, model in enumerate(results_df.columns):
        offset = (i - n_models / 2 + 0.5) * bar_w
        vals   = results_df[model].values.astype(float)
        bars   = ax.bar(
            x + offset, vals, width=bar_w,
            label=model, color=cmap(i), alpha=0.85, edgecolor="white",
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.004,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Binary F1", fontsize=11)
    title = "ML Evaluation — Binary F1 per Feature Group"
    if dataset_name:
        title += f"  [{dataset_name.upper()}]"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  막대 차트 저장: {save_path}")


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def run_ml_evaluation(
    exports_dir: Path,
    ml_dir: Path,
    dataset_name: str = "",
) -> pd.DataFrame:
    """
    exports_dir 안의 모든 비교군 CSV를 읽어 ML 평가 파이프라인을 실행합니다.

    Args:
        exports_dir  : train_*.csv / test_*.csv 가 저장된 경로
        ml_dir       : 결과 저장 경로 (results/{dataset}/ml/)
        dataset_name : 차트 제목용 데이터셋 이름 (예: "cicids2018")

    Returns:
        group × model binary-F1 DataFrame (ml_dir/f1_results.csv 에도 저장)
    """
    _setup_gpu()
    ml_dir.mkdir(parents=True, exist_ok=True)
    ag_cache = ml_dir / "ag_models"

    train_files = sorted(exports_dir.glob("train_*.csv"))
    if not train_files:
        raise FileNotFoundError(f"exports 디렉토리에 train_*.csv 없음: {exports_dir}")

    groups = [f.stem[len("train_"):] for f in train_files]
    print(f"  비교군: {groups}\n")

    all_rows: list[dict] = []

    for group in groups:
        train_path = exports_dir / f"train_{group}.csv"
        test_path  = exports_dir / f"test_{group}.csv"

        if not test_path.exists():
            print(f"  [경고] test_{group}.csv 없음 — 건너뜀")
            continue

        print(f"  ── {group} ──")
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        scores = _run_group(group, train_df, test_df, ag_cache)
        scores["group"] = group
        all_rows.append(scores)

    if not all_rows:
        print("  [경고] 평가된 비교군이 없습니다.")
        return pd.DataFrame()

    # 열 순서: MODEL_ORDER 기준 정렬
    available_cols = [m for m in MODEL_ORDER if m in all_rows[0]]
    results_df = pd.DataFrame(all_rows).set_index("group")[available_cols]

    # CSV 저장
    csv_path = ml_dir / "f1_results.csv"
    results_df.to_csv(csv_path)
    print(f"\n  F1 결과 CSV : {csv_path}")

    # 시각화
    _plot_heatmap(results_df, ml_dir / "f1_heatmap.png", dataset_name)
    _plot_bar(    results_df, ml_dir / "f1_bar.png",     dataset_name)

    # 콘솔 요약
    print(f"\n  {'그룹':<22}" + "".join(f"  {m:>13}" for m in results_df.columns))
    print(f"  {'-'*22}" + "".join(["  " + "-" * 13] * len(results_df.columns)))
    for group, row in results_df.iterrows():
        vals = "".join(
            f"  {v:>13.4f}" if not np.isnan(v) else f"  {'N/A':>13}"
            for v in row
        )
        print(f"  {group:<22}{vals}")

    return results_df
