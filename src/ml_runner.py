"""
CASS — ML Evaluator  (sklearn/XGBoost + PyTorch CNN / LSTM)

exports/ 디렉토리의 비교군 CSV를 읽어
XGBoost · RandomForest · Logistic Regression · CNN · LSTM 5개 모델을
훈련·평가하고 binary F1 score를
results/{dataset}/ml/ 에 저장합니다.

GPU 정책:
  - XGBoost / RandomForest / LogisticReg : sklearn / xgboost (CPU)
  - CNN / LSTM                           : PyTorch CUDA (GPU_MEMORY_FRACTION=0.2)

실행:
  python run_ml.py --dataset unsw_nb15
  python main.py --dataset cicids2018 --export --ml
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ── GPU 설정 ──────────────────────────────────────────────────────────────────

GPU_MEMORY_FRACTION = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_gpu() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        except RuntimeError:
            pass
        print(f"  [GPU] {torch.cuda.get_device_name(0)}  "
              f"메모리 할당: {GPU_MEMORY_FRACTION * 100:.0f}%")
    else:
        print("  [GPU] CUDA 없음 — CPU 사용")


# ── sklearn / XGBoost 모델 ────────────────────────────────────────────────────

def _make_sklearn_models() -> list[tuple[str, object]]:
    """(이름, 모델 인스턴스) 리스트를 반환합니다."""
    return [
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "LogisticReg",
            LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]


# ── PyTorch 모델 정의 ──────────────────────────────────────────────────────────

class _CNN1D(nn.Module):
    """
    1-D CNN NIDS 분류기.
    피처 벡터 (batch, n_features) 를 1-D 시계열 (batch, 1, n_features) 로 처리합니다.
    """

    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1)).squeeze(1)


class _LSTMNet(nn.Module):
    """
    LSTM NIDS 분류기.
    각 피처를 단일 타임스텝으로 취급: (batch, n_features, 1).
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
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(self.drop(out[:, -1, :])).squeeze(1)


# ── PyTorch 훈련 루프 ──────────────────────────────────────────────────────────

_TRAIN_BATCH = 1024   # 훈련 배치 크기
_INFER_BATCH = 4096   # 추론 배치 크기 (데이터를 CPU에 두고 배치씩 GPU로)


def _train_torch(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
) -> nn.Module:
    """
    BCEWithLogitsLoss + Adam + 조기종료(val F1).
    데이터는 CPU에 유지하고 배치 단위로만 GPU로 이동합니다.
    """
    model = model.to(DEVICE)

    # 데이터는 CPU 텐서로 유지
    Xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr)
    Xv = torch.FloatTensor(X_vl)

    loader    = DataLoader(TensorDataset(Xt, yt), batch_size=_TRAIN_BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv), batch_size=_INFER_BATCH)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1, best_state, wait = 0.0, None, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for (xb,) in val_loader:
                p = (torch.sigmoid(model(xb.to(DEVICE))) >= 0.5).cpu().numpy()
                preds.extend(p.astype(int))

        val_f1 = f1_score(y_vl, preds, average="binary", zero_division=0)
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _eval_torch(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """배치 단위 추론. X_test 전체를 GPU에 올리지 않습니다."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test)),
        batch_size=_INFER_BATCH,
    )
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            p = (torch.sigmoid(model(xb.to(DEVICE))) >= 0.5).cpu().numpy()
            preds.extend(p.astype(int))
    return float(f1_score(y_test, preds, average="binary", zero_division=0))


# ── 비교군 단위 평가 ───────────────────────────────────────────────────────────

def _run_group(
    group_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """비교군 하나에 대해 5개 모델 전부 훈련·평가합니다."""
    feat_cols = [c for c in train_df.columns if c not in ("attack_flag", "attack_step")]

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["attack_flag"].values.astype(np.int32)
    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df["attack_flag"].values.astype(np.int32)
    n_feat  = X_train.shape[1]

    scores: dict[str, float] = {}

    # ── sklearn / XGBoost ────────────────────────────────────────────────────
    for display, clf in _make_sklearn_models():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1 = float(f1_score(y_test, y_pred, average="binary",
                                pos_label=1, zero_division=0))
            scores[display] = f1
            print(f"    {display:<15}: F1 = {f1:.4f}")
        except Exception as exc:
            print(f"    {display:<15}: 실패 — {exc}")
            scores[display] = float("nan")

    # ── Train / Val 분리 (90 / 10) ───────────────────────────────────────────
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
        dataset_name : 차트 제목용 데이터셋 이름 (예: "unsw_nb15")

    Returns:
        group × model binary-F1 DataFrame (ml_dir/f1_results.csv 에도 저장)
    """
    _setup_gpu()
    ml_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(exports_dir.glob("train_*.csv"))
    if not train_files:
        raise FileNotFoundError(f"exports 디렉토리에 train_*.csv 없음: {exports_dir}")

    groups = [f.stem[len("train_"):] for f in train_files]

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

        scores = _run_group(group, train_df, test_df)
        scores["group"] = group
        all_rows.append(scores)

    if not all_rows:
        print("  [경고] 평가된 비교군이 없습니다.")
        return pd.DataFrame()

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
