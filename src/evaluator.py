"""
CASS — Evaluator (3단계: UMAP 차원 축소 + 실루엣 점수)

두 모드:
  fast=False (기본): UMAP_PARAMS      — 논문 보고용, n_neighbors=150
  fast=True        : UMAP_PARAMS_FAST — 1단계 스크리닝 전용, n_neighbors=30
"""
import numpy as np
from cuml.manifold import UMAP
from sklearn.metrics import silhouette_score

from .config import UMAP_PARAMS, UMAP_PARAMS_FAST, RANDOM_SEED

# 실루엣 계산은 O(n^2) → 대용량에서 서브샘플 상한 적용
_SILHOUETTE_MAX_SAMPLES = 10_000


def compute_silhouette(
    X_subset: np.ndarray,
    y: np.ndarray,
    fast: bool = False,
) -> tuple[float, np.ndarray]:
    """
    UMAP 투영 후 실루엣 점수를 계산합니다.

    Args:
        X_subset: (n_samples, n_features) — 선택된 피처만 포함
        y       : (n_samples,) 이진 레이블
        fast    : True → UMAP_PARAMS_FAST (스크리닝용)
                  False → UMAP_PARAMS (논문 보고용)

    Returns:
        silhouette: float
        embedding : (n_samples, 2) numpy array
    """
    params    = UMAP_PARAMS_FAST if fast else UMAP_PARAMS
    reducer   = UMAP(**params)
    embedding = np.asarray(reducer.fit_transform(X_subset))  # cupy → numpy

    n = len(embedding)
    if n > _SILHOUETTE_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n, _SILHOUETTE_MAX_SAMPLES, replace=False)
        sil = silhouette_score(embedding[idx], y[idx], metric="euclidean")
    else:
        sil = silhouette_score(embedding, y, metric="euclidean")

    return float(sil), embedding


def evaluate_subset(
    X_scaled: np.ndarray,
    y: np.ndarray,
    all_feature_names: list,
    selected_features: list,
    fast: bool = False,
) -> tuple[float, np.ndarray]:
    """
    전체 피처 행렬에서 선택된 피처 컬럼만 추출하여 실루엣 점수를 반환합니다.
    """
    feat_list = list(all_feature_names)
    idx = [feat_list.index(f) for f in selected_features]
    X_sub = X_scaled[:, idx]
    return compute_silhouette(X_sub, y, fast=fast)
