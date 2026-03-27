"""
CASS — Evaluator (3단계: UMAP 차원 축소 + 실루엣 점수)

두 모드:
  fast=False (기본): UMAP_PARAMS      — 논문 보고용, n_neighbors=150
  fast=True        : UMAP_PARAMS_FAST — 1단계 스크리닝 전용, n_neighbors=30

Full 재평가 시 Silhouette 외에 Boundary_Mean, Camouflage@t 도 함께 계산합니다.
이 두 지표는 UMAP 임베딩이 완성된 후 k-NN 한 번으로 추가 비용 없이 산출됩니다.
"""
import numpy as np
from cuml.manifold import UMAP
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .config import UMAP_PARAMS, UMAP_PARAMS_FAST, RANDOM_SEED, CAMOUFLAGE_THRESHOLDS

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


def compute_boundary_camouflage(
    emb: np.ndarray,
    y: np.ndarray,
    thresholds: list = None,
) -> tuple[float, dict]:
    """
    2D UMAP 임베딩에서 Boundary_Mean과 Camouflage@t 를 계산합니다.
    UMAP 실행 후 동일 임베딩을 재사용하므로 추가 비용 = k-NN 1회.

    Args:
        emb       : (n_samples, 2) UMAP 임베딩
        y         : (n_samples,) 이진 레이블 (0=benign, 1=attack)
        thresholds: Camouflage 임계값 목록 (None → config 기본값)

    Returns:
        boundary_mean : float  — 공격→nearest benign 평균 거리 (↑ 좋음)
        camouflage    : dict   — {t: 비율} (↓ 좋음)
    """
    if thresholds is None:
        thresholds = CAMOUFLAGE_THRESHOLDS

    benign_pts = emb[y == 0]
    attack_pts = emb[y == 1]

    if len(benign_pts) == 0 or len(attack_pts) == 0:
        return np.nan, {t: np.nan for t in thresholds}

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(benign_pts)
    nn_dists = nn.kneighbors(attack_pts)[0].flatten()

    boundary_mean = round(float(np.mean(nn_dists)), 4)
    camouflage    = {t: round(float((nn_dists <= t).mean()), 4) for t in thresholds}
    return boundary_mean, camouflage


def evaluate_subset(
    X_scaled: np.ndarray,
    y: np.ndarray,
    all_feature_names: list,
    selected_features: list,
    fast: bool = False,
) -> tuple[float, np.ndarray]:
    """
    전체 피처 행렬에서 선택된 피처 컬럼만 추출하여 실루엣 점수를 반환합니다.
    (Fast 스크리닝 전용 — Silhouette만 필요한 경우)
    """
    feat_list = list(all_feature_names)
    idx = [feat_list.index(f) for f in selected_features]
    X_sub = X_scaled[:, idx]
    return compute_silhouette(X_sub, y, fast=fast)


def evaluate_subset_full_metrics(
    X_scaled: np.ndarray,
    y: np.ndarray,
    all_feature_names: list,
    selected_features: list,
) -> tuple[float, float, dict, np.ndarray]:
    """
    Full UMAP 실행 후 Silhouette + Boundary_Mean + Camouflage@t 를 한 번에 반환합니다.
    2단계 Full 재평가 및 Reference 기준값 계산에 사용됩니다.

    Returns:
        sil           : float        — Silhouette Score
        boundary_mean : float        — 공격→nearest benign 평균 거리
        camouflage    : dict         — {threshold: 비율}
        embedding     : np.ndarray   — (n_samples, 2) UMAP 임베딩
    """
    feat_list = list(all_feature_names)
    idx = [feat_list.index(f) for f in selected_features]
    X_sub = X_scaled[:, idx]

    sil, emb = compute_silhouette(X_sub, y, fast=False)
    bm, cam  = compute_boundary_camouflage(emb, y)
    return sil, bm, cam, emb
