"""
CASS — Search Algorithm (2단계 스크리닝)

전체 흐름:
  [선택] Pilot 검증
    → 무작위 서브셋 PILOT_N개에 대해 Fast/Full Silhouette 모두 계산
    → Spearman r >= PILOT_MIN_SPEARMAN 이면 Fast가 Full의 유효한 proxy임을 확인

  1단계 (Fast 스크리닝)
    → Greedy 또는 Random으로 모든 후보 서브셋을 Fast UMAP으로 평가
    → Silhouette 점수 내림차순 정렬 후 Elbow 검출 → K 결정

  2단계 (Full 재평가)
    → 상위 K개 서브셋만 Full UMAP(논문 파라미터)으로 재평가
    → 최종 Silhouette 점수 기록

Elbow 검출 방식:
  정렬된 Fast Silhouette 점수에서 인접 gap이
  max_gap * ELBOW_GAP_RATIO 이하로 떨어지는 첫 지점을 Elbow로 판정.
  ELBOW_MIN_K 이상을 항상 보장.
"""
import random
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from .config import (
    SEARCH_MODE, N_RANDOM_SUBSETS, MIN_SUBSET_SIZE, MAX_SUBSET_SIZE,
    ELBOW_GAP_RATIO, ELBOW_MIN_K,
    PILOT_N, PILOT_MIN_SPEARMAN,
    RANDOM_SEED,
)
from .evaluator import evaluate_subset


# ── Elbow 검출 ────────────────────────────────────────────────────────────────

def find_elbow(scores_desc: np.ndarray, gap_ratio: float = None, min_k: int = None) -> int:
    """
    내림차순 정렬된 Silhouette 점수에서 Elbow K를 결정합니다.

    인접 점수 간 gap이 max_gap * gap_ratio 이하로 처음 떨어지는 지점을 Elbow로 봅니다.

    Args:
        scores_desc: 내림차순 정렬된 Silhouette 점수 배열
        gap_ratio  : Elbow 판정 임계 비율 (None → config 기본값)
        min_k      : 최소 K (None → config 기본값)

    Returns:
        K (int): 2단계에서 재평가할 서브셋 수
    """
    if gap_ratio is None:
        gap_ratio = ELBOW_GAP_RATIO
    if min_k is None:
        min_k = ELBOW_MIN_K

    n = len(scores_desc)
    if n <= min_k:
        return n

    gaps = np.abs(np.diff(scores_desc))   # 항상 양수 (내림차순)
    max_gap = gaps.max()

    if max_gap == 0:
        return min_k  # 모든 점수가 동일

    threshold = gap_ratio * max_gap
    for i, g in enumerate(gaps):
        if g < threshold:
            return max(min_k, i + 1)

    return n  # elbow 미검출 → 전부 사용


# ── Pilot 상관관계 검증 ───────────────────────────────────────────────────────

def pilot_validation(
    X_scaled: np.ndarray,
    y: np.ndarray,
    candidate_features: list,
    all_feature_names: list,
    n: int = None,
) -> tuple[float, pd.DataFrame]:
    """
    Fast Silhouette ↔ Full Silhouette의 Spearman 상관을 검증합니다.

    무작위 서브셋 n개에 대해 두 UMAP 파라미터로 모두 실행하여
    Fast가 Full의 유효한 proxy인지 수치로 확인합니다.

    Returns:
        spearman_r : Spearman 상관계수
        pilot_df   : 검증 결과 DataFrame (subset, fast_sil, full_sil)
    """
    if n is None:
        n = PILOT_N

    print(f"\n[Pilot] Fast ↔ Full Silhouette 상관 검증 ({n}개 서브셋) ...")

    rng_py = random.Random(RANDOM_SEED + 99)
    subsets = []
    seen = set()
    for _ in range(n * 10):
        k = rng_py.randint(MIN_SUBSET_SIZE, min(MAX_SUBSET_SIZE, len(candidate_features)))
        subset = tuple(sorted(rng_py.sample(candidate_features, k)))
        if subset not in seen:
            seen.add(subset)
            subsets.append(list(subset))
        if len(subsets) >= n:
            break

    rows = []
    for i, subset in enumerate(tqdm(subsets, desc="  Pilot 평가")):
        try:
            fast_sil, _ = evaluate_subset(X_scaled, y, all_feature_names, subset, fast=True)
            full_sil, _ = evaluate_subset(X_scaled, y, all_feature_names, subset, fast=False)
            rows.append({
                "subset_id": i,
                "n_features": len(subset),
                "features":   list(subset),
                "fast_sil":   round(fast_sil, 4),
                "full_sil":   round(full_sil, 4),
            })
            tqdm.write(
                f"  [{i+1:>2}/{n}] n={len(subset):>2}"
                f" | fast={fast_sil:+.4f}  full={full_sil:+.4f}"
            )
        except Exception as e:
            tqdm.write(f"  [{i+1:>2}] 오류: {e}")

    pilot_df = pd.DataFrame(rows)

    if len(pilot_df) < 3:
        print("  [경고] 유효한 결과가 너무 적습니다.")
        return float("nan"), pilot_df

    r, p = spearmanr(pilot_df["fast_sil"], pilot_df["full_sil"])
    print(f"\n  Spearman r = {r:.4f}  (p={p:.4f})")

    if r >= PILOT_MIN_SPEARMAN:
        print(f"  ✓ Fast가 Full의 유효한 proxy입니다 (r >= {PILOT_MIN_SPEARMAN})")
    else:
        print(f"  ✗ 상관이 낮습니다 (r < {PILOT_MIN_SPEARMAN}). "
              f"UMAP_PARAMS_FAST 조정을 고려하세요.")

    return float(r), pilot_df


# ── 1단계: Fast 스크리닝 ──────────────────────────────────────────────────────

def _greedy_fast(
    X_scaled: np.ndarray,
    y: np.ndarray,
    candidate_features: list,
    all_feature_names: list,
) -> pd.DataFrame:
    """Greedy Forward Selection — Fast UMAP으로 모든 스텝 평가."""
    print(f"  [1단계] Greedy Fast 스크리닝 (후보 {len(candidate_features)}개) ...")
    current = []
    rows = []

    for step in range(1, len(candidate_features) + 1):
        remaining = [f for f in candidate_features if f not in current]
        if not remaining:
            break

        best_sil  = -2.0
        best_feat = None

        for feat in tqdm(remaining, desc=f"    Step {step:>2}", leave=False):
            trial = current + [feat]
            try:
                sil, _ = evaluate_subset(X_scaled, y, all_feature_names, trial, fast=True)
            except Exception:
                sil = -2.0
            if sil > best_sil:
                best_sil, best_feat = sil, feat

        if best_feat is None:
            break

        current.append(best_feat)
        rows.append({
            "step":          step,
            "added_feature": best_feat,
            "n_features":    len(current),
            "features":      list(current),
            "fast_sil":      round(best_sil, 4),
        })
        print(f"    Step {step:>2}: +{best_feat:<28} | fast_sil={best_sil:+.4f}")

    return pd.DataFrame(rows)


def _random_fast(
    X_scaled: np.ndarray,
    y: np.ndarray,
    candidate_features: list,
    all_feature_names: list,
    n_subsets: int,
) -> pd.DataFrame:
    """Random 서브셋 샘플링 — Fast UMAP으로 전체 평가."""
    print(f"  [1단계] Random Fast 스크리닝 ({n_subsets}개) ...")
    rng_py = random.Random(RANDOM_SEED)

    subsets, seen = [], set()
    for _ in range(n_subsets * 10):
        k = rng_py.randint(MIN_SUBSET_SIZE, min(MAX_SUBSET_SIZE, len(candidate_features)))
        subset = tuple(sorted(rng_py.sample(candidate_features, k)))
        if subset not in seen:
            seen.add(subset)
            subsets.append(list(subset))
        if len(subsets) >= n_subsets:
            break

    rows = []
    for i, subset in enumerate(tqdm(subsets, desc="    Fast 평가")):
        try:
            sil, _ = evaluate_subset(X_scaled, y, all_feature_names, subset, fast=True)
            rows.append({
                "subset_id":  i,
                "n_features": len(subset),
                "features":   list(subset),
                "fast_sil":   round(sil, 4),
            })
            tqdm.write(f"    [{i+1:>3}/{len(subsets)}] n={len(subset):>2} | fast_sil={sil:+.4f}")
        except Exception as e:
            tqdm.write(f"    [{i+1:>3}] 오류: {e}")

    return pd.DataFrame(rows)


# ── 2단계: Elbow → Full 재평가 ───────────────────────────────────────────────

def _full_reeval(
    fast_df: pd.DataFrame,
    X_scaled: np.ndarray,
    y: np.ndarray,
    all_feature_names: list,
) -> pd.DataFrame:
    """
    Fast 결과에서 Elbow K를 결정하고 상위 K개를 Full UMAP으로 재평가합니다.

    Returns:
        full_df: 상위 K개의 full_sil이 채워진 DataFrame
    """
    # 내림차순 정렬
    sorted_df = fast_df.sort_values("fast_sil", ascending=False).reset_index(drop=True)
    scores_desc = sorted_df["fast_sil"].values

    K = find_elbow(scores_desc)
    top_df = sorted_df.head(K).copy()

    print(f"\n  [Elbow] Fast Sil 분포: max={scores_desc[0]:.4f}  min={scores_desc[-1]:.4f}")
    print(f"  [Elbow] K = {K}  (전체 {len(sorted_df)}개 중 상위 {K}개 Full 재평가)")
    print(f"\n  [2단계] Full UMAP 재평가 ({K}개) ...")

    full_sils = []
    for i, row in top_df.iterrows():
        feats = row["features"]
        try:
            sil, _ = evaluate_subset(X_scaled, y, all_feature_names, feats, fast=False)
            full_sils.append(round(sil, 4))
            print(f"    [{len(full_sils):>2}/{K}] {feats} | full_sil={sil:+.4f}")
        except Exception as e:
            full_sils.append(float("nan"))
            print(f"    [{len(full_sils):>2}/{K}] 오류: {e}")

    top_df["full_sil"] = full_sils
    return top_df


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def search(
    X_scaled: np.ndarray,
    y: np.ndarray,
    candidate_features: list,
    all_feature_names: list,
    mode: str = None,
    n_subsets: int = None,
) -> tuple[list, pd.DataFrame]:
    """
    2단계 스크리닝 탐색 파이프라인.

    Returns:
        best_subset: Full Silhouette 최고 피처 리스트
        results_df : 최종 결과 DataFrame
                     greedy: (step, added_feature, n_features, features, fast_sil, full_sil)
                     random: (subset_id, n_features, features, fast_sil, full_sil)
    """
    if mode is None:
        mode = SEARCH_MODE
    if n_subsets is None:
        n_subsets = N_RANDOM_SUBSETS

    print(f"\n[Search] 2단계 스크리닝 (mode={mode})")

    # 1단계: Fast 스크리닝
    if mode == "greedy":
        fast_df = _greedy_fast(X_scaled, y, candidate_features, all_feature_names)
    elif mode == "random":
        fast_df = _random_fast(X_scaled, y, candidate_features, all_feature_names, n_subsets)
    else:
        raise ValueError(f"알 수 없는 mode: '{mode}'. 'greedy' 또는 'random'을 사용하세요.")

    if fast_df.empty:
        return [], fast_df

    # 2단계: Elbow K → Full 재평가
    results_df = _full_reeval(fast_df, X_scaled, y, all_feature_names)

    # 최적 서브셋 (full_sil 기준)
    valid = results_df.dropna(subset=["full_sil"])
    if valid.empty:
        return [], results_df

    best_subset = valid.loc[valid["full_sil"].idxmax(), "features"]
    best_full   = valid["full_sil"].max()
    print(f"\n  최적 서브셋 (full_sil={best_full:.4f}): {best_subset}")

    return best_subset, results_df
