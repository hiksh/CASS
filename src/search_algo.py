"""
CASS — Search Algorithm (2단계 스크리닝)

목적함수:
  Boundary_Mean 최대화 — 공격 포인트에서 nearest benign까지의 평균 거리를 최대화.
  공격 트래픽이 benign 영역에 위장(Camouflage)하기 어렵게 만드는 피처 조합을 탐색.

  제약:
  Silhouette > MIN_SILHOUETTE — 최소한의 클래스 분리도를 보장하여
  공격 클러스터가 지나치게 파편화되는 것을 방지.

전체 흐름:
  [선택] Pilot 검증
    → 무작위 서브셋 PILOT_N개에 대해 Fast/Full Silhouette 상관 계산
    → Spearman r >= PILOT_MIN_SPEARMAN 이면 Fast가 Full의 유효한 proxy임을 확인

  1단계 (Fast 스크리닝)
    → Greedy 또는 Random으로 모든 후보 서브셋을 Fast UMAP으로 평가
    → Boundary_Mean 점수 내림차순 정렬 후 Elbow 검출 → K 결정
    → Silhouette > MIN_SILHOUETTE 제약 적용 (미충족 시 fallback)

  2단계 (Full 재평가)
    → 상위 K개 서브셋만 Full UMAP(논문 파라미터)으로 재평가
    → Silhouette > MIN_SILHOUETTE 제약 하에 Boundary_Mean 최댓값 선택

Elbow 검출 방식:
  정렬된 Fast Boundary_Mean 점수에서 인접 gap이
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
    RANDOM_SEED, MIN_SILHOUETTE,
    UMAP_PARAMS_FAST,   # dict 객체 직접 참조 — in-place 수정으로 evaluator에도 반영
)
from .evaluator import evaluate_subset, evaluate_subset_full_metrics, compute_boundary_camouflage


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
    """Greedy Forward Selection — Fast UMAP으로 모든 스텝 평가.

    목적함수: Boundary_Mean 최대화 (silhouette > MIN_SILHOUETTE 제약).
    제약 미충족 시 silhouette 제약을 완화하여 Boundary_Mean만으로 선택(fallback).
    """
    print(f"  [1단계] Greedy Fast 스크리닝 (후보 {len(candidate_features)}개) ...")
    current = []
    rows = []

    for step in range(1, len(candidate_features) + 1):
        remaining = [f for f in candidate_features if f not in current]
        if not remaining:
            break

        best_bm   = -np.inf
        best_sil  = -2.0
        best_feat = None

        for feat in tqdm(remaining, desc=f"    Step {step:>2}", leave=False):
            trial = current + [feat]
            try:
                sil, emb = evaluate_subset(X_scaled, y, all_feature_names, trial, fast=True)
                if sil <= MIN_SILHOUETTE:
                    continue
                bm, _ = compute_boundary_camouflage(emb, y)
            except Exception:
                continue
            if bm > best_bm:
                best_bm, best_sil, best_feat = bm, sil, feat

        # fallback: silhouette 제약 미충족 시 완화
        if best_feat is None:
            print(f"    Step {step:>2}: [Fallback] silhouette ≤ {MIN_SILHOUETTE} — 제약 완화")
            for feat in remaining:
                trial = current + [feat]
                try:
                    sil, emb = evaluate_subset(X_scaled, y, all_feature_names, trial, fast=True)
                    bm, _ = compute_boundary_camouflage(emb, y)
                except Exception:
                    continue
                if bm > best_bm:
                    best_bm, best_sil, best_feat = bm, sil, feat

        if best_feat is None:
            break

        current.append(best_feat)
        rows.append({
            "step":          step,
            "added_feature": best_feat,
            "n_features":    len(current),
            "features":      list(current),
            "fast_sil":      round(best_sil, 4),
            "fast_bm":       round(best_bm, 4),
        })
        print(f"    Step {step:>2}: +{best_feat:<28} | fast_bm={best_bm:.4f}  fast_sil={best_sil:+.4f}")

    return pd.DataFrame(rows)


def _random_fast(
    X_scaled: np.ndarray,
    y: np.ndarray,
    candidate_features: list,
    all_feature_names: list,
    n_subsets: int,
) -> pd.DataFrame:
    """Random 서브셋 샘플링 — Fast UMAP으로 전체 평가.

    목적함수: Boundary_Mean 최대화. fast_bm 기준으로 Elbow 검출 후 Full 재평가.
    """
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
            sil, emb = evaluate_subset(X_scaled, y, all_feature_names, subset, fast=True)
            bm, _    = compute_boundary_camouflage(emb, y)
            rows.append({
                "subset_id":  i,
                "n_features": len(subset),
                "features":   list(subset),
                "fast_sil":   round(sil, 4),
                "fast_bm":    round(bm, 4),
            })
            tqdm.write(f"    [{i+1:>3}/{len(subsets)}] n={len(subset):>2} | fast_bm={bm:.4f}  fast_sil={sil:+.4f}")
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
    Silhouette 외에 Boundary_Mean, Camouflage@t 도 함께 수집합니다.
    (동일 임베딩 재사용 — UMAP 추가 실행 없음)

    Returns:
        full_df: 상위 K개에 full_sil / boundary_mean / camouflage 컬럼이 채워진 DataFrame
    """
    sort_col    = "fast_bm" if "fast_bm" in fast_df.columns else "fast_sil"
    sorted_df   = fast_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    scores_desc = sorted_df[sort_col].values

    K      = find_elbow(scores_desc)
    top_df = sorted_df.head(K).copy()

    print(f"\n  [Elbow] Fast BM 분포: max={scores_desc[0]:.4f}  min={scores_desc[-1]:.4f}")
    print(f"  [Elbow] K = {K}  (전체 {len(sorted_df)}개 중 상위 {K}개 Full 재평가)")
    print(f"\n  [2단계] Full UMAP 재평가 ({K}개) ...")

    full_sils, boundary_means, camouflages = [], [], []

    for i, row in top_df.iterrows():
        feats = row["features"]
        n     = len(full_sils) + 1
        try:
            sil, bm, cam, _ = evaluate_subset_full_metrics(
                X_scaled, y, all_feature_names, feats
            )
            cam_val = list(cam.values())[0]   # 첫 번째 임계값 (기본 @1.0)
            full_sils.append(round(sil, 4))
            boundary_means.append(round(bm, 4))
            camouflages.append(round(cam_val, 4))
            print(
                f"    [{n:>2}/{K}] {feats} "
                f"| full_sil={sil:+.4f}  bm={bm:.4f}  cam={cam_val:.4f}"
            )
        except Exception as e:
            full_sils.append(float("nan"))
            boundary_means.append(float("nan"))
            camouflages.append(float("nan"))
            print(f"    [{n:>2}/{K}] 오류: {e}")

    top_df["full_sil"]      = full_sils
    top_df["boundary_mean"] = boundary_means
    top_df["camouflage"]    = camouflages
    return top_df


# ── Reference Camouflage 계산 ─────────────────────────────────────────────────

def compute_reference_camouflage(
    X_scaled: np.ndarray,
    y: np.ndarray,
    all_feature_names: list,
    ref_features: list,
) -> float:
    """
    Reference 피처 조합(umar2024 등)에 대해 Full UMAP을 실행하고
    Camouflage@t 값을 반환합니다. 이 값이 제약 기반 선택의 임계값이 됩니다.

    Args:
        ref_features: LITERATURE_BASELINES의 피처 목록

    Returns:
        cam_threshold: float — Camouflage@t (첫 번째 임계값 기준)
    """
    from .config import CAMOUFLAGE_THRESHOLDS
    t = CAMOUFLAGE_THRESHOLDS[0]
    print(
        f"\n[Reference] umar2024 Camouflage@{t} 기준값 계산 중 "
        f"({len(ref_features)}개 피처) ..."
    )
    _, _, cam, emb = evaluate_subset_full_metrics(
        X_scaled, y, all_feature_names, ref_features
    )
    cam_val = cam.get(t, list(cam.values())[0])
    print(f"  → Camouflage@{t} (umar2024) = {cam_val:.4f}  ← 제약 임계값으로 사용")
    return float(cam_val), emb


# ── Pilot 자동 재시도 ─────────────────────────────────────────────────────────

def pilot_validation_with_retry(
    X_scaled: np.ndarray,
    y: np.ndarray,
    candidate_features: list,
    all_feature_names: list,
    n: int = None,
    max_retries: int = 3,
    neighbor_step: int = 30,
) -> tuple[float, pd.DataFrame]:
    """
    Pilot 검증 + r < PILOT_MIN_SPEARMAN 시 n_neighbors 자동 증가 재시도.

    UMAP_PARAMS_FAST['n_neighbors'] 를 in-place로 수정하여
    evaluator.py의 Fast 스크리닝에도 즉시 반영됩니다.

    재시도 전략:
      시도 0: base n_neighbors (config 기본값)
      시도 k: base + k * neighbor_step  (기본 30씩 증가)

    max_retries 초과 시 n_neighbors를 원래 값으로 복원하고
    경고와 함께 파이프라인을 계속 진행합니다.

    Args:
        max_retries  : 재시도 최대 횟수 (기본 3 → 최대 base + 90)
        neighbor_step: 재시도마다 증가할 n_neighbors 폭 (기본 30)

    Returns:
        spearman_r : 최종 Spearman 상관계수
        pilot_df   : 최종 검증 결과 DataFrame
    """
    base_neighbors = UMAP_PARAMS_FAST["n_neighbors"]

    for attempt in range(max_retries + 1):
        if attempt > 0:
            new_n = base_neighbors + attempt * neighbor_step
            UMAP_PARAMS_FAST["n_neighbors"] = new_n
            print(
                f"\n[Pilot 재시도 {attempt}/{max_retries}] "
                f"n_neighbors {base_neighbors} → {new_n}"
            )

        r, pilot_df = pilot_validation(X_scaled, y, candidate_features, all_feature_names, n=n)

        if not np.isnan(r) and r >= PILOT_MIN_SPEARMAN:
            if attempt > 0:
                print(
                    f"  ✓ n_neighbors={UMAP_PARAMS_FAST['n_neighbors']} 에서 통과 "
                    f"(r={r:.4f} ≥ {PILOT_MIN_SPEARMAN})"
                )
                print(
                    f"  이후 Fast 스크리닝은 n_neighbors="
                    f"{UMAP_PARAMS_FAST['n_neighbors']} 로 진행됩니다."
                )
            return r, pilot_df

    # 최대 재시도 초과 → 원복
    UMAP_PARAMS_FAST["n_neighbors"] = base_neighbors
    print(
        f"\n[Pilot] 최대 재시도 {max_retries}회 초과 (최종 r={r:.4f}). "
        f"n_neighbors={base_neighbors} 으로 복원하고 경고와 함께 진행합니다."
    )
    return r, pilot_df


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

    목적함수: Boundary_Mean 최대화 (silhouette > MIN_SILHOUETTE 제약).
    silhouette 제약을 만족하는 후보가 없으면 제약 완화 후 Boundary_Mean 최댓값 선택.

    Returns:
        best_subset: 선택된 최적 피처 리스트
        results_df : 탐색 결과 DataFrame
                     greedy: (step, added_feature, n_features, features,
                              fast_sil, fast_bm, full_sil, boundary_mean, camouflage)
                     random: (subset_id, n_features, features,
                              fast_sil, fast_bm, full_sil, boundary_mean, camouflage)
    """
    if mode is None:
        mode = SEARCH_MODE
    if n_subsets is None:
        n_subsets = N_RANDOM_SUBSETS

    print(f"\n[Search] 2단계 스크리닝 (mode={mode})")
    print(f"  목적함수  : Boundary_Mean 최대화 (silhouette > {MIN_SILHOUETTE} 제약)")

    # 1단계: Fast 스크리닝
    if mode == "greedy":
        fast_df = _greedy_fast(X_scaled, y, candidate_features, all_feature_names)
    elif mode == "random":
        fast_df = _random_fast(X_scaled, y, candidate_features, all_feature_names, n_subsets)
    else:
        raise ValueError(f"알 수 없는 mode: '{mode}'. 'greedy' 또는 'random'을 사용하세요.")

    if fast_df.empty:
        return [], fast_df

    # 2단계: Elbow K → Full 재평가 (full_sil + boundary_mean + camouflage)
    results_df = _full_reeval(fast_df, X_scaled, y, all_feature_names)

    # ── Silhouette > MIN_SILHOUETTE 제약 + Boundary_Mean 최대화 ─────────────
    valid = results_df.dropna(subset=["full_sil", "boundary_mean"])
    if valid.empty:
        return [], results_df

    survived = valid[valid["full_sil"] > MIN_SILHOUETTE]
    if survived.empty:
        print(f"\n  [Constraint] silhouette > {MIN_SILHOUETTE} 만족하는 후보 없음. 제약 완화.")
        survived = valid
    else:
        n_drop = len(valid) - len(survived)
        print(
            f"\n  [Constraint] silhouette > {MIN_SILHOUETTE}: "
            f"{len(survived)}/{len(valid)}개 생존 ({n_drop}개 탈락)"
        )

    best_idx    = survived["boundary_mean"].idxmax()
    best_subset = survived.loc[best_idx, "features"]
    best_bm     = survived.loc[best_idx, "boundary_mean"]
    best_full   = survived.loc[best_idx, "full_sil"]
    print(
        f"  최적 서브셋 (boundary_mean={best_bm:.4f} | full_sil={best_full:.4f}): {best_subset}"
    )

    return best_subset, results_df
