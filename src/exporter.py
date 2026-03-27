"""
CASS — Exporter (비교군별 Train/Test CSV 생성)

비교군:
  cass          : UMAP Silhouette 최적 부분집합 (CASS 결과)
  anova         : ANOVA F-score 상위 N개 (pre-filter 풀 내)
  extratrees    : ExtraTrees 중요도 상위 N개 (pre-filter 풀 내)
  random[_k]    : 무작위 N개 (N_RANDOM_BASELINE회, pre-filter 풀 내)
  lit_<name>    : LITERATURE_BASELINES 수동 정의 피처 조합

N = len(best_features) 로 고정하여 차원 수 혼입을 방지합니다.
Literature 기준은 고정 피처 수를 사용합니다.

출력 위치: EXPORTS_DIR/
  train_<group>.csv  — UDBB 훈련 데이터 (선택된 피처 + 레이블)
  test_<group>.csv   — test-flow.csv 전체 (동일 scaler, 선택된 피처 + 레이블)
"""
import random
import numpy as np
import pandas as pd

from .config import (
    EXPORTS_DIR, TEST_FILE,
    N_RANDOM_BASELINE, LITERATURE_BASELINES,
    RANDOM_SEED,
)
from .data_loader import load_and_sample, preprocess


# ── 비교군 구성 ───────────────────────────────────────────────────────────────

def build_comparison_groups(
    best_features: list,
    filter_summary: pd.DataFrame,
    feature_names: list,
    n_random: int,
) -> dict:
    """
    비교군별 피처 리스트 딕셔너리를 생성합니다.

    Args:
        best_features  : CASS 최적 피처 리스트
        filter_summary : pre_filter 순위표 (Feature, Tree_Importance, ANOVA_F 컬럼 포함)
        feature_names  : 전체 로드된 피처 이름 목록
        n_random       : 랜덤 비교군 반복 횟수

    Returns:
        {group_name: [feature, ...]} 딕셔너리
    """
    N = len(best_features)
    candidate_pool = filter_summary["Feature"].tolist()
    feat_set = set(feature_names)
    groups = {}

    # 1) CASS
    groups["cass"] = list(best_features)

    # 2) ANOVA top-N  (pre-filter 풀 내 재정렬)
    anova_top = (
        filter_summary
        .sort_values("ANOVA_F", ascending=False)
        .head(N)["Feature"]
        .tolist()
    )
    groups["anova"] = anova_top

    # 3) ExtraTrees top-N  (pre-filter 풀 내 재정렬)
    tree_top = (
        filter_summary
        .sort_values("Tree_Importance", ascending=False)
        .head(N)["Feature"]
        .tolist()
    )
    groups["extratrees"] = tree_top

    # 4) Random N  (pre-filter 풀에서 무작위 선택)
    rng = random.Random(RANDOM_SEED + 1000)
    for i in range(n_random):
        key = "random" if n_random == 1 else f"random_{i + 1}"
        pool_size = len(candidate_pool)
        sampled = sorted(rng.sample(candidate_pool, min(N, pool_size)))
        groups[key] = sampled

    # 5) Literature baselines  (config.py LITERATURE_BASELINES)
    for name, feats in LITERATURE_BASELINES.items():
        valid   = [f for f in feats if f in feat_set]
        missing = [f for f in feats if f not in feat_set]
        if missing:
            print(f"    [!] lit_{name}: {len(missing)}개 피처가 데이터셋에 없어 제외됩니다.")
            print(f"        제외: {missing}")
        groups[f"lit_{name}"] = valid

    return groups


# ── 피처 추출 & DataFrame 변환 ────────────────────────────────────────────────

def _extract(
    X_scaled: np.ndarray,
    y: np.ndarray,
    attack_step: np.ndarray,
    feature_names: list,
    selected: list,
) -> pd.DataFrame:
    """선택된 피처 컬럼만 추출하여 레이블과 합친 DataFrame을 반환합니다."""
    feat_list = list(feature_names)
    idx = [feat_list.index(f) for f in selected if f in feat_list]
    actual = [feat_list[i] for i in idx]

    df = pd.DataFrame(X_scaled[:, idx], columns=actual)
    df["attack_flag"] = y
    df["attack_step"] = attack_step
    return df


# ── Test 데이터 로드 ──────────────────────────────────────────────────────────

def _load_test(scaler, feature_names: list):
    """
    test-flow.csv를 로드하고 훈련 데이터와 동일한 scaler로 전처리합니다.
    피처 목록과 순서는 훈련 데이터와 동일하게 유지됩니다.
    """
    print(f"    test-flow.csv 로드 중 ...")
    df_test = load_and_sample(TEST_FILE, use_udbb=False)
    print(f"    원본 test 행 수  : {len(df_test):,}")

    X_test, _, _ = preprocess(
        df_test,
        feature_cols=list(feature_names),
        fit_scaler=False,
        scaler=scaler,
    )
    y_test    = df_test["attack_flag"].astype(int).values
    step_test = df_test["attack_step"].values

    n_benign = int((y_test == 0).sum())
    n_attack = int((y_test == 1).sum())
    print(f"    test 클래스 분포 : benign={n_benign:,}  attack={n_attack:,}")
    return X_test, y_test, step_test


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def export_comparison_sets(
    X_scaled: np.ndarray,
    y: np.ndarray,
    attack_step: np.ndarray,
    feature_names: list,
    best_features: list,
    filter_summary: pd.DataFrame,
    scaler,
    n_random: int = None,
    export_dir=None,
) -> dict:
    """
    비교군별 train/test CSV를 exports/ 디렉토리에 저장합니다.

    Args:
        X_scaled       : 전처리된 훈련 피처 행렬 (UDBB 샘플)
        y              : 훈련 레이블
        attack_step    : 훈련 Kill Chain 레이블
        feature_names  : 로드된 전체 피처 이름 목록
        best_features  : CASS 최적 피처 리스트
        filter_summary : pre_filter 순위표 DataFrame
        scaler         : 훈련 데이터에 fit된 RobustScaler
        n_random       : 랜덤 비교군 횟수 (None → config 기본값)
        export_dir     : 저장 경로 (None → config EXPORTS_DIR)

    Returns:
        groups: {group_name: [feature, ...]} 딕셔너리
    """
    if n_random is None:
        n_random = N_RANDOM_BASELINE
    if export_dir is None:
        export_dir = EXPORTS_DIR

    export_dir.mkdir(parents=True, exist_ok=True)

    N = len(best_features)
    SEP = "=" * 65

    print(f"\n{SEP}")
    print(f"[Export] 비교군 구성 및 CSV 저장")
    print(f"  CASS 최적 피처 수 N = {N}  |  랜덤 비교군 {n_random}회")
    print(SEP)

    # ── 비교군 구성 ──────────────────────────────────────────────────────────
    print("\n  [1/3] 비교군 피처 리스트 생성 ...")
    groups = build_comparison_groups(best_features, filter_summary, feature_names, n_random)

    # 비교군 요약 출력
    print(f"\n  {'그룹':<20} {'피처 수':>6}  {'상위 4개 피처 (미리보기)'}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*35}")
    for name, feats in groups.items():
        if not feats:
            print(f"  {name:<20} {'없음':>6}")
            continue
        preview = ", ".join(feats[:4]) + ("  ..." if len(feats) > 4 else "")
        print(f"  {name:<20} {len(feats):>6}  {preview}")

    # ── Test 데이터 로드 ─────────────────────────────────────────────────────
    print(f"\n  [2/3] Test 데이터 로드 ...")
    X_test, y_test, step_test = _load_test(scaler, feature_names)

    # ── 그룹별 CSV 저장 ──────────────────────────────────────────────────────
    n_train = len(X_scaled)
    n_test  = len(X_test)
    print(f"\n  [3/3] CSV 저장 중  (train={n_train:,}행  test={n_test:,}행) ...")
    print(f"  {'그룹':<20}  {'저장 파일'}")
    print(f"  {'-'*20}  {'-'*40}")

    saved_count = 0
    for name, feats in groups.items():
        if not feats:
            print(f"  {name:<20}  [건너뜀] 유효한 피처 없음")
            continue

        train_df = _extract(X_scaled, y, attack_step, feature_names, feats)
        test_df  = _extract(X_test, y_test, step_test, feature_names, feats)

        train_path = export_dir / f"train_{name}.csv"
        test_path  = export_dir / f"test_{name}.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        saved_count += 1
        print(f"  {name:<20}  train_{name}.csv  /  test_{name}.csv")

    print(f"\n  {saved_count}개 비교군 저장 완료")
    print(f"  저장 위치: {export_dir}")
    print(SEP)

    return groups
