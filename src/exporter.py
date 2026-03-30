"""
CASS — Exporter (비교군별 Train/Test CSV 생성)

비교군:
  cass          : UMAP Boundary_Mean 최적 부분집합 (CASS 결과)
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
    literature_baselines: dict = None,
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
    feat_set = set(feature_names)
    groups = {}

    # 1) CASS
    groups["cass"] = list(best_features)

    # 2) ANOVA top-N  (전체 피처 랭킹에서 선택)
    anova_top = (
        filter_summary
        .sort_values("ANOVA_F", ascending=False)
        .head(N)["Feature"]
        .tolist()
    )
    groups["anova"] = anova_top

    # 3) ExtraTrees top-N  (전체 피처 랭킹에서 선택)
    tree_top = (
        filter_summary
        .sort_values("Tree_Importance", ascending=False)
        .head(N)["Feature"]
        .tolist()
    )
    groups["extratrees"] = tree_top

    # 4) Random N  (전체 피처에서 무작위 선택)
    rng = random.Random(RANDOM_SEED + 1000)
    all_features_list = list(feature_names)
    for i in range(n_random):
        key = "random" if n_random == 1 else f"random_{i + 1}"
        sampled = sorted(rng.sample(all_features_list, min(N, len(all_features_list))))
        groups[key] = sampled

    # 5) Literature baselines
    if literature_baselines is None:
        literature_baselines = LITERATURE_BASELINES
    for name, feats in literature_baselines.items():
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

def load_test_data(scaler, feature_names: list, clip_params: dict,
                   test_file=None, log_features: list = None):
    """
    test-flow.csv를 로드하고 훈련 데이터와 동일한 scaler·clip_params로 전처리합니다.
    피처 목록과 순서는 훈련 데이터와 동일하게 유지됩니다.

    Args:
        scaler       : 훈련 데이터에 fit된 RobustScaler
        clip_params  : 훈련 기준 {col: (lower, upper)} clip 경계
        test_file    : 테스트 CSV 경로 (None → config TEST_FILE)
        log_features : log1p 변환 대상 피처 목록 (None → CICIDS2018 LOG_FEATURES)
    """
    if test_file is None:
        test_file = TEST_FILE
    print(f"    {test_file.name} 로드 중 ...")
    df_test = load_and_sample(test_file, use_udbb=False, usecols=list(feature_names))
    print(f"    원본 test 행 수  : {len(df_test):,}")

    X_test, _, _, _ = preprocess(
        df_test,
        feature_cols=list(feature_names),
        fit_scaler=False,
        scaler=scaler,
        log_features=log_features,
        clip_params=clip_params,
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
    clip_params: dict,
    n_random: int = None,
    export_dir=None,
    test_file=None,
    literature_baselines: dict = None,
    log_features: list = None,
) -> dict:
    """
    비교군별 train/test CSV를 exports/ 디렉토리에 저장합니다.

    Args:
        X_scaled              : 전처리된 훈련 피처 행렬 (UDBB 샘플)
        y                     : 훈련 레이블
        attack_step           : 훈련 Kill Chain 레이블
        feature_names         : 로드된 전체 피처 이름 목록
        best_features         : CASS 최적 피처 리스트
        filter_summary        : pre_filter 순위표 DataFrame
        scaler                : 훈련 데이터에 fit된 RobustScaler
        clip_params           : 훈련 기준 {col: (lower, upper)} clip 경계
        n_random              : 랜덤 비교군 횟수 (None → config 기본값)
        export_dir            : 저장 경로 (None → config EXPORTS_DIR)
        literature_baselines  : 데이터셋별 literature baseline dict
                                (None → config LITERATURE_BASELINES)
        log_features          : log1p 변환 대상 피처 목록 (None → CICIDS2018 LOG_FEATURES)

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
    groups = build_comparison_groups(
        best_features, filter_summary, feature_names, n_random,
        literature_baselines=literature_baselines,
    )

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
    X_test, y_test, step_test = load_test_data(
        scaler, feature_names, clip_params,
        test_file=test_file, log_features=log_features,
    )

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


# ── UMAP 임베딩 Export ────────────────────────────────────────────────────────

def export_umap_embeddings(
    X_scaled: np.ndarray,
    y: np.ndarray,
    attack_step: np.ndarray,
    feature_names: list,
    best_features: list,
    scaler,
    clip_params: dict,
    export_dir=None,
    test_file=None,
    log_features: list = None,
    n_components_list: list = None,
) -> None:
    """
    CASS best subset 피처로 UMAP 임베딩을 생성하고 train/test CSV로 저장합니다.

    훈련: 기존 UDBB 샘플(X_scaled)에 UMAP fit_transform — 동일 샘플 구성 유지.
    테스트: test-flow.csv 전체를 동일 UMAP으로 transform.

    저장 파일 (export_dir/):
      train_umap2d.csv / test_umap2d.csv
      train_umap3d.csv / test_umap3d.csv  (n_components_list 기본값 기준)

    Args:
        n_components_list: UMAP 차원 수 목록 (기본 [2, 3])
    """
    from cuml.manifold import UMAP as cumlUMAP
    from .config import UMAP_PARAMS

    if n_components_list is None:
        n_components_list = [2, 3]
    if export_dir is None:
        export_dir = EXPORTS_DIR

    export_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("[UMAP Embedding Export] best subset → UMAP 임베딩 CSV 생성")
    print(f"  best subset 피처 수 : {len(best_features)}")
    print(f"  n_components        : {n_components_list}")
    print(f"  훈련 샘플 수        : {len(X_scaled):,}  (기존 UDBB 구성 그대로)")
    print(SEP)

    # best subset 피처 인덱스
    feat_list = list(feature_names)
    idx       = [feat_list.index(f) for f in best_features if f in feat_list]
    X_tr_sub  = X_scaled[:, idx].astype(np.float32)

    # test 데이터 로드 (기존 exporter와 동일한 scaler/clip)
    print("\n  test 데이터 로드 중 ...")
    X_test, y_test, step_test = load_test_data(
        scaler, feature_names, clip_params,
        test_file=test_file,
        log_features=log_features,
    )
    X_te_sub = X_test[:, idx].astype(np.float32)

    for nc in n_components_list:
        group_name = f"umap{nc}d"
        print(f"\n  ── {group_name} (n_components={nc}) ──")

        params  = {**UMAP_PARAMS, "n_components": nc}
        reducer = cumlUMAP(**params)

        print(f"    UMAP fit_transform (train {len(X_tr_sub):,}행) ...")
        emb_train = np.asarray(reducer.fit_transform(X_tr_sub))

        print(f"    UMAP transform     (test  {len(X_te_sub):,}행) ...")
        emb_test  = np.asarray(reducer.transform(X_te_sub))

        col_names = [f"umap_{i + 1}" for i in range(nc)]

        tr_df = pd.DataFrame(emb_train, columns=col_names)
        tr_df["attack_flag"] = y
        tr_df["attack_step"] = attack_step

        te_df = pd.DataFrame(emb_test, columns=col_names)
        te_df["attack_flag"] = y_test
        te_df["attack_step"] = step_test

        tr_path = export_dir / f"train_{group_name}.csv"
        te_path = export_dir / f"test_{group_name}.csv"
        tr_df.to_csv(tr_path, index=False)
        te_df.to_csv(te_path, index=False)
        print(f"    저장: train_{group_name}.csv  /  test_{group_name}.csv")

    print(f"\n  저장 위치: {export_dir}")
    print(SEP)
