"""
CASS — Data Loader & Preprocessor
NetFlowGap의 전처리 파이프라인을 모듈화합니다.

파이프라인:
  1. training-flow.csv 로드
  2. UDBB 샘플링 (Benign:Attack = 3:1, 공격 단계 균등화)
  3. Inf/NaN → median 대체
  4. Percentile clipping (skewed 피처: 0~0.99, 나머지: 0.01~0.99)
  5. log1p 변환 (skewed 피처)
  6. RobustScaler 정규화
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import (
    TRAIN_FILE, TEST_FILE, ALL_FEATURES, LOG_FEATURES,
    UDBB_COUNTS, PROCESSED_DIR, RANDOM_SEED,
    UNSW_LOG_FEATURES,
)


def load_and_sample(csv_path=None, use_udbb=True, udbb_counts=None) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 UDBB 샘플링을 적용합니다.

    Args:
        csv_path   : CSV 경로 (None이면 config의 TRAIN_FILE 사용)
        use_udbb   : True → UDBB 샘플링, False → 전체 데이터 반환
        udbb_counts: 샘플 수 딕셔너리 (None이면 config의 UDBB_COUNTS 사용)

    Returns:
        샘플링된 DataFrame
    """
    if csv_path is None:
        csv_path = TRAIN_FILE
    if udbb_counts is None:
        udbb_counts = UDBB_COUNTS

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  원본 행 수: {len(df):,}")

    # attack_flag, attack_step 타입 정리
    df["attack_flag"] = pd.to_numeric(df["attack_flag"], errors="coerce").fillna(0).astype(int)
    df["attack_step"] = df["attack_step"].fillna("benign").astype(str).str.strip().str.lower()

    if not use_udbb:
        return df

    print("\n[UDBB] 샘플링 중 ...")
    samples = []

    # Benign
    benign_pool = df[df["attack_flag"] == 0]
    n_benign = min(udbb_counts["benign"], len(benign_pool))
    samples.append(benign_pool.sample(n=n_benign, random_state=RANDOM_SEED))
    print(f"  benign       : {n_benign:,}")

    # 공격 단계별
    for step, n_target in udbb_counts.items():
        if step == "benign":
            continue
        pool = df[(df["attack_flag"] == 1) & (df["attack_step"] == step)]
        n_sample = min(n_target, len(pool))
        if n_sample > 0:
            samples.append(pool.sample(n=n_sample, random_state=RANDOM_SEED))
            print(f"  {step:<12} : {n_sample:,}")
        else:
            print(f"  {step:<12} : 0  (데이터 없음)")

    df_sampled = (
        pd.concat(samples)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    print(f"\n  샘플링 완료: {len(df_sampled):,} rows")
    return df_sampled


def preprocess(
    df: pd.DataFrame,
    feature_cols: list = None,
    fit_scaler: bool = True,
    scaler=None,
    log_features: list = None,
    clip_params: dict = None,
):
    """
    피처 전처리 (NetFlowGap 파이프라인 기반).

    Args:
        df: 입력 DataFrame
        feature_cols: 사용할 피처 목록 (None → ALL_FEATURES 중 존재하는 것)
        fit_scaler: True → 새 scaler fit, False → 전달된 scaler로 transform
        scaler: fit_scaler=False일 때 사용할 사전 fit된 scaler
        log_features: log1p 변환 대상 피처 목록 (None → CICIDS2018 LOG_FEATURES)
        clip_params: {col: (lower, upper)} — fit_scaler=False 시 훈련 기준 clip 경계.
                     None이면 입력 데이터 기준으로 계산 (fit_scaler=True 전용).

    Returns:
        X_scaled (ndarray), feature_names (list), scaler, clip_params (dict)
    """
    if feature_cols is None:
        feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    if log_features is None:
        log_features = LOG_FEATURES

    log_feats = [f for f in log_features if f in feature_cols]
    non_log   = [f for f in feature_cols if f not in log_feats]

    X = df[feature_cols].copy()

    # Inf/NaN 처리
    X = X.replace({np.inf: np.nan, -np.inf: np.nan, -1: np.nan})
    X = X.fillna(X.median())

    # Percentile clipping
    if fit_scaler:
        # 훈련 데이터 기준으로 clip 경계 계산 및 저장
        clip_params = {}
        for col in log_feats:
            upper = float(X[col].quantile(0.99))
            clip_params[col] = (0.0, upper)
            X[col] = X[col].clip(lower=0, upper=upper)
        for col in non_log:
            lower = float(X[col].quantile(0.01))
            upper = float(X[col].quantile(0.99))
            clip_params[col] = (lower, upper)
            X[col] = X[col].clip(lower=lower, upper=upper)
    else:
        # 훈련 기준 clip 경계 적용 (test 데이터 통계 사용 금지)
        if clip_params is None:
            raise ValueError("fit_scaler=False일 때 clip_params를 반드시 전달해야 합니다.")
        for col in log_feats:
            lo, hi = clip_params[col]
            X[col] = X[col].clip(lower=lo, upper=hi)
        for col in non_log:
            lo, hi = clip_params[col]
            X[col] = X[col].clip(lower=lo, upper=hi)

    # log1p 변환
    X[log_feats] = np.log1p(X[log_feats].clip(lower=0))

    # RobustScaler
    if fit_scaler:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("fit_scaler=False일 때 scaler를 반드시 전달해야 합니다.")
        X_scaled = scaler.transform(X)

    return X_scaled, list(feature_cols), scaler, clip_params


def load_dataset(
    csv_path=None,
    use_udbb: bool = True,
    save_processed: bool = False,
    all_features: list = None,
    udbb_counts: dict = None,
    log_features: list = None,
):
    """
    전체 로드 + 샘플링 + 전처리 파이프라인.

    Args:
        csv_path      : CSV 경로 (None → config TRAIN_FILE)
        use_udbb      : True → UDBB 샘플링
        save_processed: True → 전처리 결과 저장
        all_features  : 사용할 피처 목록 (None → config ALL_FEATURES)
        udbb_counts   : UDBB 샘플 수 (None → config UDBB_COUNTS)
        log_features  : log1p 변환 대상 피처 목록 (None → CICIDS2018 LOG_FEATURES)

    Returns:
        X_scaled     : ndarray (n_samples, n_features)
        y            : ndarray (n_samples,) — 0=benign, 1=attack
        attack_step  : ndarray (n_samples,) — 문자열 레이블
        feature_names: list
        scaler       : 학습된 RobustScaler
        clip_params  : dict {col: (lower, upper)} — 훈련 기준 clip 경계
        df           : 원본 샘플링 DataFrame (레이블 접근용)
    """
    if all_features is None:
        all_features = ALL_FEATURES

    df = load_and_sample(csv_path, use_udbb=use_udbb, udbb_counts=udbb_counts)

    available = [f for f in all_features if f in df.columns]
    X_scaled, feature_names, scaler, clip_params = preprocess(df, feature_cols=available, log_features=log_features)

    y           = df["attack_flag"].astype(int).values
    attack_step = df["attack_step"].values

    if save_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame(X_scaled, columns=feature_names)
        out["attack_flag"]  = y
        out["attack_step"]  = attack_step
        save_path = PROCESSED_DIR / "cicids2018_processed.csv"
        out.to_csv(save_path, index=False)
        print(f"전처리 데이터 저장: {save_path}")

    return X_scaled, y, attack_step, feature_names, scaler, clip_params, df
