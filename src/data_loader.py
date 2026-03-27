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
)


def load_and_sample(csv_path=None, use_udbb=True) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 UDBB 샘플링을 적용합니다.

    Args:
        csv_path: CSV 경로 (None이면 config의 TRAIN_FILE 사용)
        use_udbb: True → UDBB 샘플링, False → 전체 데이터 반환

    Returns:
        샘플링된 DataFrame
    """
    if csv_path is None:
        csv_path = TRAIN_FILE

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
    n_benign = min(UDBB_COUNTS["benign"], len(benign_pool))
    samples.append(benign_pool.sample(n=n_benign, random_state=RANDOM_SEED))
    print(f"  benign       : {n_benign:,}")

    # 공격 단계별
    for step, n_target in UDBB_COUNTS.items():
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
):
    """
    피처 전처리 (NetFlowGap 파이프라인 기반).

    Args:
        df: 입력 DataFrame
        feature_cols: 사용할 피처 목록 (None → ALL_FEATURES 중 존재하는 것)
        fit_scaler: True → 새 scaler fit, False → 전달된 scaler로 transform
        scaler: fit_scaler=False일 때 사용할 사전 fit된 scaler

    Returns:
        X_scaled (ndarray), feature_names (list), scaler
    """
    if feature_cols is None:
        feature_cols = [f for f in ALL_FEATURES if f in df.columns]

    log_feats = [f for f in LOG_FEATURES if f in feature_cols]
    non_log   = [f for f in feature_cols if f not in log_feats]

    X = df[feature_cols].copy()

    # Inf/NaN 처리
    X = X.replace({np.inf: np.nan, -np.inf: np.nan, -1: np.nan})
    X = X.fillna(X.median())

    # Percentile clipping
    for col in log_feats:
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=0, upper=upper)

    for col in non_log:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=lower, upper=upper)

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

    return X_scaled, list(feature_cols), scaler


def load_dataset(
    csv_path=None,
    use_udbb: bool = True,
    save_processed: bool = False,
):
    """
    전체 로드 + 샘플링 + 전처리 파이프라인.

    Returns:
        X_scaled     : ndarray (n_samples, n_features)
        y            : ndarray (n_samples,) — 0=benign, 1=attack
        attack_step  : ndarray (n_samples,) — 문자열 레이블
        feature_names: list
        scaler       : 학습된 RobustScaler
        df           : 원본 샘플링 DataFrame (레이블 접근용)
    """
    df = load_and_sample(csv_path, use_udbb=use_udbb)

    available = [f for f in ALL_FEATURES if f in df.columns]
    X_scaled, feature_names, scaler = preprocess(df, feature_cols=available)

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

    return X_scaled, y, attack_step, feature_names, scaler, df
