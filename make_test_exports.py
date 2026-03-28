"""
make_test_exports.py
--------------------
대용량 파일을 청크 단위로 처리하여 test_*.csv를 생성합니다.

실행:
    python make_test_exports.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.config import (
    TRAIN_FILE, TEST_FILE, EXPORTS_DIR,
    ALL_FEATURES, LOG_FEATURES,
    UDBB_COUNTS, RANDOM_SEED,
)

LABEL_COLS  = ["attack_flag", "attack_step"]
CHUNK_SIZE  = 200_000   # 청크 크기 (행 수)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def get_feature_groups() -> dict:
    """exports/ 안의 train_*.csv 헤더에서 그룹별 피처 목록을 읽습니다."""
    groups = {}
    for path in sorted(EXPORTS_DIR.glob("train_*.csv")):
        name = path.stem[len("train_"):]
        df_head = pd.read_csv(path, nrows=0)
        feats = [c for c in df_head.columns if c not in set(LABEL_COLS)]
        groups[name] = feats
    return groups


# ── Step 1: training-flow.csv → UDBB 샘플 (청크 처리) ───────────────────────

def udbb_sample_chunked(needed_cols: list) -> pd.DataFrame:
    """
    training-flow.csv를 청크 단위로 읽어 UDBB 샘플을 수집합니다.
    각 카테고리에서 목표 행 수가 채워지면 조기 종료합니다.
    """
    load_cols = list(dict.fromkeys(needed_cols + LABEL_COLS))   # 중복 제거
    targets   = dict(UDBB_COUNTS)   # {'benign': 60000, 'action': 20000, ...}
    buckets   = {k: [] for k in targets}
    counts    = {k: 0  for k in targets}

    print(f"  {TRAIN_FILE.name} 청크 읽기 시작 (청크={CHUNK_SIZE:,}행) ...")

    for chunk in pd.read_csv(TRAIN_FILE, usecols=load_cols,
                             chunksize=CHUNK_SIZE, low_memory=False):
        chunk["attack_flag"] = (pd.to_numeric(chunk["attack_flag"], errors="coerce")
                                .fillna(0).astype(int))
        chunk["attack_step"] = (chunk["attack_step"].fillna("benign")
                                .astype(str).str.strip().str.lower())

        # 카테고리별 수집
        if counts["benign"] < targets["benign"]:
            rows = chunk[chunk["attack_flag"] == 0]
            need = targets["benign"] - counts["benign"]
            buckets["benign"].append(rows.iloc[:need])
            counts["benign"] += min(len(rows), need)

        for step in ("action", "infection", "installation"):
            if counts[step] < targets[step]:
                rows = chunk[(chunk["attack_flag"] == 1) & (chunk["attack_step"] == step)]
                need = targets[step] - counts[step]
                buckets[step].append(rows.iloc[:need])
                counts[step] += min(len(rows), need)

        if all(counts[k] >= targets[k] for k in targets):
            print("  모든 카테고리 목표 충족 — 조기 종료")
            break

    for k, v in counts.items():
        print(f"    {k:<12}: {v:,} / {targets[k]:,}")

    df = pd.concat([row for rows in buckets.values() for row in rows])
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# ── Step 2: 전처리 & scaler fit (훈련 샘플 기준) ─────────────────────────────

def fit_preprocessor(df_train: pd.DataFrame, feature_cols: list):
    """
    훈련 샘플로 clip 임계값 + RobustScaler를 fit합니다.

    Returns:
        scaler      : fit된 RobustScaler
        clip_bounds : {col: (lower, upper)} — test 청크에 동일하게 적용
        log_feats   : log1p 적용 피처 목록
        non_log     : 나머지 피처 목록
    """
    log_feats = [f for f in LOG_FEATURES if f in feature_cols]
    non_log   = [f for f in feature_cols if f not in log_feats]

    X = df_train[feature_cols].copy()
    X = X.replace({np.inf: np.nan, -np.inf: np.nan, -1: np.nan})
    X = X.fillna(X.median())

    clip_bounds = {}
    for col in log_feats:
        upper = float(X[col].quantile(0.99))
        X[col] = X[col].clip(lower=0, upper=upper)
        clip_bounds[col] = (0.0, upper)
    for col in non_log:
        lower = float(X[col].quantile(0.01))
        upper = float(X[col].quantile(0.99))
        X[col] = X[col].clip(lower=lower, upper=upper)
        clip_bounds[col] = (lower, upper)

    X[log_feats] = np.log1p(X[log_feats].clip(lower=0))

    scaler = RobustScaler()
    scaler.fit(X)

    return scaler, clip_bounds, log_feats, non_log


# ── Step 3: test-flow.csv 청크 처리 → 여러 CSV에 동시 기록 ──────────────────

def process_test_chunked(
    feature_cols: list,
    groups: dict,
    scaler,
    clip_bounds: dict,
    log_feats: list,
):
    """
    test-flow.csv를 청크 단위로 읽고 전처리하여 각 그룹 test CSV에 씁니다.
    첫 청크에 header=True, 이후 header=False로 파일을 이어씁니다.
    """
    load_cols = list(dict.fromkeys(feature_cols + LABEL_COLS))
    feat_idx  = {f: i for i, f in enumerate(feature_cols)}

    out_paths    = {name: EXPORTS_DIR / f"test_{name}.csv" for name in groups}
    header_flags = {name: True for name in groups}   # 첫 청크는 헤더 포함

    total_rows = 0
    chunk_num  = 0

    print(f"  {TEST_FILE.name} 청크 처리 시작 (청크={CHUNK_SIZE:,}행) ...")

    for chunk in pd.read_csv(TEST_FILE, usecols=load_cols,
                             chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num += 1

        chunk["attack_flag"] = (pd.to_numeric(chunk["attack_flag"], errors="coerce")
                                .fillna(0).astype(int))
        chunk["attack_step"] = (chunk["attack_step"].fillna("benign")
                                .astype(str).str.strip().str.lower())

        X = chunk[feature_cols].copy()
        X = X.replace({np.inf: np.nan, -np.inf: np.nan, -1: np.nan})
        X = X.fillna(X.median())

        # clip (훈련 기준 임계값 적용)
        for col, (lo, hi) in clip_bounds.items():
            if col in log_feats:
                X[col] = X[col].clip(lower=lo, upper=hi)
            else:
                X[col] = X[col].clip(lower=lo, upper=hi)

        # log1p
        X[log_feats] = np.log1p(X[log_feats].clip(lower=0))

        # RobustScaler
        X_scaled = scaler.transform(X)

        y    = chunk["attack_flag"].values
        step = chunk["attack_step"].values
        total_rows += len(chunk)

        # 그룹별 저장
        for name, feats in groups.items():
            missing = [f for f in feats if f not in feat_idx]
            if missing:
                continue

            idx    = [feat_idx[f] for f in feats]
            df_out = pd.DataFrame(X_scaled[:, idx], columns=feats)
            df_out["attack_flag"] = y
            df_out["attack_step"] = step

            df_out.to_csv(
                out_paths[name],
                mode="a" if not header_flags[name] else "w",
                header=header_flags[name],
                index=False,
            )
            header_flags[name] = False

        if chunk_num % 5 == 0:
            print(f"    청크 {chunk_num} 처리 완료 ({total_rows:,}행 누적)")

    print(f"  test 전체 처리 완료: {total_rows:,}행")
    return total_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 65
    print(SEP)
    print("[make_test_exports] test_*.csv 생성 (대용량 청크 처리)")
    print(SEP)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 피처 그룹 읽기
    print("\n[1/4] train_*.csv 헤더에서 피처 목록 로드 ...")
    groups = get_feature_groups()
    if not groups:
        print("  [!] exports/ 에 train_*.csv 가 없습니다.")
        return
    for name, feats in groups.items():
        preview = ", ".join(feats[:4]) + ("  ..." if len(feats) > 4 else "")
        print(f"  {name:<20} {len(feats):>3}개  [{preview}]")

    # 모든 그룹에서 필요한 피처의 합집합 (ALL_FEATURES 순서 유지)
    needed_feats_set = {f for feats in groups.values() for f in feats}
    feature_cols = [f for f in ALL_FEATURES if f in needed_feats_set]

    # 2) UDBB 샘플 수집
    print("\n[2/4] training-flow.csv UDBB 샘플 수집 ...")
    df_train = udbb_sample_chunked(feature_cols)

    # 3) 전처리 & scaler fit
    print("\n[3/4] scaler fit ...")
    scaler, clip_bounds, log_feats, non_log = fit_preprocessor(df_train, feature_cols)
    print(f"  scaler fit 완료 ({df_train.shape[0]:,}행, {len(feature_cols)}개 피처)")

    # 4) test-flow.csv 청크 처리
    print("\n[4/4] test-flow.csv 청크 처리 → test_*.csv 저장 ...")
    total = process_test_chunked(feature_cols, groups, scaler, clip_bounds, log_feats)

    print(f"\n  {'그룹':<20}  {'파일'}")
    print(f"  {'-'*20}  {'-'*35}")
    for name in groups:
        path = EXPORTS_DIR / f"test_{name}.csv"
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"  {name:<20}  test_{name}.csv  ({size_mb:.1f} MB)")

    print(f"\n저장 위치: {EXPORTS_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
