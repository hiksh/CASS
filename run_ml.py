"""
run_ml.py — CASS ML 단독 평가 스크립트

--umap 없이: exports/ 에 train_*.csv / test_*.csv 가 이미 존재하는 경우,
             UMAP 파이프라인 없이 ML 평가만 바로 실행합니다.

--umap 포함: logs/search_results_*.csv 에서 best_subset 을 읽고,
             training-flow.csv 를 UDBB 샘플링 + 전처리하여 X_scaled 를 재현한 뒤
             UMAP 임베딩 export 후 전체 비교군 ML 평가를 실행합니다.

실행:
  python run_ml.py --dataset cicids2018             # ML만 (기존)
  python run_ml.py --dataset cicids2018 --umap      # UMAP 임베딩 export + ML
  python run_ml.py --dataset unsw_nb15  --umap
  python run_ml.py --dataset mirai      --umap
"""
import ast
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import get_dataset_config, MIN_SILHOUETTE
from src.data_loader import load_and_sample, preprocess
from src.exporter import export_umap_embeddings
from src.ml_runner import run_ml_evaluation

SEP  = "=" * 65
SEP2 = "-" * 65

# 데이터셋별 기대 비교군 (파일이 없으면 자동으로 건너뜀)
EXPECTED_GROUPS = {
    "cicids2018": ["cass", "anova", "extratrees", "random", "lit_umar2024"],
    "unsw_nb15":  ["cass", "anova", "extratrees", "random", "lit_yin2023"],
    "mirai":      ["cass", "anova", "extratrees", "random"],
}
UMAP_GROUPS = ["umap2d", "umap3d"]


# ── best_subset 읽기 ──────────────────────────────────────────────────────────

def _read_best_subset(logs_dir: Path) -> list:
    """
    search_results_greedy.csv 또는 search_results_random.csv 에서
    best_subset 피처 리스트를 읽습니다.

    선택 기준:
      full_sil > MIN_SILHOUETTE 를 만족하는 행 중 boundary_mean 최댓값.
      만족하는 행이 없으면 전체 대상 boundary_mean 최댓값 (제약 완화).
    """
    for mode in ("greedy", "random"):
        path = logs_dir / f"search_results_{mode}.csv"
        if path.exists():
            print(f"  search results : {path.name}")
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            f"search_results_greedy.csv / search_results_random.csv 를 찾을 수 없습니다.\n"
            f"  경로: {logs_dir}\n"
            f"  먼저 'python main.py --dataset <dataset>' 을 실행하여 탐색 결과를 생성하세요."
        )

    valid = df.dropna(subset=["full_sil", "boundary_mean"])
    if valid.empty:
        raise ValueError("유효한 full_sil / boundary_mean 값이 없습니다. 탐색 결과를 확인하세요.")

    survived = valid[valid["full_sil"] > MIN_SILHOUETTE]
    if survived.empty:
        print(f"  [경고] full_sil > {MIN_SILHOUETTE} 만족하는 행 없음 → 제약 완화")
        survived = valid

    best_row  = survived.loc[survived["boundary_mean"].idxmax()]
    features  = ast.literal_eval(best_row["features"])
    bm        = float(best_row["boundary_mean"])
    sil       = float(best_row["full_sil"])

    print(f"  best_subset ({len(features)}개 피처) | boundary_mean={bm:.4f}  full_sil={sil:.4f}")
    for f in features:
        print(f"    - {f}")

    return features


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(args) -> None:
    ds          = get_dataset_config(args.dataset)
    exports_dir = ds["exports_dir"]
    ml_dir      = ds["ml_dir"]
    logs_dir    = ds["logs_dir"]

    print(SEP)
    print("CASS — ML 단독 평가")
    print(SEP2)
    print(f"  Dataset     : {args.dataset}")
    print(f"  UMAP        : {'ON' if args.umap else 'OFF'}")
    print(f"  Exports     : {exports_dir}")
    print(f"  ML 결과 저장: {ml_dir}")
    print(SEP)

    # ── [1~3] UMAP 임베딩 export (--umap 시에만) ─────────────────────────────
    if args.umap:

        # [1] best_subset 읽기
        print(f"\n{SEP}")
        print("[1/4] best_subset 읽기")
        print(SEP2)
        best_subset = _read_best_subset(logs_dir)

        # [2] 데이터 로드 & UDBB 전처리 (best_subset 컬럼만 로드 — 메모리 절약)
        print(f"\n{SEP}")
        print("[2/4] 데이터 로드 & UDBB 전처리  (best_subset 컬럼만 로드)")
        print(SEP2)
        df_train = load_and_sample(
            ds["train_file"],
            use_udbb    = True,
            udbb_counts = ds["udbb_counts"],
            usecols     = best_subset,          # 필요한 컬럼만 읽음
        )
        relevant_log = [f for f in ds["log_features"] if f in set(best_subset)]
        X_scaled, _, scaler, clip_params = preprocess(
            df_train,
            feature_cols = best_subset,
            fit_scaler   = True,
            log_features = relevant_log,
        )
        y            = df_train["attack_flag"].astype(int).values
        attack_step  = df_train["attack_step"].values
        feature_names = best_subset             # subset만 사용

        n_benign = int((y == 0).sum())
        n_attack = int((y == 1).sum())
        print(f"  훈련 샘플 : {len(y):,}  (benign={n_benign:,}  attack={n_attack:,})")
        print(f"  로드 피처 : {len(best_subset)}개  (best_subset만)")

        # [3] UMAP 임베딩 export
        print(f"\n{SEP}")
        print("[3/4] UMAP 임베딩 Export (2D / 3D)")
        print(SEP2)
        export_umap_embeddings(
            X_scaled, y, attack_step, feature_names,
            best_subset, scaler, clip_params,
            export_dir   = exports_dir,
            test_file    = ds["test_file"],
            log_features = ds["log_features"],
        )

        stage_label = "[4/4]"
    else:
        stage_label = ""

    # ── ML 평가 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"{stage_label} ML 평가".strip())
    print(SEP2)

    if not exports_dir.exists() or not list(exports_dir.glob("train_*.csv")):
        print(f"[오류] {exports_dir} 에 train_*.csv 없음")
        print("  'python main.py --export' 또는 '--umap' 을 먼저 실행하세요.")
        sys.exit(1)

    expected = EXPECTED_GROUPS.get(args.dataset, [])
    if args.umap:
        expected = expected + UMAP_GROUPS
    found   = [f.stem[len("train_"):] for f in sorted(exports_dir.glob("train_*.csv"))]
    missing = [g for g in expected if g not in found]

    print(f"  발견된 비교군 ({len(found)}개): {found}")
    if missing:
        print(f"  [경고] 기대했지만 없는 비교군: {missing}")
    print()

    results_df = run_ml_evaluation(
        exports_dir  = exports_dir,
        ml_dir       = ml_dir,
        dataset_name = args.dataset,
    )

    # ── 완료 요약 ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("ML 평가 완료")
    print(SEP2)
    if args.umap:
        print(f"  train_umap2d.csv / test_umap2d.csv : {exports_dir}")
        print(f"  train_umap3d.csv / test_umap3d.csv : {exports_dir}")
    print(f"  f1_results.csv  : {ml_dir / 'f1_results.csv'}")
    print(f"  f1_heatmap.png  : {ml_dir / 'f1_heatmap.png'}")
    print(f"  f1_bar.png      : {ml_dir / 'f1_bar.png'}")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CASS ML 단독 평가 — exports가 이미 있는 경우 사용",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["cicids2018", "unsw_nb15", "mirai"],
        default="unsw_nb15",
        help="평가할 데이터셋",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help=(
            "logs/search_results_*.csv 에서 best_subset 을 읽어 "
            "UMAP 2D/3D 임베딩 CSV 를 생성한 뒤 ML 평가에 포함합니다. "
            "training-flow.csv 로드 및 UDBB 전처리를 수행합니다."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
