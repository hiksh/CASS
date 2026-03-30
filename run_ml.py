"""
run_ml.py — CASS ML 단독 평가 스크립트

exports/ 에 train_*.csv / test_*.csv 가 이미 존재하는 경우,
UMAP 파이프라인 없이 ML 평가만 바로 실행합니다.

실행:
  python run_ml.py --dataset unsw_nb15   # anova / cass / extratrees / random / lit_yin2023
  python run_ml.py --dataset cicids2018  # anova / cass / extratrees / random / lit_umar2024
  python run_ml.py --dataset mirai
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import get_dataset_config
from src.ml_runner import run_ml_evaluation

SEP  = "=" * 65
SEP2 = "-" * 65

# 데이터셋별 기대 비교군 (파일이 없으면 자동으로 건너뜀)
EXPECTED_GROUPS = {
    "cicids2018": ["cass", "anova", "extratrees", "random", "lit_umar2024"],
    "unsw_nb15":  ["cass", "anova", "extratrees", "random", "lit_yin2023"],
    "mirai":      ["cass", "anova", "extratrees", "random"],
}


def main(args) -> None:
    ds = get_dataset_config(args.dataset)
    exports_dir = ds["exports_dir"]
    ml_dir      = ds["ml_dir"]

    print(SEP)
    print("CASS — ML 단독 평가")
    print(SEP2)
    print(f"  Dataset     : {args.dataset}")
    print(f"  Exports     : {exports_dir}")
    print(f"  ML 결과 저장: {ml_dir}")
    print(SEP)

    # exports 존재 여부 확인
    if not exports_dir.exists():
        print(f"[오류] exports 디렉토리 없음: {exports_dir}")
        print("  먼저 'python main.py --export' 를 실행하세요.")
        sys.exit(1)

    # 기대 비교군 vs 실제 파일 비교
    expected = EXPECTED_GROUPS.get(args.dataset, [])
    missing  = [g for g in expected if not (exports_dir / f"train_{g}.csv").exists()]
    found    = [f.stem[len("train_"):] for f in sorted(exports_dir.glob("train_*.csv"))]

    print(f"\n  발견된 비교군 ({len(found)}개): {found}")
    if missing:
        print(f"  [경고] 기대했지만 없는 비교군: {missing}")
    if not found:
        print("[오류] train_*.csv 파일이 없습니다.")
        sys.exit(1)

    print()
    results_df = run_ml_evaluation(
        exports_dir=exports_dir,
        ml_dir=ml_dir,
        dataset_name=args.dataset,
    )

    print(f"\n{SEP}")
    print("ML 평가 완료")
    print(SEP2)
    print(f"  f1_results.csv  : {ml_dir / 'f1_results.csv'}")
    print(f"  f1_heatmap.png  : {ml_dir / 'f1_heatmap.png'}")
    print(f"  f1_bar.png      : {ml_dir / 'f1_bar.png'}")
    print(f"  ag_models/      : {ml_dir / 'ag_models'}")
    print(SEP)


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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
