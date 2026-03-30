"""
UNSW-NB15 ML 결과 분석:
비교군별 Boundary_Mean × 평균 F1 상관 (Spearman)
unknown 섹션 제외, section "1. all" 기준
"""

import pandas as pd
from scipy.stats import spearmanr
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# --- BM 값 로드 ---
metrics = pd.read_csv(os.path.join(BASE, "../logs/comparison_metrics.csv"))
bm = metrics.set_index("group")["Boundary_Mean"].to_dict()

# --- ML output CSV 파싱 (section "1. all"만 추출) ---
FILES = {
    "cass":       "output-cass1.csv",
    "anova":      "output-anova1.csv",
    "extratrees": "output-extratrees1.csv",
    "random":     "output-random1.csv",
    "lit_yin2023":"output-yin20231.csv",
}

MODELS = ["random_forest", "cnn", "logistic_regression", "lstm", "xgboost"]

def parse_all_section(filepath):
    """파일에서 '1. all' 섹션의 metric 딕셔너리를 반환"""
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    in_section = False
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped == "1. all":
            in_section = True
            continue
        if in_section:
            # 다음 섹션 시작: "2. " 로 시작하는 줄
            if stripped.startswith("2. ") or stripped.startswith("3. ") \
               or stripped.startswith("4. ") or stripped.startswith("5. "):
                break
            if stripped:
                rows.append(stripped)

    if not rows:
        return {}
    header = rows[0].split(",")
    result = {}
    for row in rows[1:]:
        vals = row.split(",")
        if len(vals) == len(header):
            result[vals[0]] = dict(zip(header[1:], vals[1:]))
    return result

# --- 결과 수집 ---
# all_macro_F1 × 5/4 = unknown 제외 macro F1
# 근거: macro F1 = (F1_benign + F1_recon + F1_infection + F1_action + F1_unknown=0) / 5
#       → F1_without_unknown = all_macro_F1 × 5 / 4
NUM_CLASSES_TOTAL = 5
NUM_CLASSES_NO_UNKNOWN = 4

records = []
for group, fname in FILES.items():
    fpath = os.path.join(BASE, fname)
    section = parse_all_section(fpath)
    f1_row = section.get("f1_score", {})

    per_model_f1 = {}
    for mod in MODELS:
        try:
            raw = float(f1_row[mod])
            per_model_f1[mod] = round(raw * NUM_CLASSES_TOTAL / NUM_CLASSES_NO_UNKNOWN, 2)
        except (KeyError, ValueError):
            per_model_f1[mod] = None

    valid = [v for v in per_model_f1.values() if v is not None]
    avg_f1 = round(sum(valid) / len(valid), 3) if valid else None

    records.append({
        "group": group,
        "Boundary_Mean": bm.get(group),
        "avg_F1": avg_f1,
        **{f"F1_{m}": per_model_f1[m] for m in MODELS},
    })

df = pd.DataFrame(records).sort_values("Boundary_Mean", ascending=False)
df["avg_F1"] = pd.to_numeric(df["avg_F1"], errors="coerce")
df["BM_rank"] = df["Boundary_Mean"].rank(ascending=False).astype(int)
df["F1_rank"] = df["avg_F1"].rank(ascending=False).astype(int)

# --- Spearman 상관 ---
rho, pval = spearmanr(df["Boundary_Mean"], df["avg_F1"])

# --- 출력 ---
print("=" * 70)
print("UNSW-NB15: Boundary_Mean × 평균 F1 (5개 ML 모델, unknown 제외 보정)")
print("=" * 70)

display_cols = ["group", "Boundary_Mean", "BM_rank",
                "F1_random_forest", "F1_cnn", "F1_logistic_regression",
                "F1_lstm", "F1_xgboost", "avg_F1", "F1_rank"]
print(df[display_cols].to_string(index=False))

print()
print(f"Spearman ρ (BM vs avg_F1) = {rho:.4f},  p-value = {pval:.4f}")
print()

# 순위 일치 여부
rank_match = (df["BM_rank"] == df["F1_rank"]).sum()
print(f"BM 순위 = F1 순위 일치: {rank_match} / {len(df)} 개 비교군")

# --- CSV 저장 ---
out_path = os.path.join(BASE, "bm_f1_correlation.csv")
df.to_csv(out_path, index=False)
print(f"\n저장: {out_path}")
