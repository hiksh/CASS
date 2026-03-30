"""
UNSW-NB15: UMAP 지표 × ML 성능 피어슨 상관계수 히트맵
- UMAP 지표 8개 (comparison_metrics.csv)
- ML 지표: F1 / Precision / Recall / Accuracy × 5개 모델
- unknown 제외 보정: macro 지표(F1/Precision/Recall) × 5/4
- 데이터 포인트: 비교군 5개 (cass, anova, extratrees, random, lit_yin2023)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. UMAP 지표 로드 ─────────────────────────────────────────
metrics_df = pd.read_csv(os.path.join(BASE, "../logs/comparison_metrics.csv"))
metrics_df = metrics_df.set_index("group")

UMAP_COLS = [
    "Silhouette", "Centroid_to_Benign", "Global_Mean_Dist",
    "Boundary_Mean", "Camouflage@1.0",
    "HDBSCAN_Noise_Rate", "Cluster_Count", "Cohesion_Dist",
]
# best 방향 표시 (히트맵 레이블용)
UMAP_DIR = {
    "Silhouette": "↑", "Centroid_to_Benign": "↑", "Global_Mean_Dist": "↑",
    "Boundary_Mean": "↑", "Camouflage@1.0": "↓",
    "HDBSCAN_Noise_Rate": "↓", "Cluster_Count": "↓", "Cohesion_Dist": "↓",
}

# ── 2. ML output CSV 파싱 ─────────────────────────────────────
FILES = {
    "cass":       "output-cass1.csv",
    "anova":      "output-anova1.csv",
    "extratrees": "output-extratrees1.csv",
    "random":     "output-random1.csv",
    "lit_yin2023":"output-yin20231.csv",
}
MODELS = ["random_forest", "cnn", "logistic_regression", "lstm", "xgboost"]
MACRO_METRICS = ["f1_score", "recall", "precision"]   # unknown 보정 필요
RAW_METRICS   = ["accuracy"]                           # 보정 불필요
ML_METRIC_KEYS = MACRO_METRICS + RAW_METRICS

N_TOTAL_CLASSES    = 5
N_UNKNOWN_EXCLUDED = 4
CORRECTION = N_TOTAL_CLASSES / N_UNKNOWN_EXCLUDED      # 5/4

def parse_all_section(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    in_sec, rows = False, []
    for line in lines:
        s = line.strip()
        if s == "1. all":
            in_sec = True
            continue
        if in_sec:
            if any(s.startswith(f"{i}. ") for i in range(2, 6)):
                break
            if s:
                rows.append(s)
    if not rows:
        return {}
    header = rows[0].split(",")
    result = {}
    for row in rows[1:]:
        vals = row.split(",")
        if len(vals) == len(header):
            result[vals[0]] = dict(zip(header[1:], vals[1:]))
    return result

# ── 3. ML 데이터 수집 ─────────────────────────────────────────
records = {}
for group, fname in FILES.items():
    fpath = os.path.join(BASE, fname)
    sec = parse_all_section(fpath)
    row = {}
    for metric_key in ML_METRIC_KEYS:
        metric_row = sec.get(metric_key, {})
        for model in MODELS:
            col = f"{model}_{metric_key}"
            try:
                val = float(metric_row[model])
                if metric_key in MACRO_METRICS:
                    val = val * CORRECTION
            except (KeyError, ValueError):
                val = np.nan
            row[col] = val
    records[group] = row

ml_df = pd.DataFrame(records).T   # index=group, columns=model_metric
ml_df.index.name = "group"

# ── 4. 모델별 + 평균 컬럼 구성 ──────────────────────────────
# 평균 컬럼 추가
for metric_key in ML_METRIC_KEYS:
    cols = [f"{m}_{metric_key}" for m in MODELS]
    ml_df[f"avg_{metric_key}"] = ml_df[cols].mean(axis=1)

# ── 5. 피어슨 상관 계산 ──────────────────────────────────────
# 분석 대상 ML 컬럼: 모델별 4지표 + 평균 4지표
model_metric_cols = [f"{m}_{k}" for m in MODELS for k in ML_METRIC_KEYS]
avg_metric_cols   = [f"avg_{k}" for k in ML_METRIC_KEYS]
all_ml_cols = model_metric_cols + avg_metric_cols

groups = list(FILES.keys())
umap_vals = metrics_df.loc[groups, UMAP_COLS]
ml_vals   = ml_df.loc[groups, all_ml_cols]

corr_matrix = pd.DataFrame(index=UMAP_COLS, columns=all_ml_cols, dtype=float)
pval_matrix = pd.DataFrame(index=UMAP_COLS, columns=all_ml_cols, dtype=float)

for uc in UMAP_COLS:
    for mc in all_ml_cols:
        x = umap_vals[uc].values.astype(float)
        y = ml_vals[mc].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 3:
            r, p = pearsonr(x[mask], y[mask])
        else:
            r, p = np.nan, np.nan
        corr_matrix.loc[uc, mc] = r
        pval_matrix.loc[uc, mc] = p

# ── 6. 시각화 ────────────────────────────────────────────────
MODEL_LABELS = {
    "random_forest": "RF", "cnn": "CNN",
    "logistic_regression": "LR", "lstm": "LSTM", "xgboost": "XGB",
}
METRIC_LABELS = {
    "f1_score": "F1", "precision": "Prec", "recall": "Rec", "accuracy": "Acc",
}

def make_col_label(col):
    if col.startswith("avg_"):
        k = col[4:]
        return f"Avg\n{METRIC_LABELS.get(k, k)}"
    for m in MODELS:
        if col.startswith(m + "_"):
            k = col[len(m)+1:]
            return f"{MODEL_LABELS.get(m, m)}\n{METRIC_LABELS.get(k, k)}"
    return col

col_labels = [make_col_label(c) for c in all_ml_cols]
row_labels  = [f"{c} {UMAP_DIR[c]}" for c in UMAP_COLS]

corr_arr = corr_matrix.values.astype(float)
pval_arr = pval_matrix.values.astype(float)

# 전체 히트맵 (모델별 + 평균)
fig, ax = plt.subplots(figsize=(22, 6))
im = ax.imshow(corr_arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(all_ml_cols)))
ax.set_yticks(range(len(UMAP_COLS)))
ax.set_xticklabels(col_labels, fontsize=8)
ax.set_yticklabels(row_labels, fontsize=9)
plt.setp(ax.get_xticklabels(), ha="center")

# 구분선: 모델별 4개씩 그룹 + 평균 4개
for i in range(1, len(MODELS) + 1):
    ax.axvline(i * len(ML_METRIC_KEYS) - 0.5, color="white", lw=1.5)

# 셀에 r값 표시, p<0.05 * 표시
for i in range(len(UMAP_COLS)):
    for j in range(len(all_ml_cols)):
        r = corr_arr[i, j]
        p = pval_arr[i, j]
        if np.isnan(r):
            continue
        star = "*" if p < 0.05 else ""
        txt_color = "white" if abs(r) > 0.65 else "black"
        ax.text(j, i, f"{r:.2f}{star}", ha="center", va="center",
                fontsize=7.5, color=txt_color, fontweight="bold" if star else "normal")

# 모델 그룹 레이블 (상단)
model_names_display = [MODEL_LABELS[m] for m in MODELS] + ["Avg"]
n_metrics = len(ML_METRIC_KEYS)
for idx, mname in enumerate(model_names_display):
    center = idx * n_metrics + (n_metrics - 1) / 2
    ax.text(center, -1.2, mname, ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color="dimgray" if mname != "Avg" else "navy")

plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.02, pad=0.01)
ax.set_title(
    "UNSW-NB15: Pearson Correlation — UMAP Metrics × ML Performance\n"
    "(unknown excluded, macro×5/4 corrected, n=5 comparison groups, * p<0.05)",
    fontsize=11, pad=20
)
plt.tight_layout()
out_full = os.path.join(BASE, "correlation_heatmap_full.png")
plt.savefig(out_full, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out_full}")

# 평균 지표만 히트맵 (요약용)
avg_corr = corr_matrix[avg_metric_cols].values.astype(float)
avg_pval = pval_matrix[avg_metric_cols].values.astype(float)
avg_col_labels = [METRIC_LABELS.get(c[4:], c[4:]) for c in avg_metric_cols]

fig2, ax2 = plt.subplots(figsize=(7, 6))
im2 = ax2.imshow(avg_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(avg_metric_cols)))
ax2.set_yticks(range(len(UMAP_COLS)))
ax2.set_xticklabels(avg_col_labels, fontsize=11)
ax2.set_yticklabels(row_labels, fontsize=11)

for i in range(len(UMAP_COLS)):
    for j in range(len(avg_metric_cols)):
        r = avg_corr[i, j]
        p = avg_pval[i, j]
        if np.isnan(r):
            continue
        star = "*" if p < 0.05 else ""
        txt_color = "white" if abs(r) > 0.65 else "black"
        ax2.text(j, i, f"{r:.2f}{star}", ha="center", va="center",
                 fontsize=10, color=txt_color,
                 fontweight="bold" if star else "normal")

plt.colorbar(im2, ax=ax2, label="Pearson r", fraction=0.04, pad=0.02)
ax2.set_title(
    "UNSW-NB15: UMAP Metrics × Avg ML Performance\n"
    "(unknown excluded, n=5, * p<0.05)",
    fontsize=11
)
plt.tight_layout()
out_avg = os.path.join(BASE, "correlation_heatmap_avg.png")
plt.savefig(out_avg, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out_avg}")

# ── 7. 수치 CSV 저장 ─────────────────────────────────────────
corr_matrix.to_csv(os.path.join(BASE, "correlation_pearson.csv"))
pval_matrix.to_csv(os.path.join(BASE, "correlation_pvalue.csv"))
print("저장: correlation_pearson.csv / correlation_pvalue.csv")

# ── 8. 주목할 상관 출력 ──────────────────────────────────────
print("\n=== |r| > 0.7 인 상관 ===")
for uc in UMAP_COLS:
    for mc in avg_metric_cols:
        r = corr_matrix.loc[uc, mc]
        p = pval_matrix.loc[uc, mc]
        if not np.isnan(r) and abs(r) > 0.7:
            sig = "(*p<0.05)" if p < 0.05 else f"(p={p:.3f})"
            print(f"  {uc:25s} × {mc:20s}  r={r:+.3f}  {sig}")
