"""
기존 comparison_metrics.csv → 히트맵 PNG 재생성
(cuml 없이 플롯만 재실행)

실행:
  python regen_heatmaps.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 지표 방향 정의 ─────────────────────────────────────────────────────────────
_METRIC_HIGHER_BETTER = {
    "Silhouette":         True,
    "Centroid_to_Benign": True,
    "Boundary_Mean":      True,
    "Cohesion_Dist":      False,
}

# 표시에서 제외할 지표
_EXCLUDE_METRICS = {"Global_Mean_Dist", "HDBSCAN_Noise_Rate", "Cluster_Count"}

# ── 표시용 컬럼 이름 단축 ──────────────────────────────────────────────────────
_LABEL_MAP = {
    "Silhouette":         "Silhouette",
    "Centroid_to_Benign": "Centroid",
    "Global_Mean_Dist":   "Global\nDist",
    "Boundary_Mean":      "BM",
    "HDBSCAN_Noise_Rate": "Noise\nRate",
    "Cluster_Count":      "Clusters",
    "Cohesion_Dist":      "Cohesion",
}

def _short(c):
    if c.startswith("Camouflage@"):
        return "Cam@" + c.split("@")[1]
    return _LABEL_MAP.get(c, c)


def plot_comparison_heatmap(metrics_df: pd.DataFrame, save_path: Path) -> None:
    exclude     = {"n_features"} | _EXCLUDE_METRICS
    metric_cols = [c for c in metrics_df.columns if c not in exclude]
    df_raw      = metrics_df[metric_cols].copy().astype(float)

    # 컬럼 순서 정렬
    sep_cols = [c for c in ["Silhouette", "Centroid_to_Benign"]
                if c in df_raw.columns]
    cam_cols = ([c for c in df_raw.columns if c.startswith("Camouflage")]
                + [c for c in ["Boundary_Mean"] if c in df_raw.columns])
    clu_cols = [c for c in ["Cohesion_Dist"]
                if c in df_raw.columns]
    col_order = sep_cols + cam_cols + clu_cols
    df_raw    = df_raw[col_order]

    # 정규화 (0~1, 1 = best)
    df_norm = df_raw.copy()
    for col in df_norm.columns:
        col_min = df_raw[col].min()
        col_max = df_raw[col].max()
        if col_max == col_min:
            df_norm[col] = 0.5
            continue
        norm = (df_raw[col] - col_min) / (col_max - col_min)
        higher_better = _METRIC_HIGHER_BETTER.get(col, not col.startswith("Camouflage"))
        df_norm[col] = norm if higher_better else (1 - norm)

    # 컬럼 이름 단축
    short_names     = [_short(c) for c in df_raw.columns]
    df_raw.columns  = short_names
    df_norm.columns = short_names

    # annotation: 실제 값 (소수점 2자리)
    annot = df_raw.copy().astype(object)
    for col in annot.columns:
        if col == "Clusters":
            annot[col] = annot[col].map(
                lambda v: str(int(v)) if pd.notna(v) else "N/A"
            )
        else:
            annot[col] = annot[col].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
            )

    # 히트맵
    n_rows = len(df_norm)
    n_cols = len(col_order)
    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.6), max(4, n_rows * 1.3 + 1.5)))

    sns.heatmap(
        df_norm,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 20, "weight": "bold"},
        cbar_kws={"label": "Normalized Score  (1 = best)", "shrink": 0.8},
    )

    ax.set_title(
        "CASS vs Baselines — UMAP Metric Comparison(CSE-CIC-IDS-2018)\n"
        "(green=better · red=worse · normalized per metric · actual values annotated)",
        fontsize=20, fontweight="bold", pad=14,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Feature Selection Method", fontsize=20)
    ax.tick_params(axis="x", rotation=0,  labelsize=20)
    ax.tick_params(axis="y", rotation=0,  labelsize=20)


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


# ── 실행 ──────────────────────────────────────────────────────────────────────

TARGETS = [
    {
        "csv":   "results/cicids2018/logs/comparison_metrics.csv",
        "out":   "results/cicids2018/figures/comparison_heatmap.png",
        "label": "CICIDS2018",
    },
    {
        "csv":   "results/unsw_nb15/logs/comparison_metrics.csv",
        "out":   "results/unsw_nb15/figures/comparison_heatmap.png",
        "label": "UNSW-NB15",
    },
]

BASE = Path(__file__).resolve().parent

for t in TARGETS:
    csv_path = BASE / t["csv"]
    out_path = BASE / t["out"]

    if not csv_path.exists():
        print(f"[건너뜀] {csv_path} 없음")
        continue

    df = pd.read_csv(csv_path, index_col="group")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[생성 중] {t['label']} ...")
    plot_comparison_heatmap(df, out_path)

print("\n완료.")
