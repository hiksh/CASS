"""
단일 히트맵: 행=방법(5), 열=CIC-BM / CIC-CR | UNSW-BM / UNSW-CR (4)
2열 논문 단 폭(~3.5in) 배치 기준으로 figsize 설정.

실행:
  python regen_heatmaps_compact.py
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

# ── 설정 ───────────────────────────────────────────────────────────────────────
_METRICS = ["Boundary_Mean", "Camouflage@1.0"]

_HIGHER_BETTER = {
    "Boundary_Mean":  True,
    "Camouflage@1.0": False,
}

# 열 이름: (데이터셋 키, 지표)
_COL_DEFS = [
    ("cic",  "Boundary_Mean"),
    ("cic",  "Camouflage@1.0"),
    ("unsw", "Boundary_Mean"),
    ("unsw", "Camouflage@1.0"),
]

_COL_LABELS = {
    ("cic",  "Boundary_Mean"):  "CIC\nBM",
    ("cic",  "Camouflage@1.0"): "CIC\nCR",
    ("unsw", "Boundary_Mean"):  "UNSW\nBM",
    ("unsw", "Camouflage@1.0"): "UNSW\nCR",
}

# 공통 방법 순서 (lit_*는 마지막에 병합)
_METHOD_ORDER = ["cass", "anova", "extratrees", "random"]
_LIT_ROW = "lit_*"   # CIC: lit_umar2024 / UNSW: lit_yin2023


def _find_lit(df: pd.DataFrame) -> str | None:
    for idx in df.index:
        if idx.startswith("lit_"):
            return idx
    return None


def _build_tables(df_cic: pd.DataFrame, df_unsw: pd.DataFrame):
    """
    raw_df  : 행=방법(5), 열=4개 (실제값)
    norm_df : 같은 shape, 열별 0~1 정규화 (1=best)
    """
    cic_lit  = _find_lit(df_cic)
    unsw_lit = _find_lit(df_unsw)
    methods  = _METHOD_ORDER + [_LIT_ROW]

    raw = {}
    for method in methods:
        row = {}
        for ds, met in _COL_DEFS:
            df_src = df_cic if ds == "cic" else df_unsw
            if method == _LIT_ROW:
                src_key = cic_lit if ds == "cic" else unsw_lit
            else:
                src_key = method
            if src_key is None or src_key not in df_src.index or met not in df_src.columns:
                row[(ds, met)] = np.nan
            else:
                row[(ds, met)] = df_src.loc[src_key, met]
        raw[method] = row

    raw_df = pd.DataFrame(raw).T   # (method × col_def)
    raw_df.columns = pd.MultiIndex.from_tuples(raw_df.columns)
    raw_df = raw_df.astype(float)

    # 열별 정규화
    norm_df = raw_df.copy()
    for col in norm_df.columns:
        _, met = col
        col_min, col_max = raw_df[col].min(), raw_df[col].max()
        if col_max == col_min:
            norm_df[col] = 0.5
        else:
            norm = (raw_df[col] - col_min) / (col_max - col_min)
            norm_df[col] = norm if _HIGHER_BETTER.get(met, True) else (1 - norm)

    return raw_df, norm_df


def plot_single_heatmap(
    df_cic: pd.DataFrame,
    df_unsw: pd.DataFrame,
    save_path: Path,
) -> None:
    raw_df, norm_df = _build_tables(df_cic, df_unsw)

    # ── 행↔열 전치: 행=지표(4), 열=방법(5) ────────────────────────────────────
    row_labels = [_COL_LABELS[c] for c in raw_df.columns]   # 4개 지표
    col_labels = raw_df.index.tolist()                       # 5개 방법

    raw_df  = raw_df.T.copy()
    norm_df = norm_df.T.copy()
    raw_df.index  = row_labels
    norm_df.index = row_labels
    raw_df.columns  = col_labels
    norm_df.columns = col_labels

    # annotation: 실제값
    annot = raw_df.copy().astype(object)
    for col in annot.columns:
        annot[col] = annot[col].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )

    # ── 플롯 (단 폭 2× 스케일: 4행×5열, 가로형) ──────────────────────────────
    n_rows, n_cols = norm_df.shape   # 4 × 5
    fig, ax = plt.subplots(figsize=(n_cols * 1.4 + 1.5, n_rows * 1.4 + 1.8))

    sns.heatmap(
        norm_df,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
        cbar=False,
    )

    # ── 가로 colorbar (하단) ──────────────────────────────────────────────────
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        location="bottom", shrink=0.6, pad=0.18)
    cbar.set_label("Normalized Score  (1 = best)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # ── CIC / UNSW 행 구분선 & 그룹 레이블 ───────────────────────────────────
    ax.axhline(2, color="black", linewidth=2.5)

    ax.annotate(
        "CIC-IDS-2018",
        xy=(-0.22, 1 / n_rows), xycoords="axes fraction",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#1a4e8a", rotation=90,
    )
    ax.annotate(
        "UNSW-NB15",
        xy=(-0.22, 3 / n_rows), xycoords="axes fraction",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#7a1a1a", rotation=90,
    )

    # ax.set_title(
    #     "CASS vs Baselines — UMAP Metric Comparison\n"
    #     "(green=better · red=worse · normalized per row · actual values annotated)",
    #     fontsize=13, fontweight="bold", pad=14,
    # )
    ax.set_xlabel("Feature Selection Method", fontsize=13)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=13)
    ax.tick_params(axis="y", rotation=0, labelsize=13)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


# ── 실행 ──────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent

CSV_CIC  = BASE / "results/cicids2018/logs/comparison_metrics.csv"
CSV_UNSW = BASE / "results/unsw_nb15/logs/comparison_metrics.csv"
OUT_PATH = BASE / "results/figures/comparison_heatmap_compact.png"

missing = [p for p in [CSV_CIC, CSV_UNSW] if not p.exists()]
if missing:
    for p in missing:
        print(f"[오류] 파일 없음: {p}")
else:
    df_cic  = pd.read_csv(CSV_CIC,  index_col="group")
    df_unsw = pd.read_csv(CSV_UNSW, index_col="group")
    print("[생성 중] 5×4 single heatmap ...")
    plot_single_heatmap(df_cic, df_unsw, OUT_PATH)
    print("\n완료.")
