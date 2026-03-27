"""
CASS — Main Pipeline
Cluster-Aware Feature Selection System 전체 파이프라인 실행기.

실행 예시:
  python main.py                             # greedy, top-20
  python main.py --mode random               # random 탐색
  python main.py --pilot                     # Pilot 검증 포함
  python main.py --mode random --n-subsets 100 --pilot
  python main.py --top-k 15
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from cuml.manifold import UMAP

from src.config import (
    FIGURES_DIR, LOGS_DIR, UMAP_PARAMS,
    RANDOM_SEED, SEARCH_MODE,
    PILOT_MIN_SPEARMAN,
)
from src.data_loader import load_dataset
from src.pre_filter import pre_filter
from src.search_algo import search, pilot_validation


# ── 시각화 헬퍼 ──────────────────────────────────────────────────────────────

def plot_umap_best(
    X_scaled: np.ndarray,
    y: np.ndarray,
    attack_step: np.ndarray,
    best_features: list,
    all_feature_names: list,
    save_path,
) -> None:
    """최적 피처 부분집합의 UMAP 시각화 (이진 + Kill Chain)."""
    feat_list = list(all_feature_names)
    idx = [feat_list.index(f) for f in best_features]
    X_sub = X_scaled[:, idx]

    print(f"\n[UMAP] 최적 {len(best_features)}개 피처 최종 시각화 ...")
    reducer = UMAP(**UMAP_PARAMS)
    emb = np.asarray(reducer.fit_transform(X_sub))

    STEP_COLORS = {
        "benign":       "#3498DB",
        "action":       "#E74C3C",
        "infection":    "#F39C12",
        "installation": "#2ECC71",
    }
    ALPHA, SIZE = 0.3, 0.8

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    feat_preview = ", ".join(best_features[:5]) + ("..." if len(best_features) > 5 else "")
    fig.suptitle(
        f"CASS Best Subset UMAP ({len(best_features)} features)\n{feat_preview}",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.scatter(emb[y == 0, 0], emb[y == 0, 1], c="#3498DB", alpha=ALPHA, s=SIZE,
               linewidths=0, rasterized=True)
    ax.scatter(emb[y == 1, 0], emb[y == 1, 1], c="#E74C3C", alpha=ALPHA, s=SIZE,
               linewidths=0, rasterized=True)
    ax.set_title("(A) Benign vs Attack", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(handles=[
        mpatches.Patch(color="#3498DB", label="Benign"),
        mpatches.Patch(color="#E74C3C", label="Attack"),
    ], markerscale=6, fontsize=10)
    ax.grid(alpha=0.2)

    ax = axes[1]
    legend_handles = []
    for step, color in STEP_COLORS.items():
        mask = y == 0 if step == "benign" else attack_step == step
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, alpha=ALPHA, s=SIZE,
                   linewidths=0, rasterized=True)
        legend_handles.append(mpatches.Patch(color=color, label=step))
    ax.set_title("(B) Kill Chain Stage", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(handles=legend_handles, markerscale=6, fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")

    emb_csv = save_path.parent / "best_subset_umap_embeddings.csv"
    emb_df = pd.DataFrame(emb, columns=["UMAP-1", "UMAP-2"])
    emb_df["attack_flag"] = y
    emb_df["attack_step"] = attack_step
    emb_df.to_csv(emb_csv, index=False)
    print(f"  임베딩 저장: {emb_csv}")


def plot_pilot(pilot_df: pd.DataFrame, spearman_r: float, save_path) -> None:
    """Pilot 검증: Fast vs Full Silhouette 산점도."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        pilot_df["fast_sil"], pilot_df["full_sil"],
        c=pilot_df["n_features"], cmap="viridis",
        alpha=0.8, s=70, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Number of Features")

    lims = [
        min(pilot_df["fast_sil"].min(), pilot_df["full_sil"].min()) - 0.05,
        max(pilot_df["fast_sil"].max(), pilot_df["full_sil"].max()) + 0.05,
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, linewidth=1.2, label="y = x")

    ax.set_xlabel("Fast Silhouette (n_neighbors=30)", fontsize=11)
    ax.set_ylabel("Full Silhouette (n_neighbors=150)", fontsize=11)
    ax.set_title(
        f"Pilot 검증: Fast ↔ Full Silhouette\nSpearman r = {spearman_r:.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


def plot_two_phase(results_df: pd.DataFrame, mode: str, save_path) -> None:
    """Fast vs Full Silhouette 비교 플롯."""
    has_full = "full_sil" in results_df.columns and results_df["full_sil"].notna().any()
    x_col = "step" if mode == "greedy" else "subset_id"

    fig, ax = plt.subplots(figsize=(11, 5))

    if x_col in results_df.columns:
        x_all = results_df[x_col].values
    else:
        x_all = np.arange(len(results_df))

    ax.plot(x_all, results_df["fast_sil"].values, "o--",
            color="steelblue", alpha=0.5, markersize=4, label="Fast Sil (스크리닝)")

    if has_full:
        top_idx = results_df.dropna(subset=["full_sil"]).index
        x_top   = x_all[top_idx] if x_col in results_df.columns else top_idx
        ax.scatter(x_top, results_df.loc[top_idx, "full_sil"],
                   color="tomato", s=80, zorder=5, label="Full Sil (재평가)")

    xlabel = "Greedy Step" if mode == "greedy" else "Subset ID (fast_sil 정렬)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("2단계 스크리닝: Fast → Full Silhouette", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────

def main(args) -> None:
    for d in [FIGURES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CASS — Cluster-Aware Feature Selection System")
    print(f"  Mode   : {args.mode}")
    print(f"  Top-K  : {args.top_k}")
    print(f"  Pilot  : {'ON' if args.pilot else 'OFF'}")
    print("=" * 65)

    # ── 1. 데이터 로드 & 전처리 ─────────────────────────────────────────────
    print("\n[1/4] 데이터 로드 및 전처리")
    X_scaled, y, attack_step, feature_names, scaler, df = load_dataset(
        use_udbb=True, save_processed=True,
    )
    print(f"  피처 수  : {len(feature_names)}")
    print(f"  Benign   : {int(np.sum(y == 0)):,}")
    print(f"  Attack   : {int(np.sum(y == 1)):,}")

    # ── 2. Pre-filter ────────────────────────────────────────────────────────
    print("\n[2/4] Pre-filter")
    top_features, filter_summary = pre_filter(X_scaled, y, feature_names, n_top=args.top_k)
    filter_summary.to_csv(LOGS_DIR / "pre_filter_ranking.csv", index=False)
    print(f"\n  후보 피처 ({len(top_features)}개): {top_features}")

    # ── 2.5 [선택] Pilot 검증 ───────────────────────────────────────────────
    if args.pilot:
        print("\n[Pilot] Fast ↔ Full Silhouette 상관 사전 검증")
        spearman_r, pilot_df = pilot_validation(
            X_scaled, y, top_features, feature_names
        )
        pilot_df.to_csv(LOGS_DIR / "pilot_validation.csv", index=False)
        plot_pilot(pilot_df, spearman_r, FIGURES_DIR / "pilot_fast_vs_full.png")

        if not np.isnan(spearman_r) and spearman_r < PILOT_MIN_SPEARMAN:
            print(f"\n  [경고] Spearman r={spearman_r:.4f} < {PILOT_MIN_SPEARMAN}")
            print("  UMAP_PARAMS_FAST 조정 후 재실행을 권장합니다.")

    # ── 3. 2단계 스크리닝 탐색 ──────────────────────────────────────────────
    print(f"\n[3/4] 2단계 스크리닝 (mode={args.mode})")
    best_subset, results_df = search(
        X_scaled, y, top_features, feature_names,
        mode=args.mode,
        n_subsets=args.n_subsets,
    )
    results_df.to_csv(LOGS_DIR / f"search_results_{args.mode}.csv", index=False)
    print(f"  탐색 결과 저장: {LOGS_DIR / f'search_results_{args.mode}.csv'}")

    # ── 4. 시각화 ────────────────────────────────────────────────────────────
    print("\n[4/4] 시각화")
    plot_two_phase(results_df, args.mode, FIGURES_DIR / "two_phase_screening.png")

    if best_subset:
        plot_umap_best(
            X_scaled, y, attack_step, best_subset, feature_names,
            FIGURES_DIR / "umap_best_subset.png",
        )

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("CASS Pipeline 완료")
    print(SEP)
    if best_subset:
        best_full = results_df["full_sil"].max()
        print(f"  Full Silhouette  : {best_full:.4f}")
        print(f"  최적 부분집합 ({len(best_subset)}개):")
        for f in best_subset:
            print(f"    - {f}")
    print(f"  결과 로그  : {LOGS_DIR}")
    print(f"  시각화     : {FIGURES_DIR}")
    print(SEP)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CASS — Cluster-Aware Feature Selection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["greedy", "random"], default=SEARCH_MODE,
    )
    parser.add_argument(
        "--top-k", type=int, default=20, dest="top_k",
        help="Pre-filter 후 유지할 피처 수",
    )
    parser.add_argument(
        "--n-subsets", type=int, default=None, dest="n_subsets",
        help="random 모드에서 평가할 부분집합 수",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="실행 전 Fast↔Full Silhouette 상관 검증 수행",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
