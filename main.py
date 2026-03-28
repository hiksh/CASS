"""
CASS — Main Pipeline
Cluster-Aware Feature Selection System 전체 파이프라인 실행기.

실행 예시:
  python main.py                             # greedy, top-30
  python main.py --mode random               # random 탐색
  python main.py --pilot                     # Pilot 검증 포함
  python main.py --mode random --n-subsets 100 --pilot
  python main.py --top-k 15
  python main.py --export                    # 비교군 CSV 내보내기
  python main.py --analyze                   # 8지표 히트맵 분석
  python main.py --export --analyze          # 전체 실행
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr
from cuml.manifold import UMAP

from src.config import (
    UMAP_PARAMS, UMAP_PARAMS_FAST,
    RANDOM_SEED, SEARCH_MODE,
    PILOT_MIN_SPEARMAN, N_RANDOM_BASELINE,
    get_dataset_config,
)
from src.data_loader import load_dataset
from src.pre_filter import pre_filter
from src.search_algo import (
    search, pilot_validation, pilot_validation_with_retry, compute_reference_camouflage,
)
from src.exporter import export_comparison_sets, build_comparison_groups
from src.analyzer import analyze_comparison_groups, plot_comparison_heatmap


# ── 로그 헬퍼 ────────────────────────────────────────────────────────────────

_SEP  = "=" * 65
_SEP2 = "-" * 65

def _header(tag: str, title: str) -> None:
    print(f"\n{_SEP}")
    print(f"{tag}  {title}")
    print(_SEP)

def _sub(msg: str) -> None:
    print(f"  {msg}")


# ── 시각화 헬퍼 ──────────────────────────────────────────────────────────────

def plot_umap_best(
    X_scaled: np.ndarray,
    y: np.ndarray,
    attack_step: np.ndarray,
    best_features: list,
    all_feature_names: list,
    save_path,
    step_colors: dict = None,
) -> None:
    """최적 피처 부분집합의 UMAP 시각화 (이진 + Kill Chain)."""
    feat_list = list(all_feature_names)
    idx = [feat_list.index(f) for f in best_features]
    X_sub = X_scaled[:, idx]

    print(f"\n[UMAP] 최적 {len(best_features)}개 피처 최종 시각화 ...")
    reducer = UMAP(**UMAP_PARAMS)
    emb = np.asarray(reducer.fit_transform(X_sub))

    STEP_COLORS = step_colors or {
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
    _sub(f"저장: {save_path}")

    emb_csv = save_path.parent / "best_subset_umap_embeddings.csv"
    emb_df = pd.DataFrame(emb, columns=["UMAP-1", "UMAP-2"])
    emb_df["attack_flag"] = y
    emb_df["attack_step"] = attack_step
    emb_df.to_csv(emb_csv, index=False)
    _sub(f"임베딩 저장: {emb_csv}")


def plot_pilot(pilot_df: pd.DataFrame, spearman_r: float, save_path) -> None:
    """Pilot 검증: Fast vs Full Boundary_Mean 산점도."""
    fast_n = UMAP_PARAMS_FAST["n_neighbors"]
    full_n = UMAP_PARAMS["n_neighbors"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        pilot_df["fast_bm"], pilot_df["full_bm"],
        c=pilot_df["n_features"], cmap="viridis",
        alpha=0.8, s=70, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Number of Features")

    lims = [
        min(pilot_df["fast_bm"].min(), pilot_df["full_bm"].min()) * 0.95,
        max(pilot_df["fast_bm"].max(), pilot_df["full_bm"].max()) * 1.05,
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, linewidth=1.2, label="y = x")

    ax.set_xlabel(f"Fast Boundary_Mean (n_neighbors={fast_n})", fontsize=11)
    ax.set_ylabel(f"Full Boundary_Mean (n_neighbors={full_n})", fontsize=11)
    ax.set_title(
        f"Pilot 검증: Fast ↔ Full Boundary_Mean\nSpearman r = {spearman_r:.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    _sub(f"저장: {save_path}")


def plot_two_phase(
    results_df: pd.DataFrame,
    mode: str,
    save_path,
    best_features: list = None,
) -> None:
    """
    Fast vs Full Boundary_Mean 비교 플롯.

    Top-K 포인트를 silhouette 제약 통과(초록)/탈락(주황)으로 구분하고,
    최종 선택된 서브셋(best_features)에 금색 별 마커를 표시합니다.
    """
    from src.config import MIN_SILHOUETTE
    has_full = "boundary_mean" in results_df.columns and results_df["boundary_mean"].notna().any()
    x_col = "step" if mode == "greedy" else "subset_id"

    fig, ax = plt.subplots(figsize=(11, 5))

    x_all = results_df[x_col].values if x_col in results_df.columns else np.arange(len(results_df))

    ax.plot(x_all, results_df["fast_bm"].values, "o--",
            color="steelblue", alpha=0.5, markersize=4, label="Fast BM (스크리닝)")

    if has_full:
        top_df  = results_df.dropna(subset=["boundary_mean"])
        top_idx = top_df.index
        x_top   = x_all[top_idx] if x_col in results_df.columns else top_idx

        passed = top_df["full_sil"] > MIN_SILHOUETTE
        if passed.any():
            ax.scatter(
                x_top[passed.values], top_df.loc[passed, "boundary_mean"],
                color="#2ECC71", s=90, zorder=5,
                label=f"Full BM — silhouette > {MIN_SILHOUETTE} 통과",
            )
        if (~passed).any():
            ax.scatter(
                x_top[(~passed).values], top_df.loc[~passed, "boundary_mean"],
                color="#E67E22", s=90, zorder=5, marker="X",
                label=f"Full BM — silhouette ≤ {MIN_SILHOUETTE} 탈락",
            )

    # 최종 선택 포인트 (금색 별)
    if best_features is not None and has_full:
        best_set = frozenset(best_features)
        for ridx, row in results_df.iterrows():
            feats = row.get("features")
            if feats is not None and frozenset(feats) == best_set and pd.notna(row.get("boundary_mean")):
                bx = x_all[ridx] if x_col in results_df.columns else ridx
                ax.scatter(
                    bx, row["boundary_mean"],
                    color="gold", s=250, zorder=10, marker="*",
                    edgecolors="black", linewidths=0.8,
                    label="최종 선택 (best)",
                )
                break

    xlabel = "Greedy Step" if mode == "greedy" else "Subset ID (fast_bm 정렬)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Boundary Mean")
    ax.set_title(
        f"2단계 스크리닝: Fast → Full Boundary Mean\n제약: Silhouette > {MIN_SILHOUETTE}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    _sub(f"저장: {save_path}")


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────

def main(args) -> None:
    # ── 데이터셋 설정 로드 ───────────────────────────────────────────────────
    ds = get_dataset_config(args.dataset)
    FIGURES_DIR = ds["figures_dir"]
    LOGS_DIR    = ds["logs_dir"]
    EXPORTS_DIR = ds["exports_dir"]

    for d in [FIGURES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    total_stages = 4 + (1 if args.export else 0) + (1 if args.analyze else 0)

    print(_SEP)
    print("CASS — Cluster-Aware Feature Selection System")
    print(_SEP2)
    print(f"  Dataset  : {args.dataset}")
    print(f"  Mode     : {args.mode}")
    print(f"  Top-K    : {args.top_k}")
    print(f"  Pilot    : {'ON' if args.pilot else 'OFF'}")
    print(f"  Export   : {'ON' if args.export else 'OFF'}")
    print(f"  Analyze  : {'ON' if args.analyze else 'OFF'}")
    print(f"  Stages   : {total_stages}")
    print(_SEP)

    # ── 1. 데이터 로드 & 전처리 ─────────────────────────────────────────────
    _header(f"[1/{total_stages}]", "데이터 로드 및 전처리")
    X_scaled, y, attack_step, feature_names, scaler, clip_params, df = load_dataset(
        csv_path=ds["train_file"],
        use_udbb=True,
        save_processed=(args.dataset == "cicids2018"),
        all_features=ds["all_features"],
        udbb_counts=ds["udbb_counts"],
        log_features=ds["log_features"],
    )
    n_benign = int(np.sum(y == 0))
    n_attack = int(np.sum(y == 1))
    _sub(f"전체 샘플  : {len(y):,}행")
    _sub(f"피처 수    : {len(feature_names)}")
    _sub(f"Benign     : {n_benign:,}  ({n_benign/len(y)*100:.1f}%)")
    _sub(f"Attack     : {n_attack:,}  ({n_attack/len(y)*100:.1f}%)")
    attack_steps_summary = {
        s: int((attack_step == s).sum())
        for s in ds["udbb_counts"] if s != "benign"
    }
    for step, cnt in attack_steps_summary.items():
        _sub(f"  └ {step:<14}: {cnt:,}")

    # ── 2. Pre-filter ────────────────────────────────────────────────────────
    _header(f"[2/{total_stages}]", f"Pre-filter  ({len(feature_names)}개 → 상위 {args.top_k}개)")
    top_features, filter_summary = pre_filter(X_scaled, y, feature_names, n_top=args.top_k)
    filter_summary.to_csv(LOGS_DIR / "pre_filter_ranking.csv", index=False)
    _sub(f"결과 저장  : {LOGS_DIR / 'pre_filter_ranking.csv'}")
    _sub(f"선발 피처 ({len(top_features)}개):")
    for i, f in enumerate(top_features, 1):
        _sub(f"  {i:>2}. {f}")

    # ── 2.5 [선택] Pilot 검증 ───────────────────────────────────────────────
    if args.pilot:
        _header(f"[Pilot]", "Fast ↔ Full Boundary_Mean 상관 검증")
        spearman_r, pilot_df = pilot_validation_with_retry(
            X_scaled, y, top_features, feature_names,
        )
        pilot_df.to_csv(LOGS_DIR / "pilot_validation.csv", index=False)
        plot_pilot(pilot_df, spearman_r, FIGURES_DIR / "pilot_fast_vs_full.png")

        if not np.isnan(spearman_r) and spearman_r < PILOT_MIN_SPEARMAN:
            _sub(f"[경고] 최대 재시도 후에도 Spearman r={spearman_r:.4f} < {PILOT_MIN_SPEARMAN}")
            _sub("Fast 스크리닝 신뢰도가 낮을 수 있습니다. 결과 해석에 주의하세요.")

    # ── 2.7 Reference Camouflage 참고값 계산 (literature baseline, 비교용) ──
    lit_baselines = ds["literature_baselines"]
    ref_name      = next(iter(lit_baselines), None)   # 첫 번째 baseline 사용
    ref_embedding = None

    if ref_name:
        _header(f"[2.7/{total_stages}]", f"Reference Camouflage 참고값 계산 ({ref_name}, 비교용)")
        ref_features = [f for f in lit_baselines[ref_name] if f in list(feature_names)]
        if ref_features:
            ref_cam, ref_embedding = compute_reference_camouflage(
                X_scaled, y, feature_names, ref_features, ref_name=ref_name
            )
            _sub(f"{ref_name} Camouflage 참고값 : {ref_cam:.4f}  (비교 기준, 제약으로 사용되지 않음)")
            pd.DataFrame([{
                "reference":  ref_name,
                "n_features": len(ref_features),
                "ref_cam":    ref_cam,
            }]).to_csv(LOGS_DIR / "reference_camouflage.csv", index=False)
            _sub(f"참고값 저장: {LOGS_DIR / 'reference_camouflage.csv'}")
        else:
            _sub(f"[경고] {ref_name} 피처가 데이터셋에 없습니다.")
    else:
        _header(f"[2.7/{total_stages}]", "Reference Camouflage — literature baseline 없음, skip")

    # ── 3. 2단계 스크리닝 탐색 ──────────────────────────────────────────────
    _header(f"[3/{total_stages}]", f"2단계 스크리닝  (mode={args.mode})")
    _sub(f"후보 피처 {len(top_features)}개 → Greedy/Random 탐색 시작")
    best_subset, results_df = search(
        X_scaled, y, top_features, feature_names,
        mode=args.mode,
        n_subsets=args.n_subsets,
    )
    results_csv = LOGS_DIR / f"search_results_{args.mode}.csv"
    results_df.to_csv(results_csv, index=False)
    _sub(f"탐색 결과 저장: {results_csv}")

    if best_subset:
        from src.config import MIN_SILHOUETTE as _MIN_SIL
        _valid = results_df.dropna(subset=["full_sil", "boundary_mean"])
        _survived = _valid[_valid["full_sil"] > _MIN_SIL]
        if _survived.empty:
            _survived = _valid
        best_bm = _survived["boundary_mean"].max()
        _sub(f"최적 부분집합  ({len(best_subset)}개 피처 | boundary_mean={best_bm:.4f}):")
        for f in best_subset:
            _sub(f"  - {f}")

    # ── 4. 시각화 ────────────────────────────────────────────────────────────
    _header(f"[4/{total_stages}]", "시각화")
    plot_two_phase(
        results_df, args.mode, FIGURES_DIR / "two_phase_screening.png",
        best_features=best_subset if best_subset else None,
    )

    if best_subset:
        plot_umap_best(
            X_scaled, y, attack_step, best_subset, feature_names,
            FIGURES_DIR / "umap_best_subset.png",
            step_colors=ds["step_colors"],
        )

    # ── 공유 비교군 (export / analyze 양쪽에서 사용) ─────────────────────────
    groups = None
    if (args.export or args.analyze) and best_subset:
        groups = build_comparison_groups(
            best_subset, filter_summary, feature_names, N_RANDOM_BASELINE,
            literature_baselines=lit_baselines,
        )

    # ── 5. [선택] Export ─────────────────────────────────────────────────────
    stage = 5
    if args.export:
        if not best_subset:
            print(f"\n[{stage}/{total_stages}] [경고] 최적 부분집합이 없어 Export를 건너뜁니다.")
        else:
            _header(f"[{stage}/{total_stages}]", "비교군 Export")
            export_comparison_sets(
                X_scaled, y, attack_step, feature_names,
                best_subset, filter_summary, scaler, clip_params,
                export_dir=EXPORTS_DIR,
                test_file=ds["test_file"],
                literature_baselines=lit_baselines,
                log_features=ds["log_features"],
            )
        stage += 1

    # ── 6. [선택] Analyze ────────────────────────────────────────────────────
    if args.analyze:
        if not best_subset:
            print(f"\n[{stage}/{total_stages}] [경고] 최적 부분집합이 없어 Analyze를 건너뜁니다.")
        else:
            _header(f"[{stage}/{total_stages}]", "UMAP 수치 분석 및 히트맵")
            precomputed_embeddings = {}
            if ref_embedding is not None and ref_name:
                precomputed_embeddings[f"lit_{ref_name}"] = ref_embedding
            metrics_df = analyze_comparison_groups(
                groups, X_scaled, y, feature_names,
                precomputed_embeddings=precomputed_embeddings,
                logs_dir=LOGS_DIR,
            )
            if not metrics_df.empty:
                plot_comparison_heatmap(
                    metrics_df,
                    FIGURES_DIR / "comparison_heatmap.png",
                )

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("CASS Pipeline 완료")
    print(_SEP2)
    if best_subset:
        from src.config import MIN_SILHOUETTE
        valid_rows = results_df.dropna(subset=["full_sil", "boundary_mean"])
        survived   = valid_rows[valid_rows["full_sil"] > MIN_SILHOUETTE]
        if survived.empty:
            survived = valid_rows
        best_idx  = survived["boundary_mean"].idxmax()
        best_bm   = survived.loc[best_idx, "boundary_mean"]
        best_full = survived.loc[best_idx, "full_sil"]
        best_cam  = survived.loc[best_idx, "camouflage"] if "camouflage" in survived.columns else float("nan")
        _sub(f"Boundary_Mean    : {best_bm:.4f}  ← 주 목적함수")
        _sub(f"Full Silhouette  : {best_full:.4f}  (제약: > {MIN_SILHOUETTE})")
        _sub(f"Camouflage@1.0   : {best_cam:.4f}  (참고값)")
        _sub(f"최적 피처 수      : {len(best_subset)}개")
    _sub(f"로그 저장 위치    : {LOGS_DIR}")
    _sub(f"시각화 저장 위치  : {FIGURES_DIR}")
    if args.export:
        _sub(f"Export 저장 위치  : {EXPORTS_DIR}")
    if args.analyze:
        _sub(f"히트맵 저장 위치  : {FIGURES_DIR / 'comparison_heatmap.png'}")
    print(_SEP)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CASS — Cluster-Aware Feature Selection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["cicids2018", "unsw_nb15"], default="cicids2018",
        help="사용할 데이터셋",
    )
    parser.add_argument(
        "--mode", choices=["greedy", "random"], default=SEARCH_MODE,
    )
    parser.add_argument(
        "--top-k", type=int, default=30, dest="top_k",
        help="Pre-filter 후 유지할 피처 수",
    )
    parser.add_argument(
        "--n-subsets", type=int, default=None, dest="n_subsets",
        help="random 모드에서 평가할 부분집합 수",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="실행 전 Fast↔Full Boundary_Mean 상관 검증 수행",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="비교군별 train/test CSV를 results/exports/에 저장",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="비교군별 8개 UMAP 지표 계산 및 히트맵 저장",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
