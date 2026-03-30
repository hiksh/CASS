"""
CASS — Analyzer (비교군별 UMAP 수치 지표 계산 + 히트맵)

8개 지표 (3 그룹):
  [Separability]  Silhouette, Centroid_to_Benign, Global_Mean_Dist
  [Camouflage]    Camouflage@{K}, Boundary_Mean
  [Cluster]       HDBSCAN_Noise_Rate, Cluster_Count, Cohesion_Dist

best 방향:
  ↑ 높을수록 좋음: Silhouette, Centroid_to_Benign, Global_Mean_Dist, Boundary_Mean
  ↓ 낮을수록 좋음: Camouflage@K, HDBSCAN_Noise_Rate, Cluster_Count, Cohesion_Dist

히트맵에서는 모든 지표를 0~1로 정규화하여 1 = best 방향으로 통일합니다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cuml.manifold import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

from .config import (
    UMAP_PARAMS, RANDOM_SEED, LOGS_DIR,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    CAMOUFLAGE_THRESHOLDS, MAX_CDIST_SAMPLE,
)

# ── 지표 방향 정의 ────────────────────────────────────────────────────────────
# True  = 높을수록 좋음 (히트맵 정규화 시 그대로 사용)
# False = 낮을수록 좋음 (히트맵 정규화 시 반전)
_METRIC_HIGHER_BETTER = {
    "Silhouette":         True,
    "Centroid_to_Benign": True,
    "Global_Mean_Dist":   True,
    "Boundary_Mean":      True,
    "HDBSCAN_Noise_Rate": False,
    "Cluster_Count":      False,
    "Cohesion_Dist":      False,
}
# Camouflage@K 는 동적으로 추가 — 항상 낮을수록 좋음

# 히트맵 컬럼 순서 (그룹별 구분선 위치와 연동)
_COL_GROUPS = {
    "Separability": ["Silhouette", "Centroid_to_Benign", "Global_Mean_Dist"],
    "Camouflage":   [],   # Camouflage@K + Boundary_Mean (동적 채움)
    "Cluster":      ["HDBSCAN_Noise_Rate", "Cluster_Count", "Cohesion_Dist"],
}


# ── 핵심 지표 계산 ────────────────────────────────────────────────────────────

def _compute_metrics(emb: np.ndarray, y: np.ndarray, rng) -> dict:
    """
    2D UMAP 임베딩에서 8개 수치 지표를 계산합니다.

    Args:
        emb : (n_samples, 2) UMAP embedding (numpy array)
        y   : (n_samples,) binary label (0=benign, 1=attack)
        rng : numpy default_rng

    Returns:
        dict — 지표명 → float
    """
    benign_pts = emb[y == 0]
    attack_pts = emb[y == 1]

    nan_keys = (
        ["Silhouette", "Centroid_to_Benign", "Global_Mean_Dist", "Boundary_Mean",
         "HDBSCAN_Noise_Rate", "Cluster_Count", "Cohesion_Dist"]
        + [f"Camouflage@{t}" for t in CAMOUFLAGE_THRESHOLDS]
    )
    if len(benign_pts) == 0 or len(attack_pts) == 0:
        return {k: np.nan for k in nan_keys}

    metrics = {}

    # ── 1. Silhouette Score ──────────────────────────────────────────────────
    n = len(emb)
    if n > 10_000:
        idx = rng.choice(n, 10_000, replace=False)
        sil = silhouette_score(emb[idx], y[idx], metric="euclidean")
    else:
        sil = silhouette_score(emb, y, metric="euclidean")
    metrics["Silhouette"] = round(float(sil), 4)

    # ── 2. Centroid_to_Benign ────────────────────────────────────────────────
    benign_c = benign_pts.mean(axis=0)
    attack_c = attack_pts.mean(axis=0)
    metrics["Centroid_to_Benign"] = round(float(np.linalg.norm(attack_c - benign_c)), 4)

    # ── 3. Global_Mean_Dist (서브샘플) ───────────────────────────────────────
    a_sub = attack_pts
    b_sub = benign_pts
    if len(a_sub) > MAX_CDIST_SAMPLE:
        a_sub = a_sub[rng.choice(len(a_sub), MAX_CDIST_SAMPLE, replace=False)]
    if len(b_sub) > MAX_CDIST_SAMPLE:
        b_sub = b_sub[rng.choice(len(b_sub), MAX_CDIST_SAMPLE, replace=False)]
    pw = cdist(a_sub, b_sub, metric="euclidean")
    metrics["Global_Mean_Dist"] = round(float(np.mean(pw)), 4)

    # ── 4. Boundary_Mean & 5. Camouflage@K ──────────────────────────────────
    nn_model = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(benign_pts)
    nn_dists = nn_model.kneighbors(attack_pts)[0].flatten()
    metrics["Boundary_Mean"] = round(float(np.mean(nn_dists)), 4)
    for t in CAMOUFLAGE_THRESHOLDS:
        metrics[f"Camouflage@{t}"] = round(float((nn_dists <= t).mean()), 4)

    # ── 6-8. HDBSCAN 기반 (attack points 대상) ──────────────────────────────
    clusterer  = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
    )
    atk_labels = clusterer.fit_predict(attack_pts)

    metrics["HDBSCAN_Noise_Rate"] = round(float((atk_labels == -1).mean()), 4)

    valid_mask = atk_labels != -1
    valid_labels = atk_labels[valid_mask]
    metrics["Cluster_Count"] = int(len(np.unique(valid_labels))) if valid_mask.sum() > 0 else 0

    if valid_mask.sum() > 0:
        coh_sum = 0.0
        for cid in np.unique(valid_labels):
            pts_c = attack_pts[atk_labels == cid]
            coh_sum += np.linalg.norm(pts_c - pts_c.mean(axis=0), axis=1).sum()
        metrics["Cohesion_Dist"] = round(coh_sum / valid_mask.sum(), 4)
    else:
        metrics["Cohesion_Dist"] = np.nan

    return metrics


# ── 비교군 분석 ───────────────────────────────────────────────────────────────

def analyze_comparison_groups(
    groups: dict,
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    precomputed_embeddings: dict = None,
    logs_dir=None,
) -> pd.DataFrame:
    """
    비교군별 Full UMAP 실행 → 8개 지표 계산.

    Args:
        groups                : {group_name: [feature, ...]} 딕셔너리
        X_scaled              : 전처리된 훈련 피처 행렬
        y                     : 레이블 (0=benign, 1=attack)
        feature_names         : 로드된 전체 피처 이름 목록
        precomputed_embeddings: {group_name: np.ndarray} — 사전 계산된 임베딩.
                                Stage 2.7에서 umar2024 UMAP이 이미 실행된 경우
                                'lit_umar2024' 키로 전달하면 중복 실행을 방지합니다.

    Returns:
        metrics_df: DataFrame (index=group, cols=8 metrics + n_features)
    """
    feat_list = list(feature_names)
    rng = np.random.default_rng(RANDOM_SEED)
    precomputed_embeddings = precomputed_embeddings or {}

    SEP  = "=" * 65
    SEP2 = "-" * 65
    print(f"\n{SEP}")
    print(f"[Analyze] 비교군별 UMAP 수치 분석  ({len(groups)}개 그룹 × 8개 지표)")
    print(SEP)

    rows = []
    for i, (name, feats) in enumerate(groups.items(), 1):
        print(f"\n  [{i}/{len(groups)}] {name}  ({len(feats) if feats else 0}개 피처)")

        if not feats:
            print(f"    [건너뜀] 유효한 피처 없음")
            continue

        # 피처 추출
        idx   = [feat_list.index(f) for f in feats if f in feat_list]
        X_sub = X_scaled[:, idx]

        # Full UMAP — 사전 계산된 임베딩이 있으면 재사용
        if name in precomputed_embeddings:
            emb = precomputed_embeddings[name]
            print(f"    UMAP 임베딩 재사용 (Stage 2.7 사전 계산, 중복 실행 방지)")
        else:
            print(f"    UMAP 실행 중 (n_neighbors={UMAP_PARAMS['n_neighbors']}) ...")
            reducer = UMAP(**UMAP_PARAMS)
            emb     = np.asarray(reducer.fit_transform(X_sub))

        # 8개 지표 계산
        print(f"    수치 분석 중 ...")
        m = _compute_metrics(emb, y, rng)
        m["group"]      = name
        m["n_features"] = len(feats)
        rows.append(m)

        print(
            f"    Silhouette={m.get('Silhouette', float('nan')):.4f}  "
            f"Centroid={m.get('Centroid_to_Benign', float('nan')):.4f}  "
            f"Global={m.get('Global_Mean_Dist', float('nan')):.4f}  "
            f"Noise={m.get('HDBSCAN_Noise_Rate', float('nan')):.4f}  "
            f"Clusters={m.get('Cluster_Count', '?')}"
        )

    if not rows:
        print("\n  유효한 결과가 없습니다.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("group")

    # ── 결과 요약 출력 ────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  분석 요약")
    print(SEP2)
    display_cols = ["n_features", "Silhouette", "Centroid_to_Benign",
                    "Global_Mean_Dist", "Boundary_Mean", "HDBSCAN_Noise_Rate",
                    "Cluster_Count", "Cohesion_Dist"]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[display_cols].to_string())

    # CSV 저장
    save_dir = logs_dir if logs_dir is not None else LOGS_DIR
    csv_path = save_dir / "comparison_metrics.csv"
    metrics_df.to_csv(csv_path)
    print(f"\n  지표 CSV 저장: {csv_path}")

    return metrics_df


# ── 히트맵 시각화 ─────────────────────────────────────────────────────────────

def plot_comparison_heatmap(metrics_df: pd.DataFrame, save_path) -> None:
    """
    비교군 × 8지표 정규화 히트맵을 생성합니다.

    - 각 열을 0~1로 정규화 (열 내 min/max 기준)
    - best 방향이 낮은 지표는 반전 → 모든 열에서 1 = best
    - annotation은 실제(비정규화) 값 표시
    - 그룹별 구분선으로 Separability / Camouflage / Cluster 구분
    """
    exclude   = {"n_features"}
    metric_cols = [c for c in metrics_df.columns if c not in exclude]
    df_raw   = metrics_df[metric_cols].copy().astype(float)

    # ── 컬럼 순서 정렬 ────────────────────────────────────────────────────────
    sep_cols = [c for c in ["Silhouette", "Centroid_to_Benign", "Global_Mean_Dist"]
                if c in df_raw.columns]
    cam_cols = ([c for c in df_raw.columns if c.startswith("Camouflage")]
                + [c for c in ["Boundary_Mean"] if c in df_raw.columns])
    clu_cols = [c for c in ["HDBSCAN_Noise_Rate", "Cluster_Count", "Cohesion_Dist"]
                if c in df_raw.columns]
    rest     = [c for c in df_raw.columns if c not in sep_cols + cam_cols + clu_cols]
    col_order = sep_cols + cam_cols + clu_cols + rest

    df_raw  = df_raw[col_order]

    # ── 정규화 (0~1, 1 = best) ───────────────────────────────────────────────
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

    # ── annotation: 실제 값 ──────────────────────────────────────────────────
    annot = df_raw.copy().astype(object)
    for col in annot.columns:
        if col == "Cluster_Count":
            annot[col] = annot[col].map(
                lambda v: str(int(v)) if pd.notna(v) else "N/A"
            )
        else:
            annot[col] = annot[col].map(
                lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"
            )

    # ── 히트맵 ───────────────────────────────────────────────────────────────
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
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"label": "Normalized Score  (1 = best)", "shrink": 0.8},
    )

    ax.set_title(
        "CASS vs Baselines — UMAP Metric Comparison\n"
        "(green=better · red=worse · normalized per metric · actual values annotated)",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Feature Selection Method", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)

    # 그룹 구분선 (Separability | Camouflage | Cluster)
    for boundary in [len(sep_cols), len(sep_cols) + len(cam_cols)]:
        if 0 < boundary < n_cols:
            ax.axvline(boundary, color="black", linewidth=2.0, linestyle="--")

    # 그룹 레이블 (상단)
    group_labels = [
        (len(sep_cols) / 2,               "Separability"),
        (len(sep_cols) + len(cam_cols) / 2, "Camouflage"),
        (len(sep_cols) + len(cam_cols) + len(clu_cols) / 2, "Cluster"),
    ]
    for x_pos, label in group_labels:
        if x_pos > 0:
            ax.text(
                x_pos, -0.6, label,
                ha="center", va="center", fontsize=9,
                style="italic", color="gray",
                transform=ax.get_xaxis_transform(),
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  히트맵 저장: {save_path}")
