"""
comparison_umap.py — 비교군 UMAP 시각화 (anova_umap.py 구조 동일)

대상:
  cicids2018 : random, extratrees
  unsw_nb15  : anova, extratrees, lit_yin2023, random

data/raw/ 파일만 사용 (exports/ 불필요)
출력: results/comparison_umap/umap_{dataset}_{group}{n_feat}.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "comparison_umap"

# ── UMAP 파라미터 (anova_umap.py 동일) ───────────────────────────────────────
UMAP_PARAMS = dict(
    n_neighbors=150,
    min_dist=0.01,
    metric="manhattan",
    n_components=2,
    init="spectral",
    random_state=42,
    n_jobs=-1,
)

ALPHA, SIZE = 0.3, 0.8
FLAG_COLORS = {0: "#3498DB", 1: "#E74C3C"}
FLAG_LABELS = {0: "Benign",  1: "Attack"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CICIDS2018 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CICIDS_LOG = [
    "tot fwd pkts",      "tot bwd pkts",
    "totlen fwd pkts",   "totlen bwd pkts",
    "fwd pkt len max",   "fwd pkt len min",   "fwd pkt len mean",  "fwd pkt len std",
    "bwd pkt len max",   "bwd pkt len min",   "bwd pkt len mean",  "bwd pkt len std",
    "pkt len min",       "pkt len max",       "pkt len mean",      "pkt len std",
    "pkt len var",       "pkt size avg",      "fwd seg size avg",  "bwd seg size avg",
    "flow byts/s",       "flow pkts/s",       "fwd pkts/s",        "bwd pkts/s",
    "flow iat mean",     "flow iat std",      "flow iat max",
    "fwd iat tot",       "fwd iat mean",      "fwd iat std",       "fwd iat max",
    "bwd iat tot",       "bwd iat mean",      "bwd iat std",       "bwd iat max",
    "fwd header len",    "bwd header len",
    "init fwd win byts", "init bwd win byts",
    "subflow fwd pkts",  "subflow fwd byts",  "subflow bwd pkts",  "subflow bwd byts",
    "active mean",       "active std",        "active max",
    "idle mean",         "idle std",          "idle max",
]

CICIDS_UDBB = {
    "benign":       60_000,
    "action":       20_000,
    "infection":    20_000,
    "installation": 20_000,
}

CICIDS_STEP_COLORS = {
    "benign":       "#3498DB",
    "action":       "#E74C3C",
    "infection":    "#F39C12",
    "installation": "#2ECC71",
}

CICIDS_GROUPS = {
    "random": [
        "bwd pkt len mean", "bwd pkt len min",  "bwd pkts/s",
        "dst port",         "flow duration",    "fwd iat mean",
        "fwd iat std",      "fwd pkt len max",  "fwd pkt len min",
        "fwd seg size avg", "init bwd win byts","pkt len max",
        "pkt len min",      "pkt size avg",     "protocol",
        "totlen fwd pkts",
    ],
    "extratrees": [
        "dst port",         "fwd seg size min", "fwd iat mean",
        "fwd pkts/s",       "flow iat max",     "fwd iat tot",
        "flow iat mean",    "fwd iat max",      "fwd seg size avg",
        "flow pkts/s",      "fwd pkt len mean", "init bwd win byts",
        "totlen fwd pkts",  "bwd pkts/s",       "fwd iat std",
        "flow byts/s",
    ],
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNSW-NB15 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNSW_LOG = [
    "dur",
    "spkts",        "dpkts",
    "sbytes",       "dbytes",
    "rate",
    "sload",        "dload",
    "sinpkt",       "dinpkt",
    "sjit",         "djit",
    "swin",         "stcpb",        "dtcpb",        "dwin",
    "tcprtt",       "synack",       "ackdat",
    "smean",        "dmean",
    "trans_depth",  "response_body_len",
    "ct_srv_src",   "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_src_ltm",   "ct_srv_dst",
]

UNSW_UDBB = {
    "benign":          30_000,
    "reconnaissance":  10_000,
    "action":          10_000,
    "infection":       10_000,
}

UNSW_STEP_COLORS = {
    "benign":         "#3498DB",
    "action":         "#E74C3C",
    "infection":      "#F39C12",
    "reconnaissance": "#9B59B6",
}

UNSW_GROUPS = {
    "anova": [
        "sttl",             "ct_state_ttl",     "dload",
        "dmean",            "dbytes",           "dpkts",
        "spkts",            "ct_dst_sport_ltm", "swin",
        "sbytes",           "dwin",             "stcpb",
        "dtcpb",            "sload",            "sloss",
        "rate",             "ct_src_dport_ltm",
    ],
    "extratrees": [
        "sttl",             "ct_state_ttl",     "dload",
        "dttl",             "ct_srv_dst",       "dmean",
        "sbytes",           "ct_dst_src_ltm",   "dbytes",
        "smean",            "ct_srv_src",       "rate",
        "sload",            "dpkts",            "swin",
        "sinpkt",           "is_sm_ips_ports",
    ],
    "lit_yin2023": [
        "dur",     "spkts",   "dpkts",  "sbytes",        "dbytes",
        "rate",    "sttl",    "dttl",   "dload",         "dloss",
        "sinpkt",  "dinpkt",  "djit",   "tcprtt",        "synack",
        "ackdat",  "smean",   "dmean",  "ct_state_ttl",  "ct_dst_src_ltm",
    ],
    "random": [
        "ct_dst_sport_ltm", "ct_src_dport_ltm", "ct_srv_src",
        "ct_state_ttl",     "dbytes",           "dtcpb",
        "dttl",             "rate",             "sbytes",
        "sinpkt",           "sjit",             "sload",
        "spkts",            "stcpb",            "swin",
        "synack",           "tcprtt",
    ],
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def udbb_sample(df: pd.DataFrame, counts: dict, seed: int = 42) -> pd.DataFrame:
    """UDBB 샘플링: benign은 counts["benign"]개, 각 attack_step은 counts[step]개."""
    samples = []
    benign_pool = df[df["attack_flag"] == 0]
    n_benign = min(counts["benign"], len(benign_pool))
    samples.append(benign_pool.sample(n=n_benign, random_state=seed))
    print(f"  benign       : {n_benign:,}")

    for step, n_target in counts.items():
        if step == "benign":
            continue
        pool = df[(df["attack_flag"] == 1) & (df["attack_step"] == step)]
        n = min(n_target, len(pool))
        if n > 0:
            samples.append(pool.sample(n=n, random_state=seed))
            print(f"  {step:<14} : {n:,}")
        else:
            print(f"  {step:<14} : 0  (데이터 없음)")

    return (
        pd.concat(samples)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def preprocess(df: pd.DataFrame, features: list, log_features: list) -> np.ndarray:
    """anova_umap.py와 동일한 전처리 파이프라인."""
    X = df[features].replace({np.inf: np.nan, -np.inf: np.nan, -1: np.nan})
    X = X.fillna(X.median())

    log_feats = [f for f in features if f in log_features]
    non_log   = [f for f in features if f not in log_feats]

    for col in log_feats:
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=0, upper=upper)
    for col in non_log:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=lower, upper=upper)

    X[log_feats] = np.log1p(X[log_feats].clip(lower=0))
    return RobustScaler().fit_transform(X)


def plot_umap(
    emb: np.ndarray,
    sample_df: pd.DataFrame,
    step_colors: dict,
    title: str,
    save_path: Path,
) -> None:
    """anova_umap.py와 동일한 2-subplot 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"{title}\n"
        f"n_neighbors={UMAP_PARAMS['n_neighbors']}, "
        f"min_dist={UMAP_PARAMS['min_dist']}, "
        f"metric={UMAP_PARAMS['metric']}",
        fontsize=13, fontweight="bold",
    )

    # (A) attack_flag
    ax = axes[0]
    for flag, color in FLAG_COLORS.items():
        mask = sample_df["attack_flag"] == flag
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, alpha=ALPHA, s=SIZE, linewidths=0, rasterized=True)
    ax.set_title("(A)  attack_flag  —  Benign vs Attack", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(
        handles=[mpatches.Patch(color=v, label=FLAG_LABELS[k]) for k, v in FLAG_COLORS.items()],
        markerscale=6, fontsize=10, loc="best",
    )
    ax.grid(alpha=0.2)

    # (B) attack_step
    ax = axes[1]
    handles = []
    for step, color in step_colors.items():
        if step == "benign":
            mask = sample_df["attack_flag"] == 0
        else:
            mask = sample_df["attack_step"] == step
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, alpha=ALPHA, s=SIZE, linewidths=0,
                   rasterized=True, label=step)
        handles.append(mpatches.Patch(color=color, label=step))
    ax.set_title("(B)  attack_step  —  Kill Chain Stage", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(handles=handles, markerscale=6, fontsize=10, loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path.name}")


def run_group(
    dataset: str,
    group: str,
    features: list,
    log_features: list,
    udbb_counts: dict,
    step_colors: dict,
) -> None:
    SEP = "=" * 65
    n_feat = len(features)
    print(f"\n{SEP}")
    print(f"[{dataset}  /  {group}]  ({n_feat}개 피처)")
    print(SEP)

    # 1. Load
    csv_path = RAW_DIR / dataset / "training-flow.csv"
    print(f"Loading {csv_path.name} ...")
    usecols = list(dict.fromkeys(features + ["attack_flag", "attack_step"]))
    df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)
    df["attack_flag"] = (
        pd.to_numeric(df["attack_flag"], errors="coerce").fillna(0).astype(int)
    )
    df["attack_step"] = (
        df["attack_step"].fillna("benign").astype(str).str.strip().str.lower()
    )
    print(f"  원본 행 수: {len(df):,}")

    # 2. UDBB 샘플링
    print("\n[UDBB] 샘플링 중 ...")
    sample_df = udbb_sample(df, udbb_counts)
    print(f"  샘플링 완료: {len(sample_df):,} rows")

    # 3. 전처리
    print("\n전처리 중 ...")
    X_scaled = preprocess(sample_df, features, log_features)

    # 4. UMAP
    print("UMAP 실행 중 ...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    emb = reducer.fit_transform(X_scaled)
    print("UMAP 완료.")

    # 5. 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f"umap_{dataset}_{group}{n_feat}.png"
    title = f"{dataset}  —  {group} Subset UMAP ({n_feat} features)"
    plot_umap(emb, sample_df, step_colors, title, save_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # cicids2018: random, extratrees
    for group, features in CICIDS_GROUPS.items():
        run_group(
            dataset      = "cicids2018",
            group        = group,
            features     = features,
            log_features = CICIDS_LOG,
            udbb_counts  = CICIDS_UDBB,
            step_colors  = CICIDS_STEP_COLORS,
        )

    # unsw_nb15: anova, extratrees, lit_yin2023, random
    for group, features in UNSW_GROUPS.items():
        run_group(
            dataset      = "unsw_nb15",
            group        = group,
            features     = features,
            log_features = UNSW_LOG,
            udbb_counts  = UNSW_UDBB,
            step_colors  = UNSW_STEP_COLORS,
        )

    print("\n모든 그룹 완료.")
    print(f"출력 위치: {OUT_DIR}")
