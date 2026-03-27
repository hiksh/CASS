"""
CASS — Global Configuration
모든 하이퍼파라미터, 경로, 피처 정의를 중앙 관리합니다.
"""
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = BASE_DIR / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
LOGS_DIR      = RESULTS_DIR / "logs"

TRAIN_FILE = RAW_DIR / "training-flow.csv"
TEST_FILE  = RAW_DIR / "test-flow.csv"

# ── CICIDS2018 피처 (NetFlowGap 기준 27개) ───────────────────────────────────
ALL_FEATURES = [
    "flow duration",    "tot fwd pkts",     "tot bwd pkts",
    "totlen fwd pkts",  "totlen bwd pkts",
    "fwd pkt len max",  "fwd pkt len mean",  "fwd pkt len std",
    "bwd pkt len max",  "bwd pkt len mean",  "pkt len mean",
    "flow byts/s",      "flow pkts/s",       "fwd pkts/s",
    "flow iat mean",    "flow iat std",      "flow iat max",
    "fwd iat tot",      "fwd iat mean",
    "syn flag cnt",     "ack flag cnt",      "psh flag cnt",  "fin flag cnt",
    "init fwd win byts","init bwd win byts",
    "active mean",      "idle mean",
]

# 로그 변환을 적용할 피처 (skewed 분포 보정)
LOG_FEATURES = [
    "flow duration",    "totlen fwd pkts",  "totlen bwd pkts",
    "flow byts/s",      "flow pkts/s",      "fwd pkts/s",
    "fwd iat tot",      "fwd iat mean",
    "flow iat mean",    "flow iat std",     "flow iat max",
    "active mean",      "idle mean",
    "tot fwd pkts",     "tot bwd pkts",
    "fwd pkt len max",  "bwd pkt len max",
    "init fwd win byts","init bwd win byts",
]

# ── UDBB 샘플링 (Benign:Attack = 3:1, 공격 단계 균등) ───────────────────────
# 근거: Abdulhammed et al. (2019), I-SiamIDS (2021), J.BigData (2021)
UDBB_COUNTS = {
    "benign":       60_000,
    "action":       20_000,
    "infection":    20_000,
    "installation": 20_000,
}

# ── UMAP 파라미터 (cuml.manifold.UMAP, GPU) ──────────────────────────────────
# cuML v26.2.0 확인: manhattan 지원, n_jobs 파라미터 없음

# Full: 논문 보고용 — NetFlowGap 설정 그대로
UMAP_PARAMS = dict(
    n_neighbors=150,
    min_dist=0.01,
    metric='manhattan',
    n_components=2,
    init='spectral',
    random_state=42,
)

# Fast: 1단계 스크리닝 전용 (논문에 직접 보고되지 않음)
UMAP_PARAMS_FAST = dict(
    n_neighbors=30,
    min_dist=0.1,
    metric='manhattan',
    n_components=2,
    init='random',
    random_state=42,
)

# ── 2단계 스크리닝: Elbow K 자동 결정 ────────────────────────────────────────
# gap_ratio: 인접 점수 간 gap이 max_gap의 몇 % 이하면 elbow로 판정
ELBOW_GAP_RATIO = 0.1
# 최소 K: elbow가 너무 작게 나와도 최소한 이 수만큼 재평가
ELBOW_MIN_K = 3

# ── Pilot 상관관계 검증 ───────────────────────────────────────────────────────
# Fast Silhouette ↔ Full Silhouette의 Spearman 상관이 충분한지 사전 검증
PILOT_N             = 20    # 검증에 사용할 무작위 서브셋 수
PILOT_MIN_SPEARMAN  = 0.7   # 이 값 이상이어야 2단계 스크리닝 신뢰 가능

# ── Pre-filter ────────────────────────────────────────────────────────────────
TOP_K_PREFILTER = 20

# ── 탐색 알고리즘 ─────────────────────────────────────────────────────────────
SEARCH_MODE      = "greedy"   # "greedy" | "random"
N_RANDOM_SUBSETS = 80
MIN_SUBSET_SIZE  = 3
MAX_SUBSET_SIZE  = 15

RANDOM_SEED = 42
