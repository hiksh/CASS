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

# CICIDS2018 결과 디렉토리 (기본 / 하위 호환)
FIGURES_DIR   = RESULTS_DIR / "cicids2018" / "figures"
LOGS_DIR      = RESULTS_DIR / "cicids2018" / "logs"
EXPORTS_DIR   = RESULTS_DIR / "cicids2018" / "exports"

TRAIN_FILE = RAW_DIR / "cicids2018" / "training-flow.csv"
TEST_FILE  = RAW_DIR / "cicids2018" / "test-flow.csv"

# ── CICIDS2018 전체 피처 (timestamp·레이블 컬럼 제외, 78개) ──────────────────
ALL_FEATURES = [
    # 포트 / 프로토콜 / 플로우
    "dst port",         "protocol",         "flow duration",
    # 패킷 수 / 바이트 수
    "tot fwd pkts",     "tot bwd pkts",
    "totlen fwd pkts",  "totlen bwd pkts",
    # 패킷 길이
    "fwd pkt len max",  "fwd pkt len min",  "fwd pkt len mean", "fwd pkt len std",
    "bwd pkt len max",  "bwd pkt len min",  "bwd pkt len mean", "bwd pkt len std",
    # 처리율
    "flow byts/s",      "flow pkts/s",
    # 플로우 IAT
    "flow iat mean",    "flow iat std",     "flow iat max",     "flow iat min",
    # 순방향 IAT
    "fwd iat tot",      "fwd iat mean",     "fwd iat std",      "fwd iat max",  "fwd iat min",
    # 역방향 IAT
    "bwd iat tot",      "bwd iat mean",     "bwd iat std",      "bwd iat max",  "bwd iat min",
    # PSH / URG 플래그
    "fwd psh flags",    "bwd psh flags",    "fwd urg flags",    "bwd urg flags",
    # 헤더 / 처리율
    "fwd header len",   "bwd header len",
    "fwd pkts/s",       "bwd pkts/s",
    # 패킷 길이 통계
    "pkt len min",      "pkt len max",      "pkt len mean",     "pkt len std",  "pkt len var",
    # 플래그 카운트
    "fin flag cnt",     "syn flag cnt",     "rst flag cnt",     "psh flag cnt",
    "ack flag cnt",     "urg flag cnt",     "cwe flag count",   "ece flag cnt",
    # 비율 / 평균 크기
    "down/up ratio",    "pkt size avg",     "fwd seg size avg", "bwd seg size avg",
    # Bulk 통계 (CICIDS2018에서 대부분 0 — pre-filter에서 자연 탈락)
    "fwd byts/b avg",   "fwd pkts/b avg",   "fwd blk rate avg",
    "bwd byts/b avg",   "bwd pkts/b avg",   "bwd blk rate avg",
    # 서브플로우
    "subflow fwd pkts", "subflow fwd byts",
    "subflow bwd pkts", "subflow bwd byts",
    # 윈도우 / 세그먼트
    "init fwd win byts","init bwd win byts",
    "fwd act data pkts","fwd seg size min",
    # Active / Idle
    "active mean",      "active std",       "active max",       "active min",
    "idle mean",        "idle std",         "idle max",         "idle min",
]

# 로그 변환을 적용할 피처 (skewed 분포 보정) — 음수가 될 수 없는 연속형만 포함
LOG_FEATURES = [
    # 패킷 수 / 바이트 수
    "tot fwd pkts",      "tot bwd pkts",
    "totlen fwd pkts",   "totlen bwd pkts",
    # 패킷 길이
    "fwd pkt len max",   "fwd pkt len min",   "fwd pkt len mean",  "fwd pkt len std",
    "bwd pkt len max",   "bwd pkt len min",   "bwd pkt len mean",  "bwd pkt len std",
    "pkt len min",       "pkt len max",       "pkt len mean",      "pkt len std",
    "pkt len var",       "pkt size avg",      "fwd seg size avg",  "bwd seg size avg",
    # 처리율
    "flow byts/s",       "flow pkts/s",
    "fwd pkts/s",        "bwd pkts/s",
    # IAT
    "flow iat mean",     "flow iat std",      "flow iat max",
    "fwd iat tot",       "fwd iat mean",      "fwd iat std",       "fwd iat max",
    "bwd iat tot",       "bwd iat mean",      "bwd iat std",       "bwd iat max",
    # 헤더 / 윈도우
    "fwd header len",    "bwd header len",
    "init fwd win byts", "init bwd win byts",
    # 서브플로우
    "subflow fwd pkts",  "subflow fwd byts",
    "subflow bwd pkts",  "subflow bwd byts",
    # Active / Idle
    "active mean",       "active std",        "active max",
    "idle mean",         "idle std",          "idle max",
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
# n_neighbors=80: Full(150)과의 임베딩 구조 간극을 줄여 Fast BM이 Full BM의
# 유효한 proxy가 되도록 조정. min_dist=0.05, init='spectral'도 Full에 가깝게 설정.
UMAP_PARAMS_FAST = dict(
    n_neighbors=80,
    min_dist=0.05,
    metric='manhattan',
    n_components=2,
    init='spectral',
    random_state=42,
)

# ── 2단계 스크리닝: Elbow K 자동 결정 ────────────────────────────────────────
# gap_ratio: 인접 점수 간 gap이 max_gap의 몇 % 이하면 elbow로 판정
ELBOW_GAP_RATIO = 0.1
# 최소 K: elbow가 너무 작게 나와도 최소한 이 수만큼 재평가
ELBOW_MIN_K = 8

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
MIN_SILHOUETTE   = 0.0   # Silhouette 하한 제약 — 이 값 초과해야 후보로 인정

RANDOM_SEED = 42

# ── Analyzer (UMAP 수치 분석) ─────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 50   # HDBSCAN 최소 클러스터 크기
HDBSCAN_MIN_SAMPLES      = 10   # HDBSCAN core point 최소 이웃 수
CAMOUFLAGE_THRESHOLDS    = [1.0] # Camouflage@K 임계값 목록 (필요 시 [0.5, 1.0, 2.0] 등)
MAX_CDIST_SAMPLE         = 5_000 # Global_Mean_Dist 서브샘플 상한 (메모리/속도 제어)

# ── Export 비교군 설정 ────────────────────────────────────────────────────────
# Random 비교군 반복 횟수 (1이면 "random", 2이상이면 "random_1", "random_2" ...)
N_RANDOM_BASELINE = 1

# Literature 기준 피처 조합 — 논문별로 추가/수정 가능
LITERATURE_BASELINES = {
    # Umar et al. (2024) "Effects of feature selection and normalization
    # on network intrusion detection", CSE-CIC-IDS2018 기준 12개
    "umar2024": [
        "tot fwd pkts",     "totlen fwd pkts",
        "bwd pkt len max",  "flow pkts/s",
        "fwd iat mean",     "bwd iat tot",      "bwd iat mean",
        "rst flag cnt",     "urg flag cnt",
        "init fwd win byts","fwd seg size min",
        "idle max",
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNSW-NB15 데이터셋 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_UNSW_DIR = RAW_DIR / "unsw_nb15"

UNSW_TRAIN_FILE = _UNSW_DIR / "training-flow.csv"
UNSW_TEST_FILE  = _UNSW_DIR / "test-flow.csv"

UNSW_RESULTS_DIR = RESULTS_DIR / "unsw_nb15"
UNSW_FIGURES_DIR = UNSW_RESULTS_DIR / "figures"
UNSW_LOGS_DIR    = UNSW_RESULTS_DIR / "logs"
UNSW_EXPORTS_DIR = UNSW_RESULTS_DIR / "exports"

# 수치형 피처 31개 (protocol, service, state 범주형 제외)
UNSW_ALL_FEATURES = [
    "dur",
    "spkts",        "dpkts",
    "sbytes",       "dbytes",
    "rate",
    "sload",        "dload",
    "sloss",        "dloss",
    "sinpkt",       "dinpkt",
    "sjit",         "djit",
    "swin",         "stcpb",        "dtcpb",        "dwin",
    "tcprtt",       "synack",       "ackdat",
    "smean",        "dmean",
    "trans_depth",  "response_body_len",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "is_ftp_login", "ct_ftp_cmd",   "ct_flw_http_mthd", "is_sm_ips_ports",
]

# log1p 적용 피처 (right-skewed, non-negative 연속형)
UNSW_LOG_FEATURES = [
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
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
]

# UDBB 샘플링: lateral-movement, installation 제외 (소수 + PCA leakage)
UNSW_UDBB_COUNTS = {
    "benign":         30_000,
    "reconnaissance": 10_000,
    "action":         10_000,
    "infection":      10_000,
}

# Kill Chain 단계 색상 (시각화용)
UNSW_STEP_COLORS = {
    "benign":         "#3498DB",
    "action":         "#E74C3C",
    "infection":      "#F39C12",
    "reconnaissance": "#9B59B6",
}

# Literature baseline — 인용수 높은 UNSW-NB15 피처 선택 논문 추가 예정
# TODO: 논문 확인 후 피처 리스트 채울 것
UNSW_LITERATURE_BASELINES = {
    # "author_year": ["feature_a", "feature_b", ...],
}


# ── 데이터셋 설정 통합 접근자 ────────────────────────────────────────────────

def get_dataset_config(name: str) -> dict:
    """
    데이터셋 이름으로 설정 딕셔너리를 반환합니다.

    Args:
        name: "cicids2018" 또는 "unsw_nb15"

    Returns:
        {train_file, test_file, all_features, log_features, udbb_counts,
         literature_baselines, step_colors,
         figures_dir, logs_dir, exports_dir}
    """
    if name == "cicids2018":
        return dict(
            name                 = "cicids2018",
            train_file           = TRAIN_FILE,
            test_file            = TEST_FILE,
            all_features         = ALL_FEATURES,
            log_features         = LOG_FEATURES,
            udbb_counts          = UDBB_COUNTS,
            literature_baselines = LITERATURE_BASELINES,
            step_colors          = {
                "benign":       "#3498DB",
                "action":       "#E74C3C",
                "infection":    "#F39C12",
                "installation": "#2ECC71",
            },
            figures_dir          = FIGURES_DIR,
            logs_dir             = LOGS_DIR,
            exports_dir          = EXPORTS_DIR,
        )
    elif name == "unsw_nb15":
        return dict(
            name                 = "unsw_nb15",
            train_file           = UNSW_TRAIN_FILE,
            test_file            = UNSW_TEST_FILE,
            all_features         = UNSW_ALL_FEATURES,
            log_features         = UNSW_LOG_FEATURES,
            udbb_counts          = UNSW_UDBB_COUNTS,
            literature_baselines = UNSW_LITERATURE_BASELINES,
            step_colors          = UNSW_STEP_COLORS,
            figures_dir          = UNSW_FIGURES_DIR,
            logs_dir             = UNSW_LOGS_DIR,
            exports_dir          = UNSW_EXPORTS_DIR,
        )
    else:
        raise ValueError(f"알 수 없는 데이터셋: {name!r}. 'cicids2018' 또는 'unsw_nb15' 중 선택하세요.")
