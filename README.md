# CASS — Cluster-Aware Feature Selection System

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![cuML](https://img.shields.io/badge/cuML-26.2.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 개요 (Overview)

**CASS (Cluster-Aware Feature Selection System)** 는 UMAP 기반의 모델 비종속적(model-agnostic) 피처 선택 프레임워크입니다.

### 핵심 가설

> **"저차원 매니폴드 공간(UMAP)에서 클러스터 분리도(Silhouette Score)가 높은 피처 조합은, 특정 학습 모델에 대한 편향 없이 다양한 ML 탐지 모델에서 높은 성능을 제공한다."**

전통적인 피처 선택(RF 중요도, ANOVA 등)은 특정 모델의 손실 함수나 분포 가정에 의존합니다. 반면 CASS는 **UMAP의 위상적 클러스터 구조**를 평가 기준으로 삼아, XGBoost · Random Forest · LSTM · CNN 등 어떤 학습 모델에도 bias 없이 일반화되는 피처 부분집합을 탐색합니다.

---

## 파이프라인 (Pipeline)

```
training-flow.csv
        │
        ▼
[1] 데이터 로드 & UDBB 샘플링
    Benign 60k / Attack 3단계 각 20k (3:1 균형)
        │
        ▼
[2] 전처리
    Inf/NaN → median | percentile clipping | log1p | RobustScaler
        │
        ▼
[3] Pre-filter
    ExtraTreesClassifier 중요도 + ANOVA F-score
    → 평균 순위 기반 상위 K개 피처 선발 (기본 top-20)
        │
        ▼
[선택] Pilot 검증
    Fast ↔ Full Silhouette 상관 (Spearman r ≥ 0.7 확인)
        │
        ▼
[4] 2단계 스크리닝 탐색
    ┌─ [1단계] Fast UMAP (n_neighbors=30) 전체 후보 평가
    │         Greedy 전진 선택  or  Random 서브셋 샘플링
    │
    ├─ Elbow 검출: fast_sil 내림차순 gap 분석 → 상위 K 결정
    │
    └─ [2단계] Full UMAP (n_neighbors=150, 논문 파라미터) 상위 K 재평가
        │
        ▼
[5] 시각화 & 저장
    UMAP 산점도 (Benign vs Attack / Kill Chain)
    2단계 스크리닝 비교 플롯  |  Pilot 산점도
```

---

## 프로젝트 구조 (Directory Structure)

```
CASS/
├── data/
│   ├── raw/
│   │   └── training-flow.csv          # CICIDS2018 원본 (76 피처 + 레이블)
│   └── processed/
│       └── cicids2018_processed.csv   # 전처리 완료 데이터 (자동 생성)
├── src/
│   ├── __init__.py
│   ├── config.py          # 전역 설정 (UMAP 파라미터, 경로, 샘플링 수 등)
│   ├── data_loader.py     # 로드 + UDBB 샘플링 + 전처리 파이프라인
│   ├── pre_filter.py      # ExtraTrees + ANOVA 평균순위 기반 사전 필터링
│   ├── evaluator.py       # UMAP 차원 축소 + Silhouette Score 계산 (cuML GPU)
│   └── search_algo.py     # 2단계 스크리닝 (Greedy/Random + Elbow + Full 재평가)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb   # 클래스 분포, 피처 분포, 전처리 결과 EDA
│   └── 02_silhouette_analysis.ipynb     # 탐색 결과 분석, 최적 부분집합 시각화
├── results/
│   ├── figures/           # UMAP 시각화, 스크리닝 비교 플롯 저장
│   └── logs/              # pre_filter_ranking.csv, search_results_*.csv 저장
├── main.py                # 전체 파이프라인 실행 진입점 (CLI)
└── requirements.txt
```

---

## 핵심 설계 결정 (Key Design Decisions)

### UMAP 파라미터 (NetFlowGap 기준)

| 구분 | n_neighbors | min_dist | metric | init |
|------|------------|---------|--------|------|
| **Full** (논문 보고용) | 150 | 0.01 | manhattan | spectral |
| **Fast** (스크리닝 전용) | 30 | 0.1 | manhattan | random |

Full 파라미터는 [NetFlowGap](../NetFlowGap)의 설정을 그대로 따릅니다.
Fast 파라미터는 1단계 스크리닝에만 사용되며 논문에 직접 보고되지 않습니다.

### UDBB 샘플링

Uniform Distribution Based Balancing (Abdulhammed et al., 2019):

| 클래스 | 샘플 수 |
|--------|--------|
| Benign | 60,000 |
| Action (C2) | 20,000 |
| Infection | 20,000 |
| Installation | 20,000 |

### 2단계 스크리닝 & Elbow 검출

UMAP은 연산 비용이 높아 모든 피처 조합에 Full 파라미터를 적용하면 시간이 과도하게 소요됩니다.
이를 해결하기 위해:

1. **1단계**: Fast UMAP으로 전체 후보 조합을 빠르게 평가
2. **Elbow 검출**: 내림차순 정렬된 fast_sil에서 인접 gap이 `max_gap × ELBOW_GAP_RATIO` 이하로 떨어지는 지점을 Elbow로 판정 → 상위 K 결정 (최소 `ELBOW_MIN_K` 보장)
3. **2단계**: 상위 K개에 대해서만 Full UMAP 재평가 → 최종 결과 도출

### Pilot 검증

`--pilot` 플래그 사용 시, 탐색 시작 전 무작위 서브셋 20개에 대해 Fast/Full Silhouette을 모두 계산하여 Spearman r을 측정합니다. r ≥ 0.7이면 Fast가 Full의 유효한 proxy임을 확인합니다.

---

## 빠른 시작 (Quick Start)

### 환경 설정

```bash
# cuML (RAPIDS) — GPU UMAP 필수
conda install -c rapidsai -c conda-forge cuml=26.2.0 python=3.10 cudatoolkit=12.x

# 나머지 의존성
pip install -r requirements.txt
```

### 데이터 준비

```
CASS/data/raw/training-flow.csv   # CICIDS2018 (76 피처 + attack_flag + attack_step)
```

`attack_step` 컬럼 값: `benign`, `action`, `infection`, `installation`

### 실행

```bash
# 기본 실행 (greedy, top-20)
python main.py

# random 탐색, 100개 서브셋
python main.py --mode random --n-subsets 100

# Pilot 검증 포함
python main.py --pilot

# 상위 15개 피처 후보 사용
python main.py --top-k 15

# 전체 옵션
python main.py --mode random --n-subsets 100 --top-k 15 --pilot
```

### 출력 결과

```
results/
├── figures/
│   ├── umap_best_subset.png          # 최적 부분집합 UMAP 시각화
│   ├── two_phase_screening.png       # Fast vs Full Silhouette 비교
│   ├── pilot_fast_vs_full.png        # (--pilot) Pilot 검증 산점도
│   └── best_subset_umap_embeddings.csv  # UMAP 임베딩 좌표
└── logs/
    ├── pre_filter_ranking.csv        # 사전 필터링 순위표
    ├── search_results_greedy.csv     # Greedy 탐색 결과
    └── search_results_random.csv     # Random 탐색 결과
```

---

## 주요 설정값 (config.py)

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `TOP_K_PREFILTER` | 20 | Pre-filter 후 유지할 피처 수 |
| `SEARCH_MODE` | `"greedy"` | 탐색 모드 (`"greedy"` / `"random"`) |
| `N_RANDOM_SUBSETS` | 80 | Random 모드 평가 서브셋 수 |
| `MIN_SUBSET_SIZE` | 3 | 서브셋 최소 크기 |
| `MAX_SUBSET_SIZE` | 15 | 서브셋 최대 크기 |
| `ELBOW_GAP_RATIO` | 0.1 | Elbow 판정 임계 비율 |
| `ELBOW_MIN_K` | 3 | 2단계 재평가 최소 K |
| `PILOT_N` | 20 | Pilot 검증 서브셋 수 |
| `PILOT_MIN_SPEARMAN` | 0.7 | Pilot 통과 기준 Spearman r |

---

## 의존성 (Dependencies)

| 패키지 | 용도 |
|--------|------|
| `cuml >= 26.2.0` | GPU UMAP (`cuml.manifold.UMAP`) |
| `scikit-learn >= 1.3` | ExtraTrees, ANOVA, Silhouette Score, RobustScaler |
| `numpy`, `pandas` | 데이터 처리 |
| `scipy` | Spearman 상관 계산 |
| `matplotlib` | 시각화 |
| `tqdm` | 진행 표시 |

---

## Future Work

현재 구현은 UMAP Silhouette Score를 단일 지표로 피처 부분집합을 평가합니다.
이하의 확장은 **논문 후속 작업**으로 예정되어 있습니다.

### 1. 다차원 UMAP 메트릭 분석 (`analyzer.py`)

단일 Silhouette 대신, 클러스터 구조를 3개 그룹 8개 지표로 종합 평가:

| 그룹 | 지표 |
|------|------|
| **Separability** (분리도) | Silhouette Score, Centroid-to-Benign Distance, Global Mean Distance |
| **Camouflage** (위장 탐지) | Camouflage@1.0, Boundary Mean |
| **Cluster Structure** (군집 구조) | HDBSCAN Noise Rate, Cluster Count, Cohesion Distance |

### 2. 피어슨 상관관계 히트맵 (`correlation.py`)

**목적**: UMAP 기반 피처 선택이 실제 ML 탐지 성능과 양의 상관관계를 갖는다는 것을 수치로 증명

**방법**:
- 선택된 피처 조합별로 8개 UMAP 메트릭 + 5개 이상 ML 모델(XGBoost, RF, LSTM, CNN, etc.)의 F1/Precision/Recall/Accuracy 수집
- 피어슨 상관계수 히트맵으로 각 UMAP 지표 ↔ ML 성능 지표 간 관계 시각화
- **가설 검증**: Silhouette Score가 높은 피처 조합 → 대부분의 ML 모델에서 높은 탐지율 → UMAP 기반 선택의 모델 비종속성 증명

### 3. 비교 기준선 (Comparison Baseline)

CASS의 우위를 정량적으로 보여주기 위한 비교:

| 방법 | 선택 기준 | 편향 |
|------|-----------|------|
| **CASS** (제안) | UMAP Silhouette | 없음 (model-agnostic) |
| RF Importance | 특정 RF 모델의 feature importance | RF에 편향 |
| ANOVA F-score | 선형 분리도 가정 | 선형 모델에 편향 |
| 모델별 래퍼 | 특정 모델 성능 최적화 | 해당 모델에 편향 |

### 4. 다중 데이터셋 일반화 검증

현재 CICIDS2018만 지원. 향후 아래 데이터셋으로 확장하여 일반화 검증:

- **LSPR23** — IoT 환경 공격 트래픽
- **UNSW-NB15** — 다양한 공격 유형
- **CICIoT2023** — 최신 IoT 공격 패턴
- **TON-IoT** — IoT/IIoT 다중 환경

각 데이터셋에서 CASS로 선택한 피처 조합이 다수 ML 모델에서 일관된 성능을 보이는지 검증.

---

## 참고 문헌 (References)

- **NetFlowGap** — UMAP 파라미터 및 UDBB 샘플링 기준 (본 디렉토리 기반)
- Abdulhammed et al. (2019) — UDBB 샘플링 전략
- I-SiamIDS (2021), J.BigData (2021) — 3:1 클래스 균형 비율 근거
- McInnes et al. (2018) — UMAP 원논문
- CICIDS2018 — Canadian Institute for Cybersecurity
