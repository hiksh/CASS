# CASS — Cluster-Aware Feature Selection System

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![cuML](https://img.shields.io/badge/cuML-26.2.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 개요 (Overview)

**CASS (Cluster-Aware Feature Selection System)** 는 UMAP 기반의 모델 비종속적(model-agnostic) 피처 선택 프레임워크입니다.

### 핵심 가설

> **"저차원 매니폴드 공간(UMAP)에서 클러스터 분리도(Silhouette Score)가 높은 피처 조합은,
> 특정 학습 모델에 대한 편향 없이 다양한 ML 탐지 모델에서 높은 성능을 제공한다."**

전통적인 피처 선택(RF 중요도, ANOVA 등)은 특정 모델의 손실 함수나 분포 가정에 의존합니다.
CASS는 **UMAP의 위상적 클러스터 구조**만을 평가 기준으로 삼아 어떤 ML 모델에도 bias 없이
일반화되는 피처 부분집합을 탐색합니다.

---

## 전체 파이프라인 (Pipeline)

```
data/raw/training-flow.csv
         │
         ▼
[Stage 1] 데이터 로드 & UDBB 샘플링
          Benign 60k / Action 20k / Infection 20k / Installation 20k
         │
         ▼
[Stage 2] 전처리
          Inf·NaN → median  |  percentile clipping  |  log1p  |  RobustScaler
         │
         ▼
[Stage 3] Pre-filter
          ExtraTrees 중요도 + ANOVA F-score → 평균 순위 → 상위 K개 선발 (기본 20개)
         │
         ├─ (--pilot) Pilot 검증
         │            무작위 서브셋 20개로 Fast ↔ Full Silhouette 상관 확인
         │            Spearman r ≥ 0.7 이면 Fast가 유효한 proxy
         │
         ▼
[Stage 4] 2단계 스크리닝 탐색
          ┌─ 1단계: Fast UMAP (n_neighbors=30) 으로 전체 후보 평가
          │         --mode greedy  : 전진 선택 (피처 1개씩 추가)
          │         --mode random  : 무작위 서브셋 N개 샘플링
          │
          ├─ Elbow 검출: fast_sil 내림차순 gap → 상위 K 결정
          │
          └─ 2단계: Full UMAP (n_neighbors=150) 으로 상위 K 재평가
                    → best_features 확정
         │
         ▼
[Stage 5] 시각화 & 저장
          UMAP 산점도 (Benign vs Attack · Kill Chain)
          Fast vs Full Silhouette 비교 플롯
         │
         ├─ (--export) 비교군 Export             [Stage 6 or 5]
         │             5개 비교군 × train/test CSV
         │             training-flow.csv → train_*.csv
         │             test-flow.csv     → test_*.csv  (동일 scaler)
         │
         └─ (--analyze) UMAP 수치 분석           [Stage 6 or 7]
                        5개 비교군 × 8개 지표 계산
                        → comparison_heatmap.png
```

---

## 논문 증명 전략 (Validation Strategy)

### 비교군 구성

동일 피처 수 **N = len(best_features)** 를 고정하여 차원 혼입(confounding)을 제거합니다.
선택 방법의 차이만 순수하게 비교할 수 있습니다.

| 비교군 | 선택 기준 | 피처 수 | 모델 편향 |
|--------|-----------|---------|-----------|
| `cass` | UMAP Silhouette 최적화 | N (자동) | 없음 ← **제안 방법** |
| `anova` | ANOVA F-score 상위 N개 | N | 선형 분리도 가정 |
| `extratrees` | ExtraTrees 중요도 상위 N개 | N | 트리 구조 편향 |
| `random` | 무작위 N개 | N | — (하한 기준선) |
| `lit_netflowgap` | NetFlowGap 논문 수동 선정 | 27 (고정) | 도메인 전문가 편향 |

모든 비교군은 동일한 pre-filter 후보 풀(K개)에서 선택됩니다.

### UMAP 수치 지표 8개 (`--analyze`)

비교군별 Full UMAP 실행 후 3그룹 8개 지표를 계산하여 정규화 히트맵으로 시각화합니다.

| 그룹 | 지표 | best 방향 | 의미 |
|------|------|-----------|------|
| **Separability** | Silhouette | ↑ | 클래스 간 분리도 |
| | Centroid_to_Benign | ↑ | 공격-정상 무게중심 거리 |
| | Global_Mean_Dist | ↑ | 공격-정상 전체 평균 거리 |
| **Camouflage** | Camouflage@1.0 | ↓ | benign 근방에 숨은 공격 비율 |
| | Boundary_Mean | ↑ | 공격→nearest benign 평균 거리 |
| **Cluster** | HDBSCAN_Noise_Rate | ↓ | 군집되지 않은 공격 포인트 비율 |
| | Cluster_Count | ↓ | 공격 클러스터 수 (적을수록 응집) |
| | Cohesion_Dist | ↓ | 클러스터 내 평균 분산 거리 |

히트맵은 열별로 0~1 정규화 후 **1 = best** 방향으로 통일하며, 셀에 실제 값을 함께 표기합니다.
CASS 행이 전반적으로 높은 수치를 보이면 **"UMAP 기반 선택이 기하 구조적으로도 우수하다"** 는 논문 주장을 시각적으로 증명합니다.

### ML 교차 비교 (`--export`)

`--export`로 생성된 CSV를 외부 ML 모델(XGBoost, RF, LSTM 등)에 직접 입력합니다.
UMAP이 test 데이터를 보지 않으므로 data leakage가 없습니다.

```
train_*.csv  ←  UDBB 샘플 (120k행, 훈련 데이터 scaler fit)
test_*.csv   ←  test-flow.csv 전체 (동일 scaler transform, 완전 분리)
```

---

## 프로젝트 구조 (Directory Structure)

```
CASS/
├── data/
│   ├── raw/
│   │   ├── training-flow.csv          # CICIDS2018 훈련 원본 (76 피처 + 레이블)
│   │   └── test-flow.csv              # CICIDS2018 테스트 원본 (완전 분리 보관)
│   └── processed/
│       └── cicids2018_processed.csv   # 전처리 완료 (자동 생성)
├── src/
│   ├── config.py        # 전역 설정 — UMAP 파라미터, 경로, 샘플링 수, 비교군 정의
│   ├── data_loader.py   # 로드 + UDBB 샘플링 + 전처리 파이프라인
│   ├── pre_filter.py    # ExtraTrees + ANOVA 평균 순위 기반 사전 필터링
│   ├── evaluator.py     # UMAP 차원 축소 + Silhouette Score (cuML GPU)
│   ├── search_algo.py   # 2단계 스크리닝 (Greedy/Random + Elbow + Full 재평가)
│   ├── exporter.py      # 비교군 구성 + train/test CSV 생성
│   └── analyzer.py      # 비교군별 8개 UMAP 지표 계산 + 히트맵
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   └── 02_silhouette_analysis.ipynb
├── results/             # 모든 출력 자동 저장 (gitignore 처리)
│   ├── figures/
│   ├── logs/
│   └── exports/
├── main.py              # CLI 진입점
└── requirements.txt
```

---

## 실행 가이드 (Execution Guide)

### 1단계 — 환경 설정

```bash
# cuML (RAPIDS) — GPU UMAP 필수
conda install -c rapidsai -c conda-forge cuml=26.2.0 python=3.10 cudatoolkit=12.x

# 나머지 의존성
pip install -r requirements.txt
```

### 2단계 — 데이터 준비

```
data/raw/training-flow.csv   # 컬럼: 피처 76개 + attack_flag + attack_step
data/raw/test-flow.csv       # 동일 컬럼 구조
```

`attack_flag`: 0 = benign, 1 = attack
`attack_step`: `benign` · `action` · `infection` · `installation`

### 3단계 — 파이프라인 실행

목적에 따라 플래그를 조합합니다.

```bash
# ── 기본 실행 (피처 선택만) ─────────────────────────────────
python main.py

# ── 탐색 방식 변경 ──────────────────────────────────────────
python main.py --mode random --n-subsets 100   # 무작위 탐색
python main.py --top-k 15                      # 후보 피처 15개로 축소

# ── 신뢰도 검증 추가 ────────────────────────────────────────
python main.py --pilot                         # Fast↔Full 상관 사전 확인

# ── 논문 증명용 비교 데이터 생성 ────────────────────────────
python main.py --export                        # 5개 비교군 train/test CSV
python main.py --analyze                       # 8지표 히트맵 생성
python main.py --export --analyze              # 두 가지 동시 실행

# ── 권장 실행 (논문 전체 결과 재현) ─────────────────────────
python main.py --pilot --export --analyze
```

### CLI 플래그 전체 목록

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--mode` | `greedy` | 탐색 방식 (`greedy` / `random`) |
| `--top-k` | `20` | Pre-filter 후 유지할 피처 수 |
| `--n-subsets` | `80` | random 모드 평가 서브셋 수 |
| `--pilot` | off | Fast↔Full Silhouette 상관 사전 검증 |
| `--export` | off | 비교군별 train/test CSV 저장 |
| `--analyze` | off | 비교군별 8지표 계산 + 히트맵 저장 |

### 4단계 — 출력 결과 확인

```
results/
├── figures/
│   ├── umap_best_subset.png            # CASS 최적 피처 UMAP 시각화
│   ├── two_phase_screening.png         # Fast vs Full Silhouette 비교
│   ├── pilot_fast_vs_full.png          # (--pilot) Pilot 검증 산점도
│   ├── best_subset_umap_embeddings.csv # UMAP 임베딩 좌표 (umap_analysis 입력용)
│   └── comparison_heatmap.png          # (--analyze) 비교군 × 8지표 히트맵
├── logs/
│   ├── pre_filter_ranking.csv          # 사전 필터링 전체 순위표
│   ├── search_results_greedy.csv       # Greedy 탐색 단계별 결과
│   ├── search_results_random.csv       # Random 탐색 결과
│   ├── pilot_validation.csv            # (--pilot) Fast·Full Silhouette 수치
│   └── comparison_metrics.csv          # (--analyze) 비교군별 8지표 수치
└── exports/                            # (--export) ML 학습용 CSV
    ├── train_cass.csv / test_cass.csv               # CASS 선택 피처
    ├── train_anova.csv / test_anova.csv             # ANOVA 상위 N개
    ├── train_extratrees.csv / test_extratrees.csv   # ExtraTrees 상위 N개
    ├── train_random.csv / test_random.csv           # 무작위 N개
    └── train_lit_netflowgap.csv / test_lit_netflowgap.csv  # NetFlowGap 27개
```

---

## 핵심 설계 결정 (Key Design Decisions)

### UMAP 파라미터 (NetFlowGap 기준)

| 구분 | n_neighbors | min_dist | metric | init | 용도 |
|------|------------|---------|--------|------|------|
| **Full** | 150 | 0.01 | manhattan | spectral | 논문 보고용, 최종 평가 |
| **Fast** | 30 | 0.1 | manhattan | random | 1단계 스크리닝 전용 |

Fast 파라미터는 내부 스크리닝에만 사용되며 논문에 직접 보고되지 않습니다.

### UDBB 샘플링

Uniform Distribution Based Balancing (Abdulhammed et al., 2019):

| 클래스 | 샘플 수 | 비율 |
|--------|--------|------|
| Benign | 60,000 | 50% |
| Action (C2) | 20,000 | 17% |
| Infection | 20,000 | 17% |
| Installation | 20,000 | 17% |

### 2단계 스크리닝 & Elbow 검출

UMAP 연산 비용 절감을 위해 2단계 구조를 사용합니다.

1. **1단계 (Fast)**: 전체 후보 조합을 빠르게 평가
2. **Elbow 검출**: 내림차순 fast_sil에서 gap < `max_gap × ELBOW_GAP_RATIO` 인 지점 → 상위 K 결정
3. **2단계 (Full)**: 상위 K개에 대해서만 논문 파라미터로 재평가

### Train / Test 분리

- `training-flow.csv` → UDBB 샘플링 → scaler fit → UMAP 피처 선택 → export train
- `test-flow.csv` → 동일 scaler transform (fit 없음) → export test

UMAP이 훈련 데이터만 보므로 **test leakage 없음**.

---

## 주요 설정값 (config.py)

### 탐색 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `TOP_K_PREFILTER` | 20 | Pre-filter 후 유지할 피처 수 |
| `SEARCH_MODE` | `"greedy"` | 탐색 모드 |
| `N_RANDOM_SUBSETS` | 80 | Random 모드 평가 서브셋 수 |
| `MIN_SUBSET_SIZE` | 3 | 서브셋 최소 크기 |
| `MAX_SUBSET_SIZE` | 15 | 서브셋 최대 크기 |
| `ELBOW_GAP_RATIO` | 0.1 | Elbow 판정 임계 비율 |
| `ELBOW_MIN_K` | 3 | 2단계 재평가 최소 K |

### Pilot 검증

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `PILOT_N` | 20 | Pilot 검증 서브셋 수 |
| `PILOT_MIN_SPEARMAN` | 0.7 | 통과 기준 Spearman r |

### 비교군 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `N_RANDOM_BASELINE` | 1 | 랜덤 비교군 반복 횟수 |
| `LITERATURE_BASELINES` | `{"netflowgap": [...]}` | 논문 기준 피처 조합 |

**Literature Baseline 추가 방법** — `config.py`의 dict에 항목만 추가:

```python
LITERATURE_BASELINES = {
    "netflowgap":      ["flow duration", "syn flag cnt", ...],
    "cicids2018_paper": ["feature_a", "feature_b", ...],  # 추가
}
```

자동으로 `train_lit_cicids2018_paper.csv` / `test_lit_cicids2018_paper.csv` 생성됩니다.

### Analyzer 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `HDBSCAN_MIN_CLUSTER_SIZE` | 50 | HDBSCAN 최소 클러스터 크기 |
| `HDBSCAN_MIN_SAMPLES` | 10 | HDBSCAN core point 이웃 수 |
| `CAMOUFLAGE_THRESHOLDS` | `[1.0]` | Camouflage@K 임계값 목록 |
| `MAX_CDIST_SAMPLE` | 5,000 | Global_Mean_Dist 서브샘플 상한 |

---

## 의존성 (Dependencies)

| 패키지 | 용도 |
|--------|------|
| `cuml >= 26.2.0` | GPU UMAP (`cuml.manifold.UMAP`) |
| `scikit-learn >= 1.3` | ExtraTrees, ANOVA, HDBSCAN, Silhouette, RobustScaler |
| `numpy`, `pandas` | 데이터 처리 |
| `scipy` | Spearman 상관, 거리 계산 |
| `matplotlib`, `seaborn` | 시각화 |
| `tqdm` | 진행 표시 |

cuML 설치: [https://docs.rapids.ai/install](https://docs.rapids.ai/install)

---

## Future Work

### 1. 피어슨 상관관계 히트맵 (`correlation.py`)

비교군별 8개 UMAP 지표 + 다수 ML 모델 성능(F1/Precision/Recall/Accuracy)을 수집하여
피어슨 상관계수 히트맵으로 시각화합니다.
**"Silhouette가 높을수록 ML 성능도 높다"** 는 상관관계를 수치로 증명합니다.

### 2. 다중 데이터셋 일반화 검증

- **LSPR23** — IoT 환경 공격 트래픽
- **UNSW-NB15** — 다양한 공격 유형
- **CICIoT2023** — 최신 IoT 공격 패턴
- **TON-IoT** — IoT/IIoT 다중 환경

---

## 참고 문헌 (References)

- **NetFlowGap** — UMAP 파라미터 및 UDBB 샘플링 기준
- Abdulhammed et al. (2019) — UDBB 샘플링 전략
- I-SiamIDS (2021), J.BigData (2021) — 3:1 클래스 균형 비율 근거
- McInnes et al. (2018) — UMAP 원논문
- CICIDS2018 — Canadian Institute for Cybersecurity
