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
[Stage 2.7] Reference Camouflage 기준값 계산
            netflowgap 피처 조합으로 Full UMAP 실행
            → Camouflage@1.0 실측값 추출 → 제약 임계값(θ)으로 저장
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
                    각 서브셋마다 full_sil + boundary_mean + camouflage 계산
                    (동일 임베딩 재사용 — k-NN 1회 추가)
                    │
                    └─ 제약 기반 최종 선택
                         camouflage ≤ θ(netflowgap) 를 만족하는 후보 중
                         full_sil 최댓값 → best_features 확정
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

## 알고리즘 상세 (Algorithm Details)

### 전처리 파이프라인

`data_loader.py`의 `preprocess()` 함수가 실행하는 6단계 파이프라인입니다.

```
① Inf / NaN → 열별 중앙값(median) 대체
   (inf, -inf, -1 모두 NaN 처리 후 열별 median으로 imputation)

② Percentile Clipping (이상치 제거)
   - log 변환 대상 피처 (skewed): clip(lower=0, upper=0.99분위)
   - 나머지 피처:                  clip(lower=0.01분위, upper=0.99분위)

③ log1p 변환 (log 피처만 적용)
   X[col] = log(1 + X[col])   — 0-safe, 음수 불가

④ RobustScaler 정규화
   X_scaled = (X - median) / IQR   (중앙값 기준, 이상치 내성)
```

**log1p 적용 피처 목록 (13개, `LOG_FEATURES`):**

| flow duration | totlen fwd/bwd pkts | flow byts/s | flow pkts/s |
| fwd pkts/s | fwd iat tot | fwd iat mean | flow iat mean/std/max |
| active mean | idle mean | tot fwd/bwd pkts | fwd/bwd pkt len max |
| init fwd/bwd win byts | | | |

---

### Pre-filter 결합 공식

ExtraTrees 중요도 순위 `r_tree(f)` 와 ANOVA F-점수 순위 `r_anova(f)` 를 단순 평균 순위로 결합합니다.

```
avg_rank(f) = ( r_tree(f) + r_anova(f) ) / 2

  r_*(f) : 해당 지표 내림차순 정렬 시 피처 f의 0-based 순위
           (중요도·F-score가 가장 높은 피처 = 0)

상위 K = TOP_K_PREFILTER(기본 20)개를 avg_rank 오름차순으로 선발
```

두 방법을 동등하게 결합하여 트리 구조 편향(ExtraTrees)과 선형 분리도 가정(ANOVA)을 상호 보완합니다.

---

### 제약 기반 최적 서브셋 선택 (Constraint-based Selection)

CASS의 핵심 기여는 **"UMAP 클러스터 분리도가 높은 피처 조합이 ML 탐지 성능도 높다"** 는 주장입니다. 그런데 Silhouette Score만 최적화하면 경계면 근방의 **위장 공격(Camouflage)** 이 묵인될 수 있습니다. 이를 방지하기 위해 netflowgap 실측값을 **외부 안전 기준선** 으로 사용합니다.

#### 선택 절차

```
Step 1 — Reference 기준값 계산
  netflowgap 피처 27개로 Full UMAP 실행
  → Camouflage@1.0 실측값 추출 → θ (임계값)으로 고정

Step 2 — 제약 적용 (Top-K Full 재평가 후)
  survived = { subset : camouflage(subset) ≤ θ }

Step 3 — 최종 선택
  best = argmax full_sil  over survived
  (survived 가 공집합이면 제약 완화 → 전체 Top-K 대상 argmax로 fallback)
```

#### 수식

```
θ = Camouflage@1.0( UMAP_full( X[netflowgap_features] ) )

best_features = argmax  Silhouette(UMAP_full(X[S]))
                S ∈ TopK
                subject to  Camouflage@1.0(UMAP_full(X[S])) ≤ θ
```

#### 논문 방어 논리

> *"We select the feature subset that maximizes UMAP Silhouette Score among candidates
> whose camouflage rate does not exceed that of the NetFlowGap baseline, ensuring
> that boundary-level attack concealment is at least as well-controlled as the
> domain-expert-defined reference."*

- **임계값 근거**: 데이터에서 튜닝한 값이 아니라 **기존 논문(netflowgap)의 실측값** 이므로 리뷰어의 "threshold를 왜 이 값으로 정했는가?" 질문에 객관적 근거 제시 가능
- **주 목적 함수 유지**: Primary objective 는 여전히 Silhouette → CASS의 핵심 기여인 "UMAP 클러스터 구조 기반 선택"이 훼손되지 않음
- **단방향 안전망**: Silhouette ↔ Camouflage 간 트레이드오프가 발생할 때, Camouflage가 netflowgap보다 나쁜 조합만 제거

---

### Elbow 검출 알고리즘

`search_algo.py`의 `find_elbow()` 함수입니다. Fast Silhouette 점수의 내림차순 정렬 후 다음 조건으로 Elbow K를 결정합니다.

```
scores_desc = [s_1 ≥ s_2 ≥ ... ≥ s_n]  (Fast Silhouette 내림차순)

gaps[i] = |s_i - s_{i+1}|               (인접 점수 차이, i = 1..n-1)
max_gap  = max(gaps)
threshold = max_gap × ELBOW_GAP_RATIO   (기본 0.1)

K = 첫 번째 i where gaps[i] < threshold  (0-based: K = i+1)
K = max(K, ELBOW_MIN_K)                 (최소 3 보장)
K = n  if 조건 미발생                   (모두 Full 재평가)
```

**직관:** 점수가 급격히 떨어지기 시작하는 "절벽" 직전 지점을 Elbow로 보고, 그 위의 서브셋만 Full UMAP으로 재평가합니다.

---

### Silhouette Score 서브샘플링

Silhouette는 O(n²) 연산이므로, 10,000개 초과 시 무작위 서브샘플로 근사합니다.

```python
# evaluator.py / analyzer.py 공통 적용
if n > 10_000:
    idx = rng.choice(n, 10_000, replace=False)
    sil = silhouette_score(emb[idx], y[idx], metric="euclidean")
```

---

### 비교군 구성 로직 (`exporter.py`)

N = `len(best_features)` 로 모든 비교군의 피처 수를 동일하게 고정합니다.

| 비교군 | 선택 방법 | 소스 풀 |
|--------|-----------|---------|
| `cass` | UMAP Silhouette 최적화 결과 | — |
| `anova` | `filter_summary`를 ANOVA_F 내림차순 재정렬 → 상위 N개 | Pre-filter 풀 (K개) |
| `extratrees` | `filter_summary`를 Tree_Importance 내림차순 재정렬 → 상위 N개 | Pre-filter 풀 (K개) |
| `random` | `random.sample(candidate_pool, N)` — seed=RANDOM_SEED+1000 | Pre-filter 풀 (K개) |
| `lit_<name>` | `LITERATURE_BASELINES[name]` 수동 정의 (데이터셋 내 존재하는 것만) | 전체 피처 |

> **중요:** anova / extratrees 비교군은 **Pre-filter 후보 풀(K개) 내에서** 재정렬하여 선택합니다. 전체 27개 피처 중 상위 N개가 아닙니다. 동일한 후보 풀 안에서 선택 기준만 달리하여 순수 비교를 보장합니다.

---

### 8개 수치 지표 계산 공식

`analyzer.py`의 `_compute_metrics(emb, y)` — UMAP 2D 임베딩에서 계산합니다.

#### [Separability 그룹]

**① Silhouette Score**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

  a(i) : 포인트 i와 동일 클래스 내 나머지 포인트들의 평균 유클리드 거리
  b(i) : 포인트 i와 반대 클래스 포인트들의 평균 유클리드 거리

Silhouette = mean( s(i) for all i )   ∈ [-1, 1]
```

**② Centroid_to_Benign**
```
c_B = mean(emb[y == 0], axis=0)   # benign 무게중심
c_A = mean(emb[y == 1], axis=0)   # attack 무게중심

Centroid_to_Benign = ||c_A - c_B||_2
```
두 클래스 무게중심 간 유클리드 거리. 클러스터가 멀리 분리될수록 큰 값.

**③ Global_Mean_Dist**
```
A_sub ⊆ attack_pts  (최대 5,000개 서브샘플)
B_sub ⊆ benign_pts  (최대 5,000개 서브샘플)

Global_Mean_Dist = mean( ||a - b||_2  for all a ∈ A_sub, b ∈ B_sub )
```
공격-정상 간 쌍별(pairwise) 거리의 전체 평균. 메모리 제어를 위해 서브샘플.

---

#### [Camouflage 그룹]

**④ Boundary_Mean**
```
NearestNeighbors(n_neighbors=1).fit(benign_pts)
nn_dists[i] = min( ||attack_pts[i] - b||_2  for b ∈ benign_pts )

Boundary_Mean = mean(nn_dists)
```
각 공격 포인트에서 가장 가까운 정상 포인트까지의 거리 평균. 클수록 공격이 정상 영역에서 멀리 분리됨.

**⑤ Camouflage@t**
```
Camouflage@t = mean( nn_dists[i] ≤ t )   (비율, 0~1)
```
benign 근방 반경 t 이내에 위치한 공격 포인트 비율. 작을수록 위장 공격이 적음.
기본값: t = 1.0 (`CAMOUFLAGE_THRESHOLDS = [1.0]`).

---

#### [Cluster 그룹]

공격 포인트만 대상으로 `HDBSCAN(min_cluster_size=50, min_samples=10)` 실행.

**⑥ HDBSCAN_Noise_Rate**
```
labels = HDBSCAN.fit_predict(attack_pts)
HDBSCAN_Noise_Rate = mean( labels == -1 )
```
어느 클러스터에도 할당되지 않은 noise 공격 포인트 비율. 작을수록 공격이 응집되어 있음.

**⑦ Cluster_Count**
```
Cluster_Count = len( unique(labels[labels != -1]) )
```
발견된 유효 공격 클러스터 수. 작을수록 공격 패턴이 단일한 덩어리로 응집.

**⑧ Cohesion_Dist**
```
For each cluster c:
  centroid_c = mean(attack_pts[labels == c], axis=0)
  intra_c = sum( ||p - centroid_c||_2  for p in attack_pts[labels == c] )

Cohesion_Dist = sum(intra_c for all c) / n_valid_attack_pts
```
클러스터 내 포인트들의 무게중심 기준 평균 거리(분산 거리). 작을수록 클러스터가 조밀.

---

### 히트맵 정규화 방식

```
raw_val → norm = (raw - col_min) / (col_max - col_min)   # 0~1

higher_better = True  → 히트맵 값 = norm         (큰 raw가 진한 초록)
higher_better = False → 히트맵 값 = 1 - norm      (작은 raw가 진한 초록)
```

모든 열에서 1.0(진한 초록) = best 방향으로 통일. 셀 annotation은 정규화 전 실제 값 표시.

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

### 2단계 스크리닝 & Elbow 검출 & 제약 기반 선택

UMAP 연산 비용 절감을 위해 2단계 구조를 사용합니다.

1. **Reference 기준값**: netflowgap 피처로 Full UMAP 실행 → Camouflage@1.0 임계값 θ 확정
2. **1단계 (Fast)**: 전체 후보 조합을 빠르게 평가 (Silhouette만)
3. **Elbow 검출**: 내림차순 fast_sil에서 gap < `max_gap × ELBOW_GAP_RATIO` 인 지점 → 상위 K 결정
4. **2단계 (Full)**: 상위 K개에 대해서만 논문 파라미터로 재평가 → Silhouette + Boundary_Mean + Camouflage 동시 계산 (동일 임베딩 재사용)
5. **제약 기반 선택**: `camouflage ≤ θ` 를 만족하는 후보 중 Silhouette 최댓값 선택

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

**`PILOT_MIN_SPEARMAN = 0.7` 근거**

Cohen (1988)의 효과 크기 분류에서 Spearman r ≥ 0.7은 "strong correlation"으로 정의됩니다. 결정계수로 환산하면 R² ≥ 0.49, 즉 Fast Silhouette이 Full Silhouette 분산의 49% 이상을 설명합니다. Top-K 스크리닝의 목적은 완벽한 값 예측이 아니라 상위 후보 식별이므로, 분산의 절반 이상을 설명하는 수준이면 proxy로 충분하다고 판단했습니다.

r < 0.7이면 `--pilot` 플래그 실행 시 `n_neighbors`를 30씩 최대 3회 자동 증가하여 재검증합니다 (`pilot_validation_with_retry`). 최대 재시도 후에도 통과하지 못하면 경고와 함께 진행합니다.

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
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). — `PILOT_MIN_SPEARMAN = 0.7` (strong correlation, R² ≥ 0.49) 기준
