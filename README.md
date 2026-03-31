# BOSS: Boundary-Optimized Structural Feature Selection

> **차원 축소와 클래스간 거리를 활용한 체계적인 네트워크 침입 탐지용 특징 선택 프레임워크**
> A Systematic Framework to Select Intrusion Detection Features based on Dimension Reduction and Inter-class Distance

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![cuML](https://img.shields.io/badge/cuML-26.2.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 개요 (Overview)

**BOSS (Boundary-Optimized Structural feature Selection)** 는 NIDS 데이터셋에서 클래스 간 구조적 분리도를 최대화하는 특징 조합을 자동으로 선택하는 프레임워크입니다.

기존 연구(Umar, IGRF-RFE 등)는 F1 점수 기준으로 개별 특징의 기여도를 평가하여, 특징 공간 내 공격과 정상 샘플 간의 분포를 반영하지 못합니다. BOSS는 UMAP 축소 공간에서 **경계 평균(Boundary Mean, BM)** — 공격 샘플별 최근접 정상 샘플까지의 평균 거리 — 을 직접 목적함수로 삼아, 어떤 분류기에도 편향 없이 일반화되는 특징 조합을 탐색합니다.

$$BM = \frac{1}{|A|} \sum_{i \in A} \min_{j \in B} d(i, j)$$

A: 공격 샘플 집합, B: 정상 샘플 집합, d(i, j): 유클리드 거리

---

## 파이프라인 (Pipeline)

```
NIDS 데이터셋
      │
      ▼
[Stage 1] 사전 필터 (Pre-filter)
          ExtraTrees 중요도 + ANOVA F-score 평균 순위
          → 상위 m개 특징 선발 (통계적 무의미 특징 제거)
      │
      ▼
[Stage 2] 국소 UMAP 기반 경계 평균 계산
          Fast UMAP (좁은 구조 보존)으로 전체 후보를 신속 평가
          그리디 탐색 + Elbow 검출 → 상위 K개 후보 추출
      │
      ▼
[Stage 3] 전역 UMAP 기반 경계 평균 계산
          Full UMAP (넓은 구조 보존)으로 상위 K개 후보 재평가
          → BM 최댓값을 가진 특징 조합 최종 확정
```

국소(Fast) UMAP은 전역(Full) UMAP보다 데이터 분포 구조의 유지 범위가 좁아 속도가 빠릅니다. 2단계 구조는 연산량이 많은 전역 UMAP을 소수 후보에만 적용하여 효율성을 확보합니다.

---

## 실험 결과 (Results)

### UMAP 공간에서의 특징 비교 (CIC-IDS-2018)

BOSS 특징 조합에서는 공격 샘플이 정상 샘플과 명확히 분리된 클러스터를 형성하는 반면, Umar 조합에서는 공격과 정상이 중심부에서 동심원 형태로 혼재됩니다.

| 데이터셋 | 방법 | Boundary Mean ↑ |
|---------|------|-----------------|
| CIC-IDS-2018 | **BOSS** | **8.92** |
| CIC-IDS-2018 | Umar [2] | 7.12 |
| UNSW-NB15 | **BOSS** | **5.37** |
| UNSW-NB15 | IGRF-RFE [3] | 2.05 (+163% 차이) |

### 분류 성능 비교 (F1 Score, %)

| 데이터셋 | 방법 | 피처 수 | XGBoost | RF | CNN | LSTM | LogReg |
|---------|------|---------|---------|-----|-----|------|--------|
| CIC-IDS-2018 | **BOSS** | **16** | 93.68 | 93.12 | 90.51 | 92.17 | **74.29** |
| CIC-IDS-2018 | Umar [2] | 12 | **93.75** | **93.43** | **91.58** | **93.37** | 72.51 |
| UNSW-NB15 | **BOSS** | **17** | 58.96 | 58.96 | 45.62 | **57.90** | **42.85** |
| UNSW-NB15 | yin2023 [3] | 20 | **59.24** | **59.06** | **49.86** | 55.44 | 40.06 |

BOSS는 분류 성능을 직접 최적화하지 않음에도, 선형 구조에 민감한 분류기(Logistic Regression, LSTM)에서 최대 **2.79%** 우위를 보입니다. 트리 계열 모델에서는 경쟁력 있는 성능을 유지합니다.

---

## 프로젝트 구조 (Directory Structure)

```
BOSS/
├── data/
│   └── raw/
│       ├── cicids2018/
│       │   ├── training-flow.csv      # CIC-IDS-2018 훈련 데이터
│       │   ├── test-flow.csv          # CIC-IDS-2018 테스트 데이터
│       │   └── cicids2018_download.py # Kaggle 자동 다운로드 스크립트
│       └── unsw_nb15/
│           ├── training-flow.csv      # UNSW-NB15 훈련 데이터
│           ├── test-flow.csv          # UNSW-NB15 테스트 데이터
│           └── unsw_nb15_download.py  # Kaggle 자동 다운로드 스크립트
├── src/
│   ├── config.py        # UMAP 파라미터, 경로, 샘플링 설정
│   ├── data_loader.py   # 데이터 로드 + UDBB 샘플링 + 전처리
│   ├── pre_filter.py    # ExtraTrees + ANOVA 사전 필터
│   ├── evaluator.py     # UMAP 차원 축소 + BM 계산 (cuML GPU)
│   ├── search_algo.py   # 2단계 스크리닝 (그리디 + Elbow + Full 재평가)
│   ├── exporter.py      # 비교군 train/test CSV 생성
│   └── analyzer.py      # 비교군별 UMAP 지표 계산 + 히트맵
├── results/             # 출력 자동 저장 (.gitignore)
│   ├── cicids2018/
│   │   ├── figures/     # UMAP 시각화, 스크리닝 플롯, 히트맵
│   │   ├── logs/        # 피처 랭킹, 탐색 결과 CSV
│   │   └── exports/     # ML 학습용 train/test CSV (비교군별)
│   └── unsw_nb15/       # 동일 구조
├── main.py              # CLI 진입점
├── run_ml.py            # ML 분류 성능 평가
└── requirements.txt
```

---

## 실행 가이드 (Quickstart)

### 1. 환경 설정

```bash
# cuML (RAPIDS) — GPU UMAP 필수
conda install -c rapidsai -c conda-forge cuml=26.2.0 python=3.10 cudatoolkit=12.x

# 나머지 의존성
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# CIC-IDS-2018
python data/raw/cicids2018/cicids2018_download.py

# UNSW-NB15
python data/raw/unsw_nb15/unsw_nb15_download.py
```

> UNSW-NB15는 Kaggle(`mrwellsdavid/unsw-nb15`)에서 training-set↔testing-set이 교차 배포되므로 다운로드 스크립트가 자동으로 수정합니다.

### 3. 파이프라인 실행

```bash
# CIC-IDS-2018
python main.py --pilot --export --analyze

# UNSW-NB15
python main.py --dataset unsw_nb15 --pilot --export --analyze
```

### 4. ML 분류 성능 평가

```bash
python run_ml.py --dataset cicids2018
python run_ml.py --dataset unsw_nb15
```

### CLI 플래그

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--dataset` | `cicids2018` | 데이터셋 (`cicids2018` / `unsw_nb15`) |
| `--mode` | `greedy` | 탐색 방식 (`greedy` / `random`) |
| `--top-k` | `30` | 사전 필터 후 유지할 피처 수 |
| `--pilot` | off | 국소↔전역 UMAP BM 상관 사전 검증 |
| `--export` | off | 비교군별 train/test CSV 저장 |
| `--analyze` | off | 비교군별 UMAP 지표 계산 + 히트맵 |

---

## 비교군 구성 (Baselines)

동일 피처 수 N을 고정하여 차원 혼입(confounding)을 제거합니다.

| 비교군 | 선택 기준 |
|--------|-----------|
| `boss` | UMAP Boundary Mean 최대화 ← **제안 방법** |
| `anova` | ANOVA F-score 상위 N개 |
| `extratrees` | ExtraTrees 중요도 상위 N개 |
| `random` | 무작위 N개 (하한 기준선) |
| `lit_*` | 논문 수동 선정 (Umar / yin2023) |

---

## 참고문헌 (References)

[1] M. A. Ferrag et al., "Deep Learning for Cyber Security Intrusion Detection," *J. Inf. Secur. Appl.*, 2020.

[2] M. A. Umar et al., "Effects of Feature Selection and Normalization on Network Intrusion Detection," *TechRxiv*, 2020.

[3] C. Yin et al., "IGRF-RFE: A Hybrid Feature Selection for MLP-Based NIDS on UNSW-NB15," *J. Big Data*, vol. 10, no. 15, 2023.

[4] L. McInnes et al., "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," *arXiv:1802.03426*, 2018.

