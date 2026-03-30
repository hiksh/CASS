"""
CASS — Pre-filter (1단계: 통계적 사전 필터링)

ExtraTreesClassifier 피처 중요도와 ANOVA F-점수를 결합하여
전체 피처에서 후보군을 압축합니다.

방법론:
  - ExtraTrees: 트리 기반 비선형 중요도 (순열 불필요, 빠름)
  - ANOVA F-score: 클래스 간 분산/내 분산 비율 (선형 판별력)
  - 결합: 두 랭킹의 평균 순위 → 상위 K개 선택
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif

from .config import TOP_K_PREFILTER, RANDOM_SEED


def _tree_importance(X: np.ndarray, y: np.ndarray, feature_names: list) -> list:
    """ExtraTreesClassifier 피처 중요도 순위 반환."""
    print("  [Pre-filter] ExtraTrees 중요도 계산 중 ...")
    clf = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X, y)
    ranked = sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    return ranked


def _anova_score(X: np.ndarray, y: np.ndarray, feature_names: list) -> list:
    """ANOVA F-점수 순위 반환."""
    print("  [Pre-filter] ANOVA F-점수 계산 중 ...")
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
    ranked = sorted(
        zip(feature_names, scores),
        key=lambda x: x[1], reverse=True,
    )
    return ranked


def pre_filter(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_top: int = None,
) -> tuple[list, pd.DataFrame]:
    """
    두 가지 통계 지표의 평균 순위로 상위 피처를 선택합니다.

    Args:
        X            : 전처리된 피처 행렬
        y            : 레이블 (0/1)
        feature_names: 피처 이름 목록
        n_top        : 유지할 피처 수 (None → config 기본값)

    Returns:
        top_features : 선택된 피처 이름 리스트
        summary_df   : 랭킹 요약 DataFrame
    """
    if n_top is None:
        n_top = TOP_K_PREFILTER

    print(f"\n[Pre-filter] {len(feature_names)}개 피처 → 상위 {n_top}개 선택 중 ...")

    tree_ranked = _tree_importance(X, y, feature_names)
    anova_ranked = _anova_score(X, y, feature_names)

    tree_rank  = {feat: i for i, (feat, _) in enumerate(tree_ranked)}
    anova_rank = {feat: i for i, (feat, _) in enumerate(anova_ranked)}
    tree_imp   = dict(tree_ranked)
    anova_f    = dict(anova_ranked)

    avg_rank = {f: (tree_rank[f] + anova_rank[f]) / 2.0 for f in feature_names}
    sorted_feats = sorted(avg_rank.items(), key=lambda x: x[1])

    # 전체 피처 랭킹 (ANOVA/ExtraTrees 비교군이 전체에서 선택할 수 있도록)
    all_rows = []
    for rank, (feat, avg_r) in enumerate(sorted_feats, start=1):
        all_rows.append({
            "Rank":             rank,
            "Feature":          feat,
            "Tree_Importance":  round(tree_imp[feat], 6),
            "ANOVA_F":          round(anova_f[feat], 2),
            "Avg_Rank":         round(avg_r, 1),
        })

    full_summary_df = pd.DataFrame(all_rows)
    top_features    = full_summary_df.head(n_top)["Feature"].tolist()

    print("\n  ── 상위 피처 요약 ──")
    print(full_summary_df.head(n_top).to_string(index=False))

    return top_features, full_summary_df
