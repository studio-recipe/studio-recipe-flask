# backend/services/recommender_cf.py

from typing import Any, Dict, List, Optional, Tuple
import math
import random

import numpy as np

from backend.extensions import db
from backend.models import (
    UserEmbedding,
    RecipeEmbedding,
    Recipe,
    UserReference,
)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_user_embedding(user_id: int) -> Optional[np.ndarray]:
    emb: Optional[UserEmbedding] = (
        db.session.query(UserEmbedding)
        .filter(UserEmbedding.user_id == user_id)
        .one_or_none()
    )
    if emb is None:
        return None
    return np.array(emb.to_array(), dtype=np.float32)


def _load_recipe_embeddings_with_meta() -> List[Dict[str, Any]]:
    """
    RECIPE_EMBEDDINGS + RECIPE 메타데이터 join
    """
    q = (
        db.session.query(
            RecipeEmbedding,
            Recipe,
        )
        .join(Recipe, RecipeEmbedding.rcp_sno == Recipe.rcp_sno)
    )

    results: List[Dict[str, Any]] = []
    for emb, recipe in q.all():
        results.append(
            {
                "rcp_sno": recipe.rcp_sno,
                "title": recipe.rcp_ttl,
                "name": recipe.ckg_nm,
                "img_url": recipe.rcp_img_url,
                "category": recipe.ckg_knd_acto_nm,
                "method": recipe.ckg_mth_acto_nm,
                "emb": np.array(emb.to_array(), dtype=np.float32),
            }
        )
    return results


def _fallback_popular_recipes(limit: int = 10) -> List[Dict[str, Any]]:
    """
    임베딩 또는 유저 정보가 없을 때 인기순 추천 (조회수 기반 등)
    """
    q = (
        db.session.query(Recipe)
        .order_by(Recipe.inq_cnt.desc())
        .limit(limit)
    )

    results: List[Dict[str, Any]] = []
    for r in q.all():
        results.append(
            {
                "rcpSno": r.rcp_sno,
                "title": r.rcp_ttl,
                "name": r.ckg_nm,
                "imgUrl": r.rcp_img_url,
                "score": None,
                "reason": "popular",
                "meta": {},
            }
        )
    return results


def _compute_base_scores(
    user_vec: np.ndarray,
    recipes: List[Dict[str, Any]],
) -> List[Tuple[int, float]]:
    """
    user_vec 과 각 recipe 임베딩 간 cosine similarity 계산
    반환: (index, score) 리스트
    """
    scores: List[Tuple[int, float]] = []
    for idx, r in enumerate(recipes):
        s = _cosine_sim(user_vec, r["emb"])
        scores.append((idx, s))
    return scores


def _rerank_with_diversity(
    recipes: List[Dict[str, Any]],
    base_scores: List[Tuple[int, float]],
    final_k: int,
    lambda_div: float = 0.3,
    novelty_bonus: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    간단한 다양성 보정:
    - 카테고리(ckg_knd_acto_nm) 기준으로 같은 카테고리 과다 반복에 페널티
    - 인기/조회수 낮은 것에 약간의 novelty 보너스

    base_scores: (recipe_index, relevance_score)
    """
    # category 카운트
    category_count: Dict[Optional[str], int] = {}

    # base_scores 높은 순으로 정렬
    base_scores = sorted(base_scores, key=lambda x: x[1], reverse=True)

    selected: List[Dict[str, Any]] = []

    for idx, base_score in base_scores:
        r = recipes[idx]
        cat = r.get("category")
        cat_count = category_count.get(cat, 0)

        # diversity penalty: 같은 카테고리가 많이 나올수록 페널티
        diversity_penalty = 1.0 / (1.0 + lambda_div * cat_count)

        # novelty: 조회수 / 추천수 등으로 보정할 수 있으나,
        # 여기서는 랜덤한 작은 보너스를 부여(동점 깨기 용도)
        novelty = 1.0 + novelty_bonus * random.random()

        final_score = base_score * diversity_penalty * novelty

        r_out = {
            "rcpSno": r["rcp_sno"],
            "title": r["title"],
            "name": r["name"],
            "imgUrl": r["img_url"],
            "score": final_score,
            "reason": "cf_with_diversity",
            "meta": {
                "base_similarity": base_score,
                "category": cat,
                "category_count_before": cat_count,
                "diversity_penalty": diversity_penalty,
            },
        }

        selected.append((final_score, r_out))

    # 최종 점수 기준 재정렬 후 top-k 반환
    selected = sorted(selected, key=lambda x: x[0], reverse=True)
    return [x[1] for x in selected[:final_k]]


def recommend_for_user(user_id: int, size: int = 10) -> List[Dict[str, Any]]:
    """
    1) USER_EMBEDDINGS 에 유저 벡터 있으면 → CF + 다양성 추천
    2) 없으면 → 인기 레시피로 fallback
    """
    user_vec = _load_user_embedding(user_id)
    if user_vec is None:
        # 이벤트/임베딩 없는 신규 유저 → 인기순 추천
        return _fallback_popular_recipes(limit=size)

    recipes = _load_recipe_embeddings_with_meta()
    if not recipes:
        # 레시피 임베딩이 없으면 마찬가지로 인기순
        return _fallback_popular_recipes(limit=size)

    base_scores = _compute_base_scores(user_vec, recipes)
    ranked = _rerank_with_diversity(
        recipes,
        base_scores,
        final_k=size,
        lambda_div=0.3,
        novelty_bonus=0.1,
    )
    return ranked
