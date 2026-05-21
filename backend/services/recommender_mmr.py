import json
import time
import numpy as np
from threading import RLock
from typing import List, Optional, Set, Tuple
from backend.extensions import db
import os
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"

# =========================
# 글로벌 캐시
# =========================
_CACHE_LOCK = RLock()
_RECIPE_IDS = None           # np.array of ids
_RECIPE_VECS = None          # np.array shape (N, D)
_RECIPE_LAST_LOAD = 0.0
_RECIPE_TTL_SEC = 300        # 5분 TTL

def _to_vec(v):
    if v is None:
        return None
    s = str(v).strip()
    try:
        if s.startswith("[") and s.endswith("]"):
            arr = json.loads(s)
            return np.array(arr, dtype=np.float32)
        parts = [p for p in s.split(",") if p.strip() != ""]
        return np.array([float(p) for p in parts], dtype=np.float32)
    except Exception:
        return None

def invalidate_recipe_cache():
    """학습 후 임베딩이 바뀌면 캐시를 반드시 비워야 즉시 반영된다."""
    global _RECIPE_IDS, _RECIPE_VECS, _RECIPE_LAST_LOAD
    with _CACHE_LOCK:
        _RECIPE_IDS = None
        _RECIPE_VECS = None
        _RECIPE_LAST_LOAD = 0.0

def _load_all_recipe_embeddings(force: bool = False):
    global _RECIPE_IDS, _RECIPE_VECS, _RECIPE_LAST_LOAD

    now = time.time()

    # CACHE_ENABLED=false면 항상 DB에서 새로 로딩
    if not CACHE_ENABLED:
        force = True

    if (not force) and _RECIPE_VECS is not None and (now - _RECIPE_LAST_LOAD) < _RECIPE_TTL_SEC:
        return

    with _CACHE_LOCK:
        now = time.time()
        if (not force) and _RECIPE_VECS is not None and (now - _RECIPE_LAST_LOAD) < _RECIPE_TTL_SEC:
            return

        t0 = time.time()
        rows = db.session.execute(
            db.text("SELECT rcp_sno, vector FROM recipe_embeddings")
        ).fetchall()

        ids = []
        vecs = []
        bad = 0

        for rid, v in rows:
            vec = _to_vec(v)
            if vec is None:
                bad += 1
                continue
            ids.append(int(rid))
            vecs.append(vec)

        if not vecs:
            _RECIPE_IDS = np.array([], dtype=np.int64)
            _RECIPE_VECS = np.zeros((0, 1), dtype=np.float32)
        else:
            _RECIPE_IDS = np.array(ids, dtype=np.int64)
            _RECIPE_VECS = np.vstack(vecs).astype(np.float32)

        _RECIPE_LAST_LOAD = time.time()
        took = int((_RECIPE_LAST_LOAD - t0) * 1000)
        print(f"[CACHE] recipe_embeddings loaded N={len(ids)}, bad={bad}, tookMs={took}")

def _load_user_embedding(user_id: int) -> Optional[np.ndarray]:
    row = db.session.execute(
        db.text("SELECT vector FROM user_embeddings WHERE user_id=:uid"),
        {"uid": int(user_id)}
    ).fetchone()
    if not row or row[0] is None:
        return None
    vec = _to_vec(row[0])
    return vec

def _cosine_vec_scores(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    # X: (N, D), u: (D,)
    u = u.astype(np.float32)
    u_norm = np.linalg.norm(u) + 1e-12
    X_norm = np.linalg.norm(X, axis=1) + 1e-12
    return (X @ u) / (X_norm * u_norm)

def _get_recent_event_recipe_ids(user_id: int, limit: int = 50) -> List[int]:
    """
    최근 이벤트(좋아요/조회)를 가져온다.
    - LIKE는 더 강하게, VIEW는 약하게 반영할거라서 둘 다 가져옴
    - UNLIKE는 제외(원하면 로직 추가 가능)
    """
    rows = db.session.execute(
        db.text("""
            SELECT rcp_sno, preference_type
            FROM user_references
            WHERE user_id = :uid
              AND preference_type IN ('LIKE','VIEW')
            ORDER BY CREATED_AT DESC
            LIMIT :lim
        """),
        {"uid": int(user_id), "lim": int(limit)}
    ).fetchall()

    out = []
    for rid, _t in rows:
        if rid is None:
            continue
        out.append(int(rid))
    return out

def _get_seen_recipe_ids(user_id: int, limit: int = 5000) -> Set[int]:
    """
    사용자가 이미 본/좋아요한 아이템을 추천에서 제외하기 위한 seen set
    (너무 크면 limit로 컷)
    """
    rows = db.session.execute(
        db.text("""
            SELECT rcp_sno
            FROM user_references
            WHERE user_id = :uid
              AND preference_type IN ('LIKE','VIEW','UNLIKE')
            ORDER BY CREATED_AT DESC
            LIMIT :lim
        """),
        {"uid": int(user_id), "lim": int(limit)}
    ).fetchall()

    s = set()
    for (rid,) in rows:
        if rid is None:
            continue
        s.add(int(rid))
    return s

def _build_runtime_user_vec(base_user_vec: np.ndarray, recent_ids: List[int]) -> np.ndarray:
    """
    핵심: 재학습 없이도 '최근 이벤트'를 user vec에 살짝 반영
    - 이벤트가 없으면 base_user_vec 그대로 => 추천이 안정적
    - 이벤트가 쌓이면 그 방향으로 조금씩 움직임(편향/다양성 균형)
    """
    if base_user_vec is None:
        return None
    if not recent_ids:
        return base_user_vec

    # recent recipe vec를 평균내서 user vec에 섞는다.
    rows = db.session.execute(
        db.text("""
            SELECT rcp_sno, vector
            FROM recipe_embeddings
            WHERE rcp_sno IN :ids
        """).bindparams(ids=tuple(set(recent_ids)))
    ).fetchall()

    vecs = []
    for _rid, v in rows:
        vv = _to_vec(v)
        if vv is not None:
            vecs.append(vv)

    if not vecs:
        return base_user_vec

    recent_mean = np.mean(np.vstack(vecs).astype(np.float32), axis=0)

    #  섞는 비율: 너무 크면 "갑자기 다른 음식"이 튀어나옴
    # base가 메인, recent는 보정만 (추천 흔들림 방지)
    alpha = 0.20
    u = (1 - alpha) * base_user_vec + alpha * recent_mean
    return u.astype(np.float32)

def _mmr_rerank(
    cand_ids: np.ndarray,
    cand_vecs: np.ndarray,
    cand_rel: np.ndarray,
    k: int,
    lambda_mmr: float,
) -> List[int]:
    """
    MMR 재랭킹 (결정적으로 동작: 랜덤 사용 X)
    - tie-break은 (score, rid)로 고정
    """
    selected: List[int] = []
    selected_vecs: List[np.ndarray] = []

    lam = float(lambda_mmr)

    # 초기 후보 순서를 결정적으로 만들기 위해 rid로 tie-break
    order = np.lexsort((cand_ids, -cand_rel))  # rel desc, rid asc
    cand_ids = cand_ids[order]
    cand_vecs = cand_vecs[order]
    cand_rel = cand_rel[order]

    for _ in range(min(k, len(cand_ids))):
        if not selected_vecs:
            # 가장 rel 높은 것(동점은 rid 작은 것)
            selected.append(int(cand_ids[0]))
            selected_vecs.append(cand_vecs[0])
            cand_ids = cand_ids[1:]
            cand_vecs = cand_vecs[1:]
            cand_rel = cand_rel[1:]
            if len(cand_ids) == 0:
                break
            continue

        S = np.vstack(selected_vecs)
        S_norm = np.linalg.norm(S, axis=1) + 1e-12
        cand_norm = np.linalg.norm(cand_vecs, axis=1) + 1e-12
        sim_mat = (cand_vecs @ S.T) / (cand_norm[:, None] * S_norm[None, :])
        div = np.max(sim_mat, axis=1)

        mmr_scores = lam * cand_rel - (1 - lam) * div

        # tie-break: mmr desc, rid asc
        best_idx = int(np.lexsort((cand_ids, -mmr_scores))[0])

        selected.append(int(cand_ids[best_idx]))
        selected_vecs.append(cand_vecs[best_idx])

        # remove selected
        cand_ids = np.delete(cand_ids, best_idx)
        cand_vecs = np.delete(cand_vecs, best_idx, axis=0)
        cand_rel = np.delete(cand_rel, best_idx)

        if len(cand_ids) == 0:
            break

    return selected

def recommend_mmr(
    user_id: int,
    k: int = 10,
    lambda_mmr: float = 0.8,
    seed_recipe_id=None,
    filter_seen: bool = True,
    exclude_ids=None
) -> List[int]:
    """
    최종 구조:
    1) base user embedding 로드 (학습 결과)
    2) 최근 이벤트(조회/좋아요)로 user vec 살짝 보정 (재학습 없이도 반영)
    3) TopN 후보: relevance topN (Top-K 취향)
    4) 그 후보 안에서 MMR로 다양성 (편향 완화)
    """
    t0 = time.time()
    _load_all_recipe_embeddings(force=False)

    ids = _RECIPE_IDS
    X = _RECIPE_VECS
    if X is None or len(ids) == 0:
        return []

    base_user_vec = _load_user_embedding(user_id)
    if base_user_vec is None:
        # 신규 유저: 임베딩 없으면 인기/랜덤이 아니라 "결정적" fallback
        # (Spring에서 인기순으로 처리하는게 더 좋음. 여기서는 상위 id 일부 반환)
        out = sorted([int(x) for x in ids[:k]])
        return out

    # 최근 이벤트 반영 (재학습 없이도 즉시 반영되는 포인트)
    recent_ids = _get_recent_event_recipe_ids(user_id=user_id, limit=50)
    user_vec = _build_runtime_user_vec(base_user_vec, recent_ids)

    # seed 반영은 optional (현재는 유지하되 너무 세게 섞지 않음)
    if seed_recipe_id is not None:
        seed_row = db.session.execute(
            db.text("SELECT vector FROM recipe_embeddings WHERE rcp_sno=:rid"),
            {"rid": int(seed_recipe_id)}
        ).fetchone()
        if seed_row and seed_row[0] is not None:
            seed_vec = _to_vec(seed_row[0])
            if seed_vec is not None:
                alpha = 0.15
                user_vec = (1 - alpha) * user_vec + alpha * seed_vec

    rel = _cosine_vec_scores(X, user_vec)

    # 후보를 relevance 기준으로 topN만 뽑고 그 안에서 MMR
    topN = min(max(k * 80, 400), len(ids))
    cand_idx = np.argpartition(-rel, topN - 1)[:topN]
    cand_ids = ids[cand_idx]
    cand_vecs = X[cand_idx]
    cand_rel = rel[cand_idx]

    # exclude 구성
    ex = set(exclude_ids or [])
    if filter_seen:
        ex |= _get_seen_recipe_ids(user_id=user_id, limit=5000)

    if ex:
        mask = np.array([int(rid) not in ex for rid in cand_ids], dtype=bool)
        cand_ids = cand_ids[mask]
        cand_vecs = cand_vecs[mask]
        cand_rel = cand_rel[mask]

    if len(cand_ids) == 0:
        return []

    selected = _mmr_rerank(
        cand_ids=cand_ids,
        cand_vecs=cand_vecs,
        cand_rel=cand_rel,
        k=k,
        lambda_mmr=lambda_mmr
    )

    took = int((time.time() - t0) * 1000)
    print(f"[RECOMMEND] user={user_id} k={k} lambda={lambda_mmr} tookMs={took} returned={len(selected)} recentEvents={len(recent_ids)}")
    return selected

def warmup_recipe_cache(force: bool = True):
    try:
        _load_all_recipe_embeddings(force=force)
        print("[WARMUP] recipe cache warmed up")
    except Exception as e:
        print(f"[WARMUP] failed: {e}")