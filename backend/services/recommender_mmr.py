import json
import time
import numpy as np
from typing import List, Optional, Set
from backend.extensions import db
import os
import redis as redis_lib

CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"

# 캐시 미스 시 DB 재조회를 단 하나의 요청/워커만 수행하도록 하는 분산 락 설정
_LOAD_LOCK_KEY = "recipe_embeddings:load_lock"
_LOAD_LOCK_TTL_SEC = 60
_LOAD_WAIT_POLL_MS = 100
_LOAD_WAIT_MAX_RETRIES = 20  # 100ms * 20 = 최대 2초 대기

# Redis 연결 (서킷 브레이커)
_redis_client = None
_redis_last_fail_ts = 0.0
_REDIS_FAIL_COOLDOWN_SEC = float(os.environ.get("REDIS_FAIL_COOLDOWN_SEC", "5"))

def _mark_redis_failed():
    """
    Redis 연결/명령 실패 시 클라이언트를 폐기하고 실패 시각을 기록한다.
    이후 쿨다운 동안은 _get_redis()가 재연결 시도 없이 바로 None을 반환해
    Redis 장애 중 매 요청마다 재연결 타임아웃을 반복해서 겪지 않도록 한다.
    """
    global _redis_client, _redis_last_fail_ts
    _redis_client = None
    _redis_last_fail_ts = time.time()

def _get_redis() -> Optional[redis_lib.Redis]:
    global _redis_client, _redis_last_fail_ts
    if _redis_client is not None:
        return _redis_client

    if time.time() - _redis_last_fail_ts < _REDIS_FAIL_COOLDOWN_SEC:
        return None

    try:
        from backend.config import REDIS_HOST, REDIS_PORT, REDIS_DB
        _redis_client = redis_lib.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False,  # bytes 그대로 처리
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        _redis_client.ping()
        print(f"[REDIS] connected {REDIS_HOST}:{REDIS_PORT} db={REDIS_DB}")
        return _redis_client
    except Exception as e:
        print(f"[REDIS] connection failed: {e}")
        _mark_redis_failed()
        return None

# Redis 임베딩 캐시
def _redis_load_embeddings() -> Optional[tuple]:
    """
    Redis에서 임베딩 로딩
    반환: (ids: np.array, vecs: np.array) or None
    """
    try:
        r = _get_redis()
        if r is None:
            return None
        ids_raw = r.get("recipe_embeddings:ids")
        vecs_raw = r.get("recipe_embeddings:vecs")
        shape_raw = r.get("recipe_embeddings:shape")
        if ids_raw is None or vecs_raw is None or shape_raw is None:
            return None
        ids = np.array(json.loads(ids_raw), dtype=np.int64)
        shape = json.loads(shape_raw)
        vecs = np.frombuffer(vecs_raw, dtype=np.float32).reshape(shape)
        return ids, vecs
    except Exception as e:
        print(f"[REDIS] load embeddings error: {e}")
        _mark_redis_failed()
        return None

def _redis_save_embeddings(ids: np.ndarray, vecs: np.ndarray):
    try:
        r = _get_redis()
        if r is None:
            return
        from backend.config import REDIS_TTL
        pipe = r.pipeline()
        pipe.setex("recipe_embeddings:ids",   REDIS_TTL, json.dumps(ids.tolist()))
        pipe.setex("recipe_embeddings:vecs",  REDIS_TTL, vecs.tobytes())
        pipe.setex("recipe_embeddings:shape", REDIS_TTL, json.dumps(list(vecs.shape)))
        pipe.execute()
        size_mb = vecs.nbytes / 1024 / 1024
        print(f"[REDIS] embeddings saved N={len(ids)} size={size_mb:.2f}MB")
    except Exception as e:
        print(f"[REDIS] save embeddings error: {e}")
        _mark_redis_failed()

def _redis_invalidate_embeddings():
    """재학습 후 Redis 임베딩 캐시 삭제."""
    try:
        r = _get_redis()
        if r is None:
            return
        r.delete("recipe_embeddings:ids",
                 "recipe_embeddings:vecs",
                 "recipe_embeddings:shape")
        print("[REDIS] embeddings cache invalidated")
    except Exception as e:
        print(f"[REDIS] invalidate error: {e}")
        _mark_redis_failed()

# 벡터 파싱 유틸
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
    """재학습 후 Redis 캐시 삭제."""
    _redis_invalidate_embeddings()
    print("[CACHE] redis cache invalidated")

def _load_from_db() -> tuple:
    """DB에서 임베딩 로딩."""
    t0 = time.time()
    rows = db.session.execute(
        db.text("SELECT rcp_sno, vector FROM recipe_embeddings")
    ).fetchall()

    ids, vecs, bad = [], [], 0
    for rid, v in rows:
        vec = _to_vec(v)
        if vec is None:
            bad += 1
            continue
        ids.append(int(rid))
        vecs.append(vec)

    if not vecs:
        ids_arr = np.array([], dtype=np.int64)
        vecs_arr = np.zeros((0, 1), dtype=np.float32)
    else:
        ids_arr = np.array(ids, dtype=np.int64)
        vecs_arr = np.vstack(vecs).astype(np.float32)

    took = int((time.time() - t0) * 1000)
    print(f"[DB] recipe_embeddings loaded N={len(ids)}, bad={bad}, tookMs={took}")
    return ids_arr, vecs_arr

def _load_all_recipe_embeddings(force: bool = False) -> tuple:
    # 캐시 미사용/강제 재로딩: DB에서 읽고 Redis 갱신
    if force or not CACHE_ENABLED:
        ids, vecs = _load_from_db()
        _redis_save_embeddings(ids, vecs)
        return ids, vecs

    # Redis HIT
    t0 = time.time()
    result = _redis_load_embeddings()
    if result is not None:
        ids, vecs = result
        took = int((time.time() - t0) * 1000)
        print(f"[REDIS HIT] embeddings N={len(ids)} tookMs={took}")
        return ids, vecs

    # Redis MISS → 분산 락으로 단 하나의 요청만 DB 재조회 (캐시 스탬피드 방지)
    print("[REDIS MISS] loading from DB...")
    if _try_acquire_load_lock():
        try:
            ids, vecs = _load_from_db()
            _redis_save_embeddings(ids, vecs)
        finally:
            _release_load_lock()
        return ids, vecs

    # 락 획득 실패: 다른 요청/워커가 이미 채우는 중 → 짧게 폴링하며 대기
    for _ in range(_LOAD_WAIT_MAX_RETRIES):
        time.sleep(_LOAD_WAIT_POLL_MS / 1000)
        result = _redis_load_embeddings()
        if result is not None:
            print("[REDIS] cache filled by another request while waiting for lock")
            return result

    # 최대 대기 시간을 넘겨도 안 채워짐(락 소유자 지연/실패) → 최후 수단으로 캐싱 없이 직접 DB 조회
    print("[REDIS MISS] lock wait timed out, falling back to direct DB read (no cache write)")
    return _load_from_db()

def _load_user_embedding(user_id: int) -> Optional[np.ndarray]:
    row = db.session.execute(
        db.text("SELECT vector FROM user_embeddings WHERE user_id=:uid"),
        {"uid": int(user_id)}
    ).fetchone()
    if not row or row[0] is None:
        return None
    return _to_vec(row[0])

def _cosine_vec_scores(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = u.astype(np.float32)
    u_norm = np.linalg.norm(u) + 1e-12
    X_norm = np.linalg.norm(X, axis=1) + 1e-12
    return (X @ u) / (X_norm * u_norm)

def _get_recent_event_recipe_ids(user_id: int, limit: int = 50) -> List[int]:
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
    return [int(rid) for rid, _t in rows if rid is not None]

def _get_seen_recipe_ids(user_id: int, limit: int = 5000) -> Set[int]:
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
    return {int(rid) for (rid,) in rows if rid is not None}

def _build_runtime_user_vec(base_user_vec: np.ndarray, recent_ids: List[int]) -> np.ndarray:
    if base_user_vec is None:
        return None
    if not recent_ids:
        return base_user_vec

    rows = db.session.execute(
        db.text("""
            SELECT rcp_sno, vector
            FROM recipe_embeddings
            WHERE rcp_sno IN :ids
        """).bindparams(ids=tuple(set(recent_ids)))
    ).fetchall()

    vecs = [_to_vec(v) for _rid, v in rows if _to_vec(v) is not None]
    if not vecs:
        return base_user_vec

    recent_mean = np.mean(np.vstack(vecs).astype(np.float32), axis=0)
    alpha = 0.20
    return ((1 - alpha) * base_user_vec + alpha * recent_mean).astype(np.float32)

def _mmr_rerank(
    cand_ids: np.ndarray,
    cand_vecs: np.ndarray,
    cand_rel: np.ndarray,
    k: int,
    lambda_mmr: float,
) -> List[int]:
    selected, selected_vecs = [], []
    lam = float(lambda_mmr)

    order = np.lexsort((cand_ids, -cand_rel))
    cand_ids = cand_ids[order]
    cand_vecs = cand_vecs[order]
    cand_rel = cand_rel[order]

    for _ in range(min(k, len(cand_ids))):
        if not selected_vecs:
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

        best_idx = int(np.lexsort((cand_ids, -mmr_scores))[0])
        selected.append(int(cand_ids[best_idx]))
        selected_vecs.append(cand_vecs[best_idx])
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
    t0 = time.time()
    ids, X = _load_all_recipe_embeddings(force=False)

    if X is None or len(ids) == 0:
        return []

    base_user_vec = _load_user_embedding(user_id)
    if base_user_vec is None:
        out = sorted([int(x) for x in ids[:k]])
        return out

    recent_ids = _get_recent_event_recipe_ids(user_id=user_id, limit=50)
    user_vec = _build_runtime_user_vec(base_user_vec, recent_ids)

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
    topN = min(max(k * 80, 400), len(ids))
    cand_idx = np.argpartition(-rel, topN - 1)[:topN]
    cand_ids = ids[cand_idx]
    cand_vecs = X[cand_idx]
    cand_rel = rel[cand_idx]

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
    print(f"[RECOMMEND] user={user_id} k={k} tookMs={took}")
    return selected

def warmup_recipe_cache(force: bool = True):
    try:
        _load_all_recipe_embeddings(force=force)
        print("[WARMUP] recipe cache warmed up (redis)")
    except Exception as e:
        print(f"[WARMUP] failed: {e}")

def _try_acquire_load_lock() -> bool:
    """
    여러 워커/프로세스가 동시에 DB 재조회(스케줄 refresh-ahead 또는 캐시 미스)를
    실행하지 않도록 하는 분산 락. SET NX EX로 하나만 락을 획득해 DB 재조회 +
    Redis 재적재를 수행하게 한다.
    """
    r = _get_redis()
    if r is None:
        return False
    try:
        return bool(r.set(_LOAD_LOCK_KEY, "1", nx=True, ex=_LOAD_LOCK_TTL_SEC))
    except Exception as e:
        print(f"[REDIS] load lock error: {e}")
        _mark_redis_failed()
        return False

def _release_load_lock():
    """작업이 끝나면 TTL 만료를 기다리지 않고 즉시 락을 반납한다."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(_LOAD_LOCK_KEY)
    except Exception as e:
        print(f"[REDIS] load lock release error: {e}")
        _mark_redis_failed()

def refresh_ahead_job():
    """
    Redis 캐시 TTL이 만료되기 전에 주기적으로 미리 캐시를 재적재한다.
    TTL 만료 시점에 다수의 요청이 동시에 캐시 미스를 겪고 각자 DB를
    조회하는 캐시 스탬피드(thundering herd)를 방지하기 위한 refresh-ahead 전략.
    """
    if not CACHE_ENABLED:
        return
    if not _try_acquire_load_lock():
        print("[REFRESH-AHEAD] skipped (lock held by another worker)")
        return
    try:
        warmup_recipe_cache(force=True)
        print("[REFRESH-AHEAD] recipe cache refreshed proactively")
    except Exception as e:
        print(f"[REFRESH-AHEAD] failed: {e}")
    finally:
        _release_load_lock()