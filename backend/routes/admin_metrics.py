from flask import Blueprint, current_app, jsonify, make_response
from sqlalchemy import text
from backend.extensions import db
from backend.services.recommender_mmr import recommend_mmr

bp = Blueprint("admin_metrics", __name__)

K = 10
LAMBDA_MMR = 0.8


# 응답 헬퍼
def _no_cache_json(payload, status=200):
    resp = make_response(jsonify(payload), status)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _empty_result(total_recipes=0, pos_table=None, reason=""):
    return {
        "recallAt10": 0.0,
        "ndcgAt10": 0.0,
        "hitRateAt10": 0.0,
        "coverage": 0.0,
        "usersEvaluated": 0,
        "usersSkipped": 0,
        "failedRecommendUsers": 0,
        "positivesTotal": 0,
        "avgPositivesPerUser": 0.0,
        "totalRecipes": total_recipes,
        "posTable": pos_table,
        "debugReason": reason,
    }


# 평가 지표 계산
def _dcg(binary_rels):
    return sum(
        1.0 / ((i + 1) ** 0.5)
        for i, rel in enumerate(binary_rels, start=1)
        if rel
    )


def _ndcg_at_k(rec_ids, gt_set, k):
    rels = [1 if rid in gt_set else 0 for rid in rec_ids[:k]]
    dcg = _dcg(rels)
    idcg = _dcg([1] * min(len(gt_set), k))
    return dcg / idcg if idcg > 0 else 0.0


# DB 헬퍼
def _find_positive_table(conn):
    """user_id + rcp_sno/recipe_id 컬럼을 가진 테이블 자동 탐색"""
    dbname = conn.execute(text("SELECT DATABASE()")).scalar()
    if not dbname:
        return None

    rows = conn.execute(
        text("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = :db
              AND LOWER(column_name) IN ('user_id', 'rcp_sno', 'recipe_id')
        """),
        {"db": dbname},
    ).fetchall()

    table_cols = {}
    for t, c in rows:
        table_cols.setdefault(t, set()).add(c.lower())

    for t, cols in table_cols.items():
        if "user_id" in cols and ("rcp_sno" in cols or "recipe_id" in cols):
            recipe_col = "rcp_sno" if "rcp_sno" in cols else "recipe_id"
            return {"table": t, "user_col": "user_id", "recipe_col": recipe_col}

    return None


def _get_total_recipe_count(conn):
    try:
        return int(conn.execute(text("SELECT COUNT(*) FROM recipes")).scalar() or 0)
    except Exception:
        return 0


# 수정 1: created_at까지 함께 로딩 
def _load_user_positives(conn, table, user_col, recipe_col, max_users=200, min_pos=2):
    """
    반환 형식 변경:
    기존: {uid: {rid1, rid2, ...}}                         (set, 시간순서 정보 없음)
    변경: {uid: [(rid1, created_at1), (rid2, created_at2), ...]} (시간 내림차순 정렬된 list)
    """
    user_rows = conn.execute(
        text(f"""
            SELECT {user_col} AS uid, COUNT(*) AS cnt
            FROM {table}
            GROUP BY {user_col}
            HAVING COUNT(*) >= :min_pos
            ORDER BY cnt DESC
            LIMIT :lim
        """),
        {"min_pos": min_pos, "lim": max_users},
    ).fetchall()

    users = [int(r[0]) for r in user_rows if r[0] is not None]
    if not users:
        return {}

    pos_rows = conn.execute(
        text(f"""
            SELECT {user_col} AS uid, {recipe_col} AS rid, created_at
            FROM {table}
            WHERE {user_col} IN :uids
            ORDER BY {user_col}, created_at DESC
        """).bindparams(uids=tuple(users))
    ).fetchall()

    by_user = {}
    for uid, rid, created_at in pos_rows:
        if uid is None or rid is None:
            continue
        by_user.setdefault(int(uid), []).append((int(rid), created_at))
    return by_user


# ── 수정 2: PK 최댓값이 아닌 실제 최신 행동을 test로 분리 ──
def _holdout_split(user_events):
    """
    leave-one-out: created_at 기준 가장 최근 행동 1건을 test로,
    나머지를 train_seen으로 사용한다.

    user_events: [(recipe_id, created_at), ...] — 이미 created_at DESC로 정렬되어 들어옴
    """
    # 같은 레시피 중복 행동(VIEW 여러 번 등) 제거 — 먼저 나온(=더 최근) 것 우선 유지
    seen_rids = set()
    deduped = []
    for rid, created_at in user_events:
        if rid in seen_rids:
            continue
        seen_rids.add(rid)
        deduped.append((rid, created_at))

    if len(deduped) < 2:
        return None, None

    test_item = deduped[0][0]                       # 가장 최근 행동
    train_seen = {rid for rid, _ in deduped[1:]}     # 나머지 전부
    return test_item, train_seen


# 라우트 
@bp.get("/api/admin/metrics")
def admin_metrics():
    try:
        with db.engine.connect() as conn:
            found = _find_positive_table(conn)
            total_recipes = _get_total_recipe_count(conn)

            if not found:
                return _no_cache_json(_empty_result(
                    total_recipes=total_recipes,
                    reason="positive table not found (need user_id + rcp_sno/recipe_id columns)",
                ))

            pos_by_user = _load_user_positives(
                conn,
                table=found["table"],
                user_col=found["user_col"],
                recipe_col=found["recipe_col"],
            )

            if not pos_by_user:
                return _no_cache_json(_empty_result(
                    total_recipes=total_recipes,
                    pos_table=found,
                    reason="need at least 2 positives per user for holdout eval",
                ))

        recall_sum = ndcg_sum = hit_sum = 0.0
        union_recs = set()
        positives_total = users_evaluated = users_skipped = failed_users = 0

        for uid, events in pos_by_user.items():
            positives_total += len(events)

            test_item, train_seen = _holdout_split(events)
            if test_item is None:
                users_skipped += 1
                continue

            try:
                rec_ids = recommend_mmr(
                    user_id=uid,
                    k=K,
                    lambda_mmr=LAMBDA_MMR,
                    seed_recipe_id=None,
                    filter_seen=False,  # train_seen으로 이미 제외했으므로 test_item을 후보에 남겨야 함
                    exclude_ids=train_seen,
                )
            except Exception:
                failed_users += 1
                continue

            rec_ids = [int(x) for x in rec_ids] if rec_ids else []
            union_recs.update(rec_ids)

            gt_test = {int(test_item)}
            hit = any(rid in gt_test for rid in rec_ids[:K])

            recall_sum += 1.0 if hit else 0.0
            hit_sum += 1.0 if hit else 0.0
            ndcg_sum += _ndcg_at_k(rec_ids, gt_test, K)
            users_evaluated += 1

        n = max(users_evaluated, 1)
        coverage = len(union_recs) / total_recipes if total_recipes > 0 else 0.0

        debug_reason = "ok" if users_evaluated > 0 else (
            f"usersEvaluated=0 — failedRecommendUsers={failed_users}"
        )

        return _no_cache_json({
            "recallAt10": recall_sum / n,
            "ndcgAt10": ndcg_sum / n,
            "hitRateAt10": hit_sum / n,
            "coverage": coverage,
            "usersEvaluated": users_evaluated,
            "usersSkipped": users_skipped,
            "failedRecommendUsers": failed_users,
            "positivesTotal": positives_total,
            "avgPositivesPerUser": positives_total / len(pos_by_user) if pos_by_user else 0.0,
            "totalRecipes": total_recipes,
            "posTable": found,
            "debugReason": debug_reason,
        })

    except Exception as e:
        current_app.logger.exception(e)
        return _no_cache_json(_empty_result(reason=f"exception: {e}"))