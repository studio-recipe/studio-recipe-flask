# backend/routes/admin_metrics.py
from flask import Blueprint, current_app, jsonify, make_response
from sqlalchemy import text
from backend.extensions import db

from backend.services.recommender_mmr import recommend_mmr

bp = Blueprint("admin_metrics", __name__)

# ---------- helpers ----------

def _dcg(binary_rels):
    # binary_rels: [0/1, 0/1, ...]
    s = 0.0
    for i, rel in enumerate(binary_rels, start=1):
        if rel:
            s += 1.0 / ((i + 1) ** 0.5)
    return s

def _ndcg_at_k(rec_ids, gt_set, k):
    topk = rec_ids[:k]
    rels = [1 if rid in gt_set else 0 for rid in topk]
    dcg = _dcg(rels)
    ideal = [1] * min(len(gt_set), k)
    idcg = _dcg(ideal)
    if idcg <= 0:
        return 0.0
    return dcg / idcg

def _find_positive_table(conn):
    """
    (user_id, recipe_id/rcp_sno) 형태의 테이블을 자동 탐색
    - MariaDB/MySQL 기준 information_schema
    """
    dbname = conn.execute(text("SELECT DATABASE()")).scalar()
    if not dbname:
        return None

    rows = conn.execute(
        text("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = :db
          AND (LOWER(column_name) IN ('user_id','rcp_sno','recipe_id'))
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

def _load_user_positives(conn, table, user_col, recipe_col, max_users=200, min_pos=2):
    """
    유저별 positive(좋아요 등) 레시피 set을 로드
    - 너무 많으면 샘플링(최대 max_users)
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
        SELECT {user_col} AS uid, {recipe_col} AS rid
        FROM {table}
        WHERE {user_col} IN :uids
        """).bindparams(uids=tuple(users))
    ).fetchall()

    by_user = {}
    for uid, rid in pos_rows:
        if uid is None or rid is None:
            continue
        by_user.setdefault(int(uid), set()).add(int(rid))
    return by_user

def _holdout_split(gt_set):
    """
    B안: leave-one-out 홀드아웃
    - gt_set에서 1개를 test로 떼고 나머지는 train_seen으로 반환
    - (현재는 "가장 큰 id"를 test로 잡음: 재현성 확보)
    """
    gt_list = sorted([int(x) for x in gt_set])
    if len(gt_list) < 2:
        return None, None
    test_item = int(gt_list[-1])
    train_seen = set(int(x) for x in gt_list[:-1])
    return test_item, train_seen

def _no_cache_json(payload, status=200):
    resp = make_response(jsonify(payload), status)
    # 캐시 고정값 방지
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# ---------- route ----------

@bp.get("/api/admin/metrics")
def admin_metrics():
    """
    반환 형식 (Spring이 기대하는 키):
      - recallAt10, ndcgAt10, hitRateAt10, coverage
    + 디버그 필드:
      - usersEvaluated, positivesTotal, avgPositivesPerUser, totalRecipes, posTable
    """
    k = 10
    lam = 0.8  # 지표는 기본값으로 계산

    try:
        eng = db.engine
        with eng.connect() as conn:
            found = _find_positive_table(conn)
            total_recipes = _get_total_recipe_count(conn)

            if not found:
                return _no_cache_json({
                    "recallAt10": 0.0,
                    "ndcgAt10": 0.0,
                    "hitRateAt10": 0.0,
                    "coverage": 0.0,
                    "usersEvaluated": 0,
                    "positivesTotal": 0,
                    "avgPositivesPerUser": 0.0,
                    "totalRecipes": total_recipes,
                    "posTable": None,
                    "debugReason": "positive table not found (need user_id + rcp_sno/recipe_id columns)",
                }, 200)

            pos_by_user = _load_user_positives(
                conn,
                table=found["table"],
                user_col=found["user_col"],
                recipe_col=found["recipe_col"],
                max_users=200,
                min_pos=2,
            )

            users = list(pos_by_user.keys())
            if not users:
                return _no_cache_json({
                    "recallAt10": 0.0,
                    "ndcgAt10": 0.0,
                    "hitRateAt10": 0.0,
                    "coverage": 0.0,
                    "usersEvaluated": 0,
                    "positivesTotal": 0,
                    "avgPositivesPerUser": 0.0,
                    "totalRecipes": total_recipes,
                    "posTable": found,
                    "debugReason": "need at least 2 positives per user for holdout eval",
                }, 200)

            recall_sum = 0.0
            ndcg_sum = 0.0
            hit_sum = 0.0
            union_recs = set()
            positives_total = 0
            users_evaluated = 0
            users_skipped = 0
            failed_users = 0

            for uid in users:
                gt = pos_by_user.get(uid, set())
                positives_total += len(gt)
                if len(gt) < 2:
                    users_skipped += 1
                    continue

                test_item, train_seen = _holdout_split(gt)
                if test_item is None:
                    users_skipped += 1
                    continue

                try:
                    # 핵심: train_seen만 제외하고, test_item은 추천 후보에 남겨둔다.
                    rec_ids = recommend_mmr(
                        user_id=uid,
                        k=k,
                        lambda_mmr=lam,
                        seed_recipe_id=None,
                        filter_seen=True,
                        exclude_ids=train_seen
                    )
                except Exception:
                    failed_users += 1
                    continue

                rec_ids = [int(x) for x in rec_ids] if rec_ids else []
                union_recs.update(rec_ids)

                gt_test = {int(test_item)}
                hits = [rid for rid in rec_ids[:k] if rid in gt_test]

                hitrate = 1.0 if len(hits) > 0 else 0.0
                recall = hitrate
                ndcg = _ndcg_at_k(rec_ids, gt_test, k)

                recall_sum += recall
                hit_sum += hitrate
                ndcg_sum += ndcg
                users_evaluated += 1

            n = max(users_evaluated, 1)
            recall_at_10 = recall_sum / n
            hit_at_10 = hit_sum / n
            ndcg_at_10 = ndcg_sum / n

            coverage = 0.0
            if total_recipes > 0:
                coverage = len(union_recs) / float(total_recipes)

            debug_reason = "ok"
            if users_evaluated == 0:
                debug_reason = (
                    "usersEvaluated=0 (common reasons: user embedding missing, "
                    "recipe embeddings empty, or recommend_mmr failing). "
                    f"failedRecommendUsers={failed_users}"
                )
                print("uid:", uid)
                print("gt:", gt)
                print("test_item:", test_item)
                print("train_seen size:", len(train_seen))
                print("rec_ids:", rec_ids[:10])
                print("hit:", test_item in rec_ids[:k])

            return _no_cache_json({
                "recallAt10": recall_at_10,
                "ndcgAt10": ndcg_at_10,
                "hitRateAt10": hit_at_10,
                "coverage": coverage,
                "usersEvaluated": users_evaluated,
                "usersSkipped": users_skipped,
                "failedRecommendUsers": failed_users,
                "positivesTotal": positives_total,
                "avgPositivesPerUser": (positives_total / float(len(users))) if users else 0.0,
                "totalRecipes": total_recipes,
                "posTable": found,
                "debugReason": debug_reason,
            }, 200)


    except Exception as e:
        current_app.logger.exception(e)
        return _no_cache_json({
            "recallAt10": 0.0,
            "ndcgAt10": 0.0,
            "hitRateAt10": 0.0,
            "coverage": 0.0,
            "usersEvaluated": 0,
            "positivesTotal": 0,
            "avgPositivesPerUser": 0.0,
            "totalRecipes": 0,
            "posTable": None,
            "debugReason": f"exception: {str(e)}",
        }, 200)
