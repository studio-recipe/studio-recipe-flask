from sqlalchemy import text
from backend.extensions import db

def get_latest_metrics():
    sql = text("""
        SELECT
            recall_at_10,
            ndcg_at_10,
            hit_rate_at_10,
            coverage,
            created_at
        FROM recommend_metrics
        ORDER BY created_at DESC
        LIMIT 1
    """)

    row = db.session.execute(sql).mappings().first()

    if not row:
        return {
            "recallAt10": None,
            "ndcgAt10": None,
            "hitRateAt10": None,
            "coverage": None,
            "createdAt": None,
            "message": "recommend_metrics에 데이터가 없습니다."
        }

    return {
        "recallAt10": float(row["recall_at_10"]) if row["recall_at_10"] is not None else None,
        "ndcgAt10": float(row["ndcg_at_10"]) if row["ndcg_at_10"] is not None else None,
        "hitRateAt10": float(row["hit_rate_at_10"]) if row["hit_rate_at_10"] is not None else None,
        "coverage": float(row["coverage"]) if row["coverage"] is not None else None,
        "createdAt": str(row["created_at"]) if row["created_at"] is not None else None,
    }
