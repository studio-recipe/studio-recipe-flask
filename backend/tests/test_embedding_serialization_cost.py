"""
실제 DB의 레시피 임베딩을 가져와 직렬화/역직렬화 비용을 측정.

backend/services/recommender_mmr.py 의 _load_from_db (DB 조회 + 파싱),
_redis_save_embeddings / _redis_load_embeddings 가 Redis에 넣기 전/꺼낸 후
수행하는 직렬화(json.dumps + tobytes) / 역직렬화(json.loads + frombuffer)
로직과 동일한 코드를, recipe_embeddings 테이블의 실제 데이터로 측정한다.
Redis 연결이나 로컬 캐시 적용은 하지 않는다.

사전 조건: backend/config.py 의 DB 설정(DB_HOST/DB_USER/DB_PASS/DB_NAME)으로
MySQL에 접속 가능해야 한다.

실행:
    python -m backend.tests.test_embedding_serialization_cost
"""
import json
import time
import unittest

import numpy as np
from flask import Flask

from backend.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from backend.extensions import db

REPEAT = 20  # 반복 측정 횟수


def _make_app() -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)
    return app


def _to_vec(v):
    """recommender_mmr._to_vec 와 동일한 파싱 로직."""
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


def load_real_embeddings_from_db():
    """recommender_mmr._load_from_db 와 동일한 쿼리로 실제 레시피 임베딩을 가져온다."""
    app = _make_app()
    with app.app_context():
        rows = db.session.execute(
            db.text("SELECT rcp_sno, vector FROM recipe_embeddings")
        ).fetchall()

    ids, vecs = [], []
    for rid, v in rows:
        vec = _to_vec(v)
        if vec is None:
            continue
        ids.append(int(rid))
        vecs.append(vec)

    if not vecs:
        raise RuntimeError("recipe_embeddings 테이블에서 유효한 임베딩을 하나도 읽지 못했습니다.")

    ids_arr = np.array(ids, dtype=np.int64)
    vecs_arr = np.vstack(vecs).astype(np.float32)
    return ids_arr, vecs_arr


def serialize(ids: np.ndarray, vecs: np.ndarray):
    """recommender_mmr._redis_save_embeddings 의 직렬화 로직과 동일."""
    ids_raw = json.dumps(ids.tolist())
    vecs_raw = vecs.tobytes()
    shape_raw = json.dumps(list(vecs.shape))
    return ids_raw, vecs_raw, shape_raw


def deserialize(ids_raw: str, vecs_raw: bytes, shape_raw: str):
    """recommender_mmr._redis_load_embeddings 의 역직렬화 로직과 동일."""
    ids = np.array(json.loads(ids_raw), dtype=np.int64)
    shape = json.loads(shape_raw)
    vecs = np.frombuffer(vecs_raw, dtype=np.float32).reshape(shape)
    return ids, vecs


def benchmark(ids: np.ndarray, vecs: np.ndarray, repeat: int = REPEAT):
    serialize(ids, vecs)  # warm-up

    ser_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ids_raw, vecs_raw, shape_raw = serialize(ids, vecs)
        ser_times.append(time.perf_counter() - t0)

    deser_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        deserialize(ids_raw, vecs_raw, shape_raw)
        deser_times.append(time.perf_counter() - t0)

    return {
        "n": len(ids),
        "size_mb": vecs.nbytes / 1024 / 1024,
        "serialize_ms_avg": sum(ser_times) / repeat * 1000,
        "serialize_ms_min": min(ser_times) * 1000,
        "serialize_ms_max": max(ser_times) * 1000,
        "deserialize_ms_avg": sum(deser_times) / repeat * 1000,
        "deserialize_ms_min": min(deser_times) * 1000,
        "deserialize_ms_max": max(deser_times) * 1000,
    }


class EmbeddingSerializationCostTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ids, cls.vecs = load_real_embeddings_from_db()

    def test_roundtrip_correctness(self):
        """직렬화 후 역직렬화하면 DB에서 읽은 원본과 동일해야 한다."""
        ids_raw, vecs_raw, shape_raw = serialize(self.ids, self.vecs)
        ids2, vecs2 = deserialize(ids_raw, vecs_raw, shape_raw)
        np.testing.assert_array_equal(self.ids, ids2)
        np.testing.assert_array_equal(self.vecs, vecs2)

    def test_benchmark_report(self):
        """DB의 전체 레시피 임베딩을 기준으로 직렬화/역직렬화 비용을 출력한다."""
        r = benchmark(self.ids, self.vecs)
        print("\n=== 레시피 임베딩 직렬화/역직렬화 비용 (실측, recipe_embeddings 테이블 전체) ===")
        print(f"레시피 개수(N)        : {r['n']:,}")
        print(f"임베딩 차원           : {self.vecs.shape[1]}")
        print(f"데이터 크기           : {r['size_mb']:.2f} MB")
        print(f"반복 횟수             : {REPEAT}")
        print("-" * 60)
        print(f"직렬화   avg/min/max  : {r['serialize_ms_avg']:.3f} / {r['serialize_ms_min']:.3f} / {r['serialize_ms_max']:.3f} ms")
        print(f"역직렬화 avg/min/max  : {r['deserialize_ms_avg']:.3f} / {r['deserialize_ms_min']:.3f} / {r['deserialize_ms_max']:.3f} ms")
        self.assertGreater(r["serialize_ms_avg"], 0)
        self.assertGreater(r["deserialize_ms_avg"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
