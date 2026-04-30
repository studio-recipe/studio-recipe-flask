import threading
import time
import traceback

TRAIN_STATE = {
    "running": False,
    "lastStartedAt": None,
    "lastFinishedAt": None,
    "lastSuccess": None,
    "lastError": None,
}

_LOCK = threading.Lock()

def _run_train():
    with _LOCK:
        TRAIN_STATE["running"] = True
        TRAIN_STATE["lastStartedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        TRAIN_STATE["lastFinishedAt"] = None
        TRAIN_STATE["lastSuccess"] = None
        TRAIN_STATE["lastError"] = None

    try:
        from backend.ml.train_bpr import train_bpr
        train_bpr()

        # 학습이 끝나면 추천 캐시를 반드시 날린다 (즉시 반영)
        try:
            from backend.services.recommender_mmr import invalidate_recipe_cache, warmup_recipe_cache
            invalidate_recipe_cache()
            warmup_recipe_cache(force=True)
        except Exception as e:
            print(f"[TRAIN] cache invalidate failed: {e}")

        with _LOCK:
            TRAIN_STATE["lastSuccess"] = True

    except Exception:
        with _LOCK:
            TRAIN_STATE["lastSuccess"] = False
            TRAIN_STATE["lastError"] = traceback.format_exc()

    finally:
        with _LOCK:
            TRAIN_STATE["running"] = False
            TRAIN_STATE["lastFinishedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")

def start_train_job():
    with _LOCK:
        if TRAIN_STATE["running"]:
            return False
        th = threading.Thread(target=_run_train, daemon=True)
        th.start()
        return True

def get_train_status():
    with _LOCK:
        return dict(TRAIN_STATE)