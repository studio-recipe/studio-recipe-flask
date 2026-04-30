from flask import Blueprint
from backend.services.train_service import start_train_job, get_train_status

bp = Blueprint("admin_train", __name__)

@bp.post("/api/admin/train-bpr")
def start_train():
    """
    실제 학습은 train_service에서 수행
    - 중복 실행 방지
    """
    started = start_train_job()
    status = get_train_status()

    return {
        "ok": True,
        "message": "started" if started else "already running",
        "state": status,
    }, 200

@bp.get("/api/admin/train-bpr/status")
def train_status():
    status = get_train_status()
    return {
        "ok": True,
        "state": status,
    }, 200