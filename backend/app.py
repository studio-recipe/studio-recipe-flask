from flask import Flask
from flask_cors import CORS

from backend.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from backend.extensions import db

# 워밍업(없어도 서버는 떠야함)
try:
    from backend.services.recommender_mmr import warmup_recipe_cache
except Exception:
    warmup_recipe_cache = None


def create_app():
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS

    CORS(
        app,
        resources={r"/api/*": {"origins": ["http://localhost:5173"]}},
        supports_credentials=True,
    )

    db.init_app(app)

    from backend.routes.recommendations import bp as rec_bp
    from backend.routes.admin_train import bp as admin_train_bp
    from backend.routes.admin_metrics import bp as admin_metrics_bp

    app.register_blueprint(rec_bp)
    app.register_blueprint(admin_train_bp)
    app.register_blueprint(admin_metrics_bp)

    try:
        from backend.routes.events import bp as events_bp
        app.register_blueprint(events_bp, url_prefix="/api/events")
    except Exception:
        pass

    @app.get("/api/health")
    def health():
        return {"ok": True}

    # 서버 시작 전에 캐시 워밍업 (앱 컨텍스트 필요)
    if warmup_recipe_cache is not None:
        with app.app_context():
            warmup_recipe_cache(force=True)

    return app


app = create_app()

if __name__ == "__main__":
    # debug=True면 리로더로 2번 실행되며 워밍업도 2번 돌아서 느려질 수 있음
    # 개발 중이라도 느려지면 use_reloader=False 권장
    app.run(debug=True, use_reloader=False)