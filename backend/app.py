import os
from flask import Flask
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics

from backend.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from backend.extensions import db

# Feature Flag: 캐시 ON/OFF (docker-compose CACHE_ENABLED 환경변수로 제어)
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"


def create_app():
    app = Flask(__name__)
    PrometheusMetrics(app)

    app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS

    CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173"]}}, supports_credentials=True)

    db.init_app(app)

    from backend.routes.recommendations import bp as rec_bp
    from backend.routes.admin_train import bp as admin_train_bp
    from backend.routes.admin_metrics import bp as admin_metrics_bp

    app.register_blueprint(rec_bp)
    app.register_blueprint(admin_train_bp)
    app.register_blueprint(admin_metrics_bp)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    try:
        from backend.services.recommender_mmr import warmup_recipe_cache
        with app.app_context():
            warmup_recipe_cache(force=True)
    except Exception as e:
        app.logger.warning(f"[WARMUP] recipe cache warmup failed: {e}")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)