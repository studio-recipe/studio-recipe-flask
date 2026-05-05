import os
from flask import Flask
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics

from backend.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from backend.extensions import db

CACHE_ENABLED = os.environ.get('CACHE_ENABLED', 'true').lower() == 'true'

try:
    from backend.services.recommender_mmr import warmup_recipe_cache
except Exception:
    warmup_recipe_cache = None


def create_app():
    app = Flask(__name__)
    metrics = PrometheusMetrics(app)

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

    if warmup_recipe_cache is not None:
        with app.app_context():
            warmup_recipe_cache(force=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)