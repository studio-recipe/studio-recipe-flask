from flask import Blueprint, request, jsonify, current_app
from backend.services.recommender_mmr import recommend_mmr

bp = Blueprint("recommendations", __name__, url_prefix="/api")


@bp.get("/recommend")
def recommend():
    user_id = int(request.args.get("userId", "0"))
    k = int(request.args.get("k", "10"))
    lam = float(request.args.get("lambda", "0.8"))

    seed = request.args.get("seedRecipeId")
    seed_recipe_id = int(seed) if seed and seed.strip() else None

    try:
        ids = recommend_mmr(
            user_id=user_id,
            k=k,
            lambda_mmr=lam,
            seed_recipe_id=seed_recipe_id,
        ) or []
        return jsonify([int(x) for x in ids if x is not None]), 200

    except Exception as e:
        current_app.logger.exception(e)
        return jsonify([]), 200