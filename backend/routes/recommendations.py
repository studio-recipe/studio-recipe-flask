from flask import Blueprint, request, jsonify, current_app
from backend.services.recommender_mmr import recommend_mmr

# 기존 서비스에 이미 popular fallback 로직이 있다면 그걸 사용
try:
    from backend.services.recommender_cf import recommend_for_user as recommend_fallback
except Exception:
    recommend_fallback = None

bp = Blueprint("recommendations", __name__, url_prefix="/api")

@bp.get("/recommend")
def recommend():
    user_id = int(request.args.get("userId", "0"))
    k = int(request.args.get("k", "10"))
    lam = float(request.args.get("lambda", "0.8"))

    seed = request.args.get("seedRecipeId")
    seed_recipe_id = int(seed) if seed is not None and str(seed).strip() != "" else None

    try:
        ids = recommend_mmr(
            user_id=user_id,
            k=k,
            lambda_mmr=lam,
            seed_recipe_id=seed_recipe_id,
        ) or []

        # 정상: list[int]
        return jsonify([int(x) for x in ids if x is not None]), 200

    except Exception as e:
        # 여기서 500 내면 프론트 추천이 통째로 죽음 → 절대 500 내지 말기
        current_app.logger.exception(e)

        # fallback 가능하면 fallback (프론트가 dict 리스트를 기대하는 경우가 있을 수 있어 방어)
        if recommend_fallback is not None:
            try:
                items = recommend_fallback(user_id=user_id, size=k) or []
                # fallback이 dict 형태면 그대로 반환
                return jsonify(items), 200
            except Exception as e2:
                current_app.logger.exception(e2)

        # fallback도 없으면 빈 배열이라도 200
        return jsonify([]), 200