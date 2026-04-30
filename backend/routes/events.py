from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
from backend.extensions import db

bp = Blueprint("events", __name__, url_prefix="/api/events")

@bp.get("")
def list_events():
    return jsonify([])

@bp.post("/view")
def log_view():
    """
    테스트용 VIEW 이벤트 적재 (스프링 이벤트가 없을 때만 사용)
    body: { "userId": 1, "recipeId": 23191 }
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("userId")
    recipe_id = data.get("recipeId")
    if user_id is None or recipe_id is None:
        return jsonify({"ok": False, "error": "userId and recipeId required"}), 400

    sql = text("""
        INSERT INTO USER_REFERENCES
        (USER_ID, RCP_SNO, PREFERENCE_TYPE, CREATED_AT, MODIFIED_AT)
        VALUES (:user_id, :recipe_id, 'VIEW', NOW(), NOW())
    """)

    try:
        db.session.execute(sql, {"user_id": int(user_id), "recipe_id": int(recipe_id)})
        db.session.commit()
        return jsonify({"ok": True, "type": "VIEW"}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception(e)
        return jsonify({"ok": False, "error": str(e)}), 400

@bp.post("/like")
def log_like():
    """
    테스트용 LIKE 이벤트 적재 (스프링 이벤트가 없을 때만 사용)
    body: { "userId": 1, "recipeId": 23191 }
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("userId")
    recipe_id = data.get("recipeId")
    if user_id is None or recipe_id is None:
        return jsonify({"ok": False, "error": "userId and recipeId required"}), 400

    sql = text("""
        INSERT INTO USER_REFERENCES
        (USER_ID, RCP_SNO, PREFERENCE_TYPE, CREATED_AT, MODIFIED_AT)
        VALUES (:user_id, :recipe_id, 'LIKE', NOW(), NOW())
    """)

    try:
        db.session.execute(sql, {"user_id": int(user_id), "recipe_id": int(recipe_id)})
        db.session.commit()
        return jsonify({"ok": True, "type": "LIKE"}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception(e)
        return jsonify({"ok": False, "error": str(e)}), 400
