# backend/db_init/test_db.py
from backend.app import create_app, db
from sqlalchemy import text 

app = create_app()

with app.app_context():
    db.session.execute(text("SELECT 1"))  # text()로 감싸기
    print("DB 연결 성공")
