# backend/db_init/create_tables.py
from backend.app import create_app, db
import backend.models  # Recipe 등을 로드

app = create_app()

with app.app_context():
    db.create_all()
    print("테이블 생성 완료")
