from flask_sqlalchemy import SQLAlchemy

# 프로젝트 전체에서 단 하나만 쓰는 공통 db 인스턴스
db = SQLAlchemy()