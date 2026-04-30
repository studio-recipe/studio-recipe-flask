import pandas as pd
from sqlalchemy import create_engine
from backend.config import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI)

def load_interactions():
    sql = """
    SELECT USER_ID as user_id,
           RCP_SNO as item_id,
           CASE WHEN PREFERENCE_TYPE = 'LIKE' THEN 1 ELSE 0 END AS label
    FROM user_references
    """
    return pd.read_sql(sql, engine)

def main():
    df = load_interactions()
    # 여기서 user_id, item_id를 index로 매핑하고
    # BPR/NCF 학습 진행
    ...
