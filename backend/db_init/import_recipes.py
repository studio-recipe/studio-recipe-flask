# backend/db_init/import_recipes.py
import csv
from datetime import datetime
from pathlib import Path

from backend.app import create_app, db
from backend.models import Recipe

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "recipe_data_241226.csv"


def parse_int(value, default=0):
    if value is None:
        return default
    s = str(value).strip()
    if s == "":
        return default
    try:
        return int(s)
    except ValueError:
        return default


def parse_datetime_yyyyMMdd(raw: str) -> datetime:
    if not raw:
        return datetime.now()
    s = str(raw).strip()
    try:
        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d")
        elif len(s) == 14:
            return datetime.strptime(s, "%Y%m%d%H%M%S")
        else:
            return datetime.now()
    except Exception:
        return datetime.now()


def run():
    app = create_app()
    with app.app_context():
        print(f"CSV 파일 경로: {CSV_PATH}")
        if not CSV_PATH.exists():
            print("CSV 파일을 찾을 수 없습니다.")
            return

        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            batch = []
            batch_size = 500
            total = 0

            for row in reader:
                recipe = Recipe(
                    rcp_sno=parse_int(row.get("RCP_SNO")),
                    rcp_ttl=row.get("RCP_TTL"),
                    ckg_nm=row.get("CKG_NM"),
                    inq_cnt=row.get("INQ_CNT"),
                    rcmm_cnt=row.get("RCMM_CNT"),
                    ckg_mth_acto_nm=row.get("CKG_MTH_ACTO_NM"),
                    ckg_mtrl_acto_nm=row.get("CKG_MTRL_ACTO_NM"),
                    ckg_knd_acto_nm=row.get("CKG_KND_ACTO_NM"),
                    ckg_mtrl_cn=row.get("CKG_MTRL_CN"),
                    ckg_inbun_nm=row.get("CKG_INBUN_NM"),
                    ckg_dodf_nm=row.get("CKG_DODF_NM"),
                    ckg_time_nm=row.get("CKG_TIME_NM"),
                    first_reg_dt=parse_datetime_yyyyMMdd(row.get("FIRST_REG_DT")),
                    rcp_img_url=row.get("RCP_IMG_URL"),
                )

                batch.append(recipe)
                total += 1

                if len(batch) >= batch_size:
                    db.session.add_all(batch)
                    db.session.commit()
                    print(f"현재까지 {total}개 레코드 저장 완료")
                    batch.clear()

            if batch:
                db.session.add_all(batch)
                db.session.commit()
                print(f"최종 {total}개 레코드 저장 완료")


if __name__ == "__main__":
    run()
