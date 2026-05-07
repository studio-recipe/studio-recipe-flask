import csv
from datetime import datetime
from pathlib import Path

from backend.app import create_app, db
from backend.models import Recipe

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "recipe_data_241226.csv"
BATCH_SIZE = 500


def parse_int(value, default=0):
    if value is None:
        return default
    s = str(value).strip()
    try:
        return int(s) if s else default
    except ValueError:
        return default


def parse_datetime(raw: str) -> datetime:
    s = str(raw).strip() if raw else ""
    try:
        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d")
        if len(s) == 14:
            return datetime.strptime(s, "%Y%m%d%H%M%S")
    except Exception:
        pass
    return datetime.now()


def run():
    app = create_app()
    with app.app_context():
        # 테이블이 없으면 생성
        db.create_all()
        print("테이블 확인/생성 완료")

        if not CSV_PATH.exists():
            print(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
            return

        print(f"CSV 파일 경로: {CSV_PATH}")

        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            batch = []
            total = 0

            for row in reader:
                batch.append(Recipe(
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
                    first_reg_dt=parse_datetime(row.get("FIRST_REG_DT")),
                    rcp_img_url=row.get("RCP_IMG_URL"),
                ))
                total += 1

                if len(batch) >= BATCH_SIZE:
                    db.session.add_all(batch)
                    db.session.commit()
                    print(f"{total}개 저장 완료")
                    batch.clear()

            if batch:
                db.session.add_all(batch)
                db.session.commit()

        print(f"최종 {total}개 레코드 적재 완료")


if __name__ == "__main__":
    run()