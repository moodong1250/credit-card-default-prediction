import pandas as pd
import sqlite3


# Excel 데이터 불러오기
df = pd.read_excel(
    "data/default of credit card clients.xls",
    header=1
)


# SQLite DB 연결
conn = sqlite3.connect(
    "data/credit_clients.db"
)


# customers 테이블로 저장
df.to_sql(
    "customers",
    conn,
    if_exists="replace",
    index=False
)


# DB 연결 종료
conn.close()


print("DB 생성 완료")