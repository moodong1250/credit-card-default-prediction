import pandas as pd
import joblib
import sqlite3


# 저장된 모델 불러오기
model = joblib.load(
    "models/credit_default_model.joblib"
)

threshold = joblib.load(
    "models/credit_default_threshold.joblib"
)


# 고객 ID 입력
user_id = int(
    input("조회할 고객 ID : ")
)


# DB 연결
conn = sqlite3.connect(
    "data/credit_clients.db"
)


# 해당 고객 한 명만 조회
query = """
SELECT *
FROM customers
WHERE ID = ?
"""

customer = pd.read_sql_query(
    query,
    conn,
    params=(user_id,)
)


conn.close()


# 고객이 없는 경우
if customer.empty:

    print("존재하지 않는 고객입니다.")


else:

    # 모델 학습에 사용하지 않았던 컬럼 제거
    customer_X = customer.drop(
        columns=[
            "default payment next month",
            "ID"
        ]
    )


    # 연체 확률 계산
    probability = model.predict_proba(
        customer_X
    )[0, 1]


    # 위험 등급
    if probability < 0.30:

        risk_level = "저위험"

    elif probability < threshold:

        risk_level = "주의"

    else:

        risk_level = "고위험"


    print(
        f"\n고객 ID: {user_id}"
    )

    print(
        f"연체 확률: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"위험 등급: {risk_level}"
    )