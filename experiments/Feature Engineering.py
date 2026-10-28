import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


pay_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

bill_cols = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

pay_amt_cols = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]




df = pd.read_excel(
    "data/default of credit card clients.xls",
    header=1
)


# 최근 6개월 동안 연체한 횟수
df["LATE_COUNT"] = (
    df[pay_cols] >= 1
).sum(axis=1)

df["AVG_BILL_AMT"] = (
    df[bill_cols]
    .mean(axis=1)
)

df["AVG_PAY_AMT"] = (
    df[pay_amt_cols]
    .mean(axis=1)
)

df["PAY_TO_BILL_RATIO"] = (
    df["AVG_PAY_AMT"]
    / (df["AVG_BILL_AMT"] + 1)
)

X = df.drop(
    columns=[
        "default payment next month",
        "ID"
    ]
)

y = df["default payment next month"]


categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

numeric_cols = X.columns.difference(
    categorical_cols
)


preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore"
        ),
        categorical_cols
    ),
    (
        "num",
        "passthrough",
        numeric_cols
    )
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\n[Cross Validation]")
print("각 Fold F1:", cv_scores)
print(
    f"평균 F1: {cv_scores.mean() * 100:.2f}%"
)


pipeline.fit(
    X_train,
    y_train
)


probability = pipeline.predict_proba(
    X_test
)[:, 1]


threshold_list = [
    0.3,
    0.4,
    0.5
]


print("\n[Threshold 비교]")

for threshold in threshold_list:

    prediction = (
        probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        prediction
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    print(
        f"Threshold={threshold} | "
        f"Precision={precision * 100:.2f}% | "
        f"Recall={recall * 100:.2f}% | "
        f"F1={f1 * 100:.2f}%"
    )


print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)


# LATE_COUNT
#→ CV F1 52.38 → 52.27
#→ 효과 거의 없음

# AVG_BILL_AMT  
#→ CV F1 52.27 → 52.26
#→ 효과 거의 없음
#→ Threshold 0.4 F1은 오히려 하락