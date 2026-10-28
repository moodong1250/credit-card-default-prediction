import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_excel( 
    "data/default of credit card clients.xls", 
    header=1 ##두번쨰 줄을 컬럼명 으로 사용
)

X = df.drop(columns=["default payment next month","ID"])

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
        OneHotEncoder(handle_unknown="ignore"),
        categorical_cols
    ),
    (
        "num",
        StandardScaler(),
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
            LogisticRegression(
                max_iter=1000,
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


# 2. 테스트 고객들의 연체 확률 계산
probability = pipeline.predict_proba(
    X_test
)[:, 1]


# 3. 확률을 어떤 기준으로 0/1로 나눌지 설정
threshold_list = [0.3, 0.4, 0.5]


# 4. 각 threshold별 성능 비교
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




print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# Logistic Regression 결과

# 기본 모델, threshold=0.3
# Precision: 55.18%
# Recall: 45.74%
# F1: 50.02%

# class_weight="balanced", threshold=0.5
# Precision: 36.77%
# Recall: 63.23%
# F1: 46.49%

# 5-Fold CV 평균 F1: 47.66%
