import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================
# 데이터 불러오기
# =========================

df = pd.read_excel(
    "data/default of credit card clients.xls",
    header=1
)


X = df.drop(
    columns=[
        "default payment next month",
        "ID"
    ]
)

y = df["default payment next month"]


# =========================
# 컬럼 구분
# =========================

categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

numeric_cols = X.columns.difference(
    categorical_cols
)


# =========================
# 전처리
# =========================

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


# =========================
# 1차 분리
# 80% Train+Validation
# 20% Test
# =========================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 2차 분리
# 전체 기준:
# Train 60%
# Validation 20%
# Test 20%
# =========================

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# =========================
# Random Forest Pipeline
# =========================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# =========================
# Hyperparameter 후보
# =========================

param_grid = {
    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        5,
        10,
        None
    ],

    "model__min_samples_leaf": [
        1,
        5
    ]
}


# =========================
# GridSearch
# Train 데이터만 사용
# =========================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


print("\n[GridSearch 시작]")

grid_search.fit(
    X_train,
    y_train
)


print("\nBest Parameters")
print(
    grid_search.best_params_
)

print("\nBest CV F1")
print(
    f"{grid_search.best_score_ * 100:.2f}%"
)


best_model = grid_search.best_estimator_


# =========================
# Validation에서 Threshold 선택
# =========================

val_probability = best_model.predict_proba(
    X_val
)[:, 1]


best_threshold = 0
best_f1 = 0


print("\n[Validation Threshold 비교]")

for threshold in [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]:

    prediction = (
        val_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        prediction
    )

    recall = recall_score(
        y_val,
        prediction
    )

    f1 = f1_score(
        y_val,
        prediction
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision * 100:.2f}% | "
        f"Recall={recall * 100:.2f}% | "
        f"F1={f1 * 100:.2f}%"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\n[Best Validation Threshold]")
print(
    f"Threshold = {best_threshold:.2f}"
)
print(
    f"Validation F1 = {best_f1 * 100:.2f}%"
)


# =========================
# 마지막 Test 평가
# =========================

test_probability = best_model.predict_proba(
    X_test
)[:, 1]


test_prediction = (
    test_probability >= best_threshold
).astype(int)


test_precision = precision_score(
    y_test,
    test_prediction
)

test_recall = recall_score(
    y_test,
    test_prediction
)

test_f1 = f1_score(
    y_test,
    test_prediction
)

test_cm = confusion_matrix(
    y_test,
    test_prediction
)


print("\n=========================")
print("      FINAL TEST")
print("=========================")

print(
    f"Threshold : {best_threshold:.2f}"
)
print(
    f"Precision : {test_precision * 100:.2f}%"
)
print(
    f"Recall    : {test_recall * 100:.2f}%"
)
print(
    f"F1 Score  : {test_f1 * 100:.2f}%"
)

print("\nConfusion Matrix")
print(test_cm)


# =========================
# 최종 배포용 데이터
# =========================

X_final_train = pd.concat(
    [
        X_train,
        X_val
    ]
)

y_final_train = pd.concat(
    [
        y_train,
        y_val
    ]
)


print(
    "\nFinal Training Data:",
    X_final_train.shape
)


# =========================
# 최종 모델 재학습
# =========================

final_pipeline = grid_search.best_estimator_

final_pipeline.fit(
    X_final_train,
    y_final_train
)


# =========================
# 모델 저장
# =========================

joblib.dump(
    final_pipeline,
    "credit_default_model.joblib"
)

joblib.dump(
    best_threshold,
    "credit_default_threshold.joblib"
)


print("\n최종 모델 저장 완료")