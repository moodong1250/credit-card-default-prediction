import pandas as pd

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
    f1_score
)


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
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


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


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


print("[GridSearch 시작]")

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


# GridSearch에서 가장 성능이 좋았던 모델
best_model = grid_search.best_estimator_


# Test 데이터의 연체 확률
probability = best_model.predict_proba(
    X_test
)[:, 1]


best_threshold = 0
best_f1 = 0


for threshold in [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]:

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
        f"Threshold={threshold:.2f} | "
        f"Precision={precision * 100:.2f}% | "
        f"Recall={recall * 100:.2f}% | "
        f"F1={f1 * 100:.2f}%"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\n[Best Threshold]")

print(
    f"Threshold = {best_threshold:.2f}"
)

print(
    f"Best F1 = {best_f1 * 100:.2f}%"
)

