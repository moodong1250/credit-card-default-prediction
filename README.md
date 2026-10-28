# Credit Card Default Prediction

신용카드 고객 정보를 기반으로 다음 달 연체 가능성을 예측하는 머신러닝 테스트 프로젝트


## 데이터

- 데이터 수: 30,000명
- Target
  - 0 = 연체하지 않음
  - 1 = 연체

### 주요 Feature

- 신용 한도
- 나이
- 최근 6개월 연체 상태
- 청구 금액
- 납부 금액
- 결혼 여부
- 학력
- 성별


## 테스트한 모델

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost


## 모델별 테스트 결과

5-Fold Cross Validation의 F1 Score를 기준으로 비교

| Model | CV F1 Score |
|---|---:|
| Logistic Regression | 47.66% |
| Random Forest | **52.38%** |
| Gradient Boosting | 47.37% |
| XGBoost | 47.34% |


## 추가로 테스트한 내용

- StandardScaler
- Class Weight
- Threshold 조정
- Cross Validation
- Feature Engineering → 최종 제외
- GridSearchCV
- Train / Validation / Test 분리


### Feature Engineering 추가 테스트

- LATE_COUNT
- AVG_BILL_AMT
- AVG_PAY_AMT
- PAY_TO_BILL_RATIO

기존 Feature만 사용했을 때보다 성능이 크게 개선되지 않아 최종 모델에서는 제외


## 최종 모델

Model: Random Forest

### Best Parameters

- n_estimators = 200
- max_depth = 10
- min_samples_leaf = 5
- class_weight = balanced

### Validation Threshold

- 0.55


## Final Test Result

| Precision | Recall | F1 Score |
|---:|---:|---:|
| 53.93% | 53.81% | **53.87%** |

Validation에서 Threshold를 선택한 뒤,
마지막 Test 데이터는 Threshold 선택에 사용하지 않고
최종 성능 확인에만 사용했습니다.


## 모델 저장

최종 학습된 모델과 Threshold를 joblib으로 저장했습니다.

- `credit_default_model.joblib`
- `credit_default_threshold.joblib`


## SQLite DB 연결

기존에는 Excel 파일 전체를 불러온 뒤 고객 ID를 검색했지만,
최종 단계에서는 데이터를 SQLite DB로 변환했습니다.

현재 예측 흐름:

```text
고객 ID 입력
↓
SQLite DB에서 고객 조회
↓
저장된 Random Forest 모델
↓
연체 확률 계산
↓
위험 등급 출력

```

## 최종 테스트 예시

```text

조회할 고객 ID : 31
고객 ID: 31
연체 확률: 17.54%
위험 등급: 저위험
```


`experiments/`에는 모델 비교, 특성 공학 및 하이퍼파라미터 탐색 코드가 있습니다.
