# Financial Fraud Detection

A machine learning project that builds a fraud detection pipeline on financial transaction data. The pipeline uses logistic regression with class balancing to identify fraudulent TRANSFER and CASH_OUT transactions.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Feature Engineering](#3-feature-engineering)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Building](#5-model-building)
6. [Model Evaluation](#6-model-evaluation)
7. [Model Export](#7-model-export)
8. [Gradio Deployment App](#8-gradio-deployment-app)
9. [Project Structure](#9-project-structure)
10. [How to Run](#10-how-to-run)
11. [Dependencies](#11-dependencies)

---

## 1. Dataset Overview

The dataset is the **AIML Dataset.csv**, a simulated financial transactions dataset. Each row represents one transaction.

**Key columns:**

| Column | Description |
|---|---|
| `step` | Time step of the transaction |
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN) |
| `amount` | Transaction amount |
| `nameOrig` | Sender account ID |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver account ID |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Target label (1 = fraud, 0 = legit) |
| `isFlaggedFraud` | System-flagged fraud indicator |

The dataset has **no null values**. Fraud accounts for roughly **0.13%** of all transactions, making it a heavily imbalanced classification problem.

**Class distribution:**

![Class Distribution](images/01_class_distribution.png)

---

## 2. Exploratory Data Analysis

### 2.1 Transaction Types

Five transaction types exist in the dataset. PAYMENT and CASH_OUT are the most frequent.

![Transaction Types](images/02_transaction_types.png)

### 2.2 Fraud Rate by Transaction Type

Fraud occurs **only in TRANSFER and CASH_OUT** transactions. All other types have zero fraud cases.

![Fraud Rate by Type](images/03_fraud_rate_by_type.png)

### 2.3 Transaction Amount Distribution

Amounts are highly right-skewed. Applying a log transform reveals a cleaner distribution for analysis.

![Amount Distribution](images/04_amount_distribution.png)

### 2.4 Amount vs Fraud

Fraudulent transactions tend to involve larger amounts compared to legitimate ones, though overlap exists.

![Amount vs Fraud](images/05_amount_vs_fraud.png)

### 2.5 Fraud in TRANSFER and CASH_OUT

Filtering down to the two fraud-relevant types, fraud is rare but concentrated in CASH_OUT.

![Transfer and Cash Out Fraud](images/07_transfer_cashout_fraud.png)

### 2.6 Correlation Matrix

`oldbalanceOrg` and `newbalanceOrig` are strongly correlated. `isFraud` has low linear correlation with all features, which motivates engineered features.

![Correlation Heatmap](images/06_correlation_heatmap.png)

---

## 3. Feature Engineering

Two balance difference features are added to capture suspicious balance movements:

```python
df["balancedDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balancedDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
```

A fraud pattern to watch: the sender's balance drops to zero after a TRANSFER or CASH_OUT while the receiver's balance does not increase proportionally. This mismatch is a strong fraud signal.

```python
zero_after_transfer = df[
    (df["oldbalanceOrg"] > 0) &
    (df["newbalanceOrig"] == 0) &
    (df["type"].isin(["TRANSFER", "CASH_OUT"]))
]
```

**Time-step analysis** confirms fraud spikes at certain steps, but `step` is dropped from the model as it leaks temporal ordering without adding predictive generalization.

---

## 4. Data Preprocessing

### 4.1 Columns Dropped

Before modeling, these columns are removed:

| Column | Reason |
|---|---|
| `nameOrig` | High cardinality identifier, no predictive signal |
| `nameDest` | High cardinality identifier, no predictive signal |
| `isFlaggedFraud` | System flag, leaks target information |
| `step` | Temporal leak, dropped after time analysis |

### 4.2 Train/Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, stratify=y
)
```

`stratify=y` preserves the fraud/legit ratio in both splits.

### 4.3 ColumnTransformer

Numeric and categorical features receive separate transformations:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder="drop"
)
```

- **StandardScaler** normalizes numeric columns (amount, balances).
- **OneHotEncoder** encodes the `type` column. `drop="first"` avoids dummy variable collinearity.

---

## 5. Model Building

The full pipeline combines preprocessing and classification in one object:

```python
Pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

Pipeline.fit(x_train, y_train)
```

**Why Logistic Regression?**

- Interpretable baseline for imbalanced fraud detection.
- `class_weight="balanced"` automatically adjusts for the 99.87% / 0.13% imbalance by increasing the penalty for misclassifying the minority class.

**Pipeline architecture:**

![Pipeline Diagram](images/08_pipeline_diagram.png)

---

## 6. Model Evaluation

```python
print(classification_report(y_test, y_pred))
```

The model is evaluated on:

- **Precision** (of flagged fraud, how many are real)
- **Recall** (of all real fraud, how many are caught)
- **F1-Score** (harmonic mean of precision and recall)

In fraud detection, **recall** for the fraud class is the most critical metric. Missing a fraud case is costlier than a false alarm.

![Classification Report](images/09_classification_report.png)

The confusion matrix is also inspected to count false negatives (fraud missed) and false positives (legit flagged as fraud).

---

## 7. Model Export

The trained pipeline is saved as a `.pkl` file for deployment or inference:

```python
import joblib
joblib.dump(Pipeline, "fraud_detection_pipeline.pkl")
```

To load and predict on new data:

```python
import joblib
import pandas as pd

pipeline = joblib.load("fraud_detection_pipeline.pkl")

new_transaction = pd.DataFrame([{
    "type": "CASH_OUT",
    "amount": 250000,
    "oldbalanceOrg": 250000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 250000
}])

prediction = pipeline.predict(new_transaction)
print("Fraud" if prediction[0] == 1 else "Legit")
```

---

## 8. Gradio Deployment App

The notebook `Fraud_Detection_2.ipynb` builds an interactive web app using [Gradio](https://gradio.app/) on top of the saved pipeline.

![Gradio App](images/10_gradio_app.png)

### How it works

The app loads `fraud_detection_pipeline.pkl` and exposes a form where you enter transaction details. It returns a plain-text prediction label immediately.

```python
import gradio as gr
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

def predict_fraud(transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        return "This transaction may be FRAUDULENT."
    else:
        return "This transaction looks LEGITIMATE."
```

### Input fields

| Field | Type | Description |
|---|---|---|
| Transaction Type | Dropdown | PAYMENT, TRANSFER, CASH_OUT, DEPOSIT |
| Amount | Number | Transaction amount |
| Old Balance (Sender) | Number | Sender balance before transaction |
| New Balance (Sender) | Number | Sender balance after transaction |
| Old Balance (Receiver) | Number | Receiver balance before transaction |
| New Balance (Receiver) | Number | Receiver balance after transaction |

### Output

The app returns one of two results:

- `This transaction may be FRAUDULENT.`
- `This transaction looks LEGITIMATE.`

### Run the app locally

```bash
jupyter nbconvert --to script Fraud_Detection_2.ipynb
python Fraud_Detection_2.py
```

The Gradio interface opens in your browser at `http://localhost:7860`. Set `share=True` in `iface.launch()` for a temporary public link. For permanent hosting, deploy to [Hugging Face Spaces](https://huggingface.co/spaces):

```bash
gradio deploy
```

---

## 9. Project Structure

```
fraud-detection/
│
├── financial_fraud_detection5.py   # Full EDA and modeling script
├── Fraud_Detection_2.ipynb         # Gradio deployment app notebook
├── fraud_detection_pipeline.pkl    # Saved model pipeline
├── README.md                       # This file
│
└── images/
    ├── 01_class_distribution.png
    ├── 02_transaction_types.png
    ├── 03_fraud_rate_by_type.png
    ├── 04_amount_distribution.png
    ├── 05_amount_vs_fraud.png
    ├── 06_correlation_heatmap.png
    ├── 07_transfer_cashout_fraud.png
    ├── 08_pipeline_diagram.png
    ├── 09_classification_report.png
    └── 10_gradio_app.png
```

---

## 10. How to Run

**Clone the repository:**

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the full pipeline:**

```bash
python financial_fraud_detection5.py
```

**Load the saved model:**

```python
import joblib
pipeline = joblib.load("fraud_detection_pipeline.pkl")
```

---

## 11. Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn==1.6.1
joblib
gradio
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn==1.6.1 joblib gradio
```

---

## Key Findings

- Fraud appears **only** in TRANSFER and CASH_OUT transactions.
- A balance dropping to zero after a transfer is a strong fraud indicator.
- The dataset is **severely imbalanced** (0.13% fraud), requiring `class_weight="balanced"` in the model.
- Logistic regression with preprocessing delivers a strong, interpretable baseline for this problem.
