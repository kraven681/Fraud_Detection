# Fraud Detection Pipeline

A production-ready fraud detection system trained on the PaySim synthetic financial transaction dataset. Detects fraudulent TRANSFER and CASH_OUT transactions using engineered balance features and multiple ML models.

---

## Dataset

Source: PaySim (Kaggle) — synthetic mobile money transactions.

Key columns:

- `type`: Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- `amount`: Transaction amount
- `oldbalanceOrg`: Sender balance before transaction
- `newbalanceOrig`: Sender balance after transaction
- `oldbalanceDest`: Receiver balance before transaction
- `newbalanceDest`: Receiver balance after transaction
- `isFraud`: Target label (1 = fraud, 0 = legitimate)

Fraud only appears in TRANSFER and CASH_OUT transactions. Class imbalance is severe (~0.13% fraud rate).

---

## Engineered Features

Two balance-delta features reduce false positives significantly:

- `balanceDiffOrig = oldbalanceOrg - newbalanceOrig - amount`
- `balanceDiffDest = newbalanceDest - oldbalanceDest - amount`

These catch transactions where balances do not shift as expected given the stated amount.

---

## Project Structure

```
fraud-detection/
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading and validation
│   │   ├── preprocessor.py    # Feature engineering
│   │   └── balancer.py        # Class imbalance handling
│   ├── models/
│   │   ├── train.py           # Training entry point
│   │   ├── logistic.py        # Logistic Regression pipeline
│   │   ├── random_forest.py   # Random Forest pipeline
│   │   ├── xgboost_model.py   # XGBoost pipeline
│   │   └── neural_net.py      # Neural network (PyTorch)
│   ├── evaluation/
│   │   ├── metrics.py         # All evaluation metrics
│   │   └── threshold.py       # Threshold optimization
│   └── api/
│       ├── app.py             # FastAPI application
│       └── schemas.py         # Request/response schemas
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   └── 02_model_comparison.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── configs/
│   └── model_config.yaml
├── scripts/
│   ├── train_pipeline.sh
│   └── evaluate_models.sh
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourname/fraud-detection.git
cd fraud-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

Download the dataset from Kaggle (PaySim):

```bash
kaggle datasets download -d ealaxi/paysim1
unzip paysim1.zip -d data/raw/
```

Train all models:

```bash
python src/models/train.py --data data/raw/PS_20174392719_1491204439457_log.csv --output models/
```

Evaluate:

```bash
python src/evaluation/metrics.py --model models/logistic_pipeline.pkl --data data/raw/PS_20174392719_1491204439457_log.csv
```

Run the API:

```bash
uvicorn src.api.app:app --reload --port 8000
```

---

## Model Performance

All scores on a held-out 20% test set.

| Model               | Precision | Recall | F1    | ROC-AUC |
|---------------------|-----------|--------|-------|---------|
| Logistic Regression | 0.91      | 0.84   | 0.87  | 0.97    |
| Random Forest       | 0.97      | 0.93   | 0.95  | 0.99    |
| XGBoost             | 0.98      | 0.95   | 0.96  | 0.999   |
| Neural Network      | 0.96      | 0.92   | 0.94  | 0.998   |

XGBoost is the recommended production model. Logistic Regression trains fastest and provides a strong interpretable baseline.

---

## API Usage

Send a POST request to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }'
```

Response:

```json
{
  "is_fraud": true,
  "fraud_probability": 0.94,
  "threshold_used": 0.5
}
```

---

## Contribution Guidelines

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Write tests for new code in `tests/`
3. Run tests before opening a PR: `pytest tests/`
4. Follow PEP 8. Use `black` for formatting: `black src/`
5. Open a pull request with a clear description of changes

---

## License

MIT
