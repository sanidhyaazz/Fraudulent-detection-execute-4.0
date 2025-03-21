# Fraud Detection API

A RESTful API for real-time and batch fraud detection with configurable rules and ML model integration.

## Overview

This API implements the backend for a Fraud Detection, Alert, and Monitoring (FDAM) system with:

1. **Real-time Fraud Detection API**: Classifies a single transaction as fraudulent or legitimate with low latency (<300ms)
2. **Batch Fraud Detection API**: Processes multiple transactions in parallel
3. **Rule Engine API**: Configure fraud detection rules via API

## API Endpoints

### Real-time Fraud Detection API

**Endpoint**: `/api/fraud-detection`
**Method**: POST
**Input**: Single transaction in JSON format
**Output**: 
```json
{
  "transaction_id": "<string>",
  "is_fraud": "<boolean>",
  "fraud_source": "<string: 'rule'/'model'>",
  "fraud_reason": "<string>",
  "fraud_score": "<float>"
}
```

### Batch Fraud Detection API

**Endpoint**: `/api/fraud-detection/batch`
**Method**: POST
**Input**: Array of transactions in JSON format
**Output**: 
```json
{
  "<transaction_id>": {
    "is_fraud": "<boolean>",
    "fraud_reason": "<string>",
    "fraud_score": "<float>"
  },
  ...
}
```

### Rule Engine Configuration

**API Endpoints**: 
- GET `/api/rules`: Retrieve current rules
- POST `/api/rules`: Update rules

## Installation

1. Install dependencies:
```bash
pip install flask pandas numpy scikit-learn joblib sqlite3
```

2. Create folders for models and data:
```bash
mkdir -p models data
```

3. Put your trained models in the models directory:
   - `models/optimized_logistic_regression_model.pkl` (optional)
   - `models/feature_scaler.pkl` (optional)
   - `models/hist_gradient_boosting_model.pkl` (optional)

## Running the API

Start the API server:

```bash
python fraud_detection_api.py
```

The server will run on http://localhost:5000 by default.

## Testing

Use the included test script to test the API:

```bash
python test_api.py
```

This will:
1. Test the real-time fraud detection API with both low-risk and high-risk transactions
2. Test the batch fraud detection API with multiple transactions

## Integration with Frontend

### Transaction Format

The API expects transaction data in the following format:

```json
{
  "transaction_id": "unique-transaction-id",
  "transaction_date": "2023-10-15 14:30:45",
  "transaction_amount": 1500.00,
  "transaction_channel": "web", 
  "transaction_payment_mode_anonymous": 3,
  "payment_gateway_bank_anonymous": 5,
  "payer_browser_anonymous": 2000,
  "payer_email_anonymous": "user@example.com",
  "payer_mobile_anonymous": "XXXXX123.0",
  "payee_id_anonymous": "ANON_42"
}
```

### API Response

The API responds with fraud detection results that include:

- `is_fraud`: Whether the transaction is flagged as fraudulent
- `fraud_source`: The source of the detection (rule/model)
- `fraud_reason`: The reason for flagging the transaction
- `fraud_score`: A score indicating the confidence in the fraud detection

## Database

All detection results are automatically stored in an SQLite database (`data/fraud_detection.db`) for monitoring and analysis. 