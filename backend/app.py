from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
from datetime import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def convert_anonymous_score(value):
    """Convert anonymous score to float, handling string and numeric values."""
    try:
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            # If it's a hash string, convert to numeric value between 0-100
            return float(hash(value) % 100)
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def analyze_fraud(transaction):
    """
    Analyze a transaction for fraud based on rules.
    Returns a tuple of (is_fraud, reason, score)
    """
    logger.debug(f"Analyzing fraud for transaction: {transaction}")
    score = 0.0
    reasons = []
    
    # Check transaction amount
    amount = float(transaction.get('transaction_amount', 0))
    logger.debug(f"Transaction amount: {amount}")
    if amount > 10000:
        score += 0.3
        reasons.append("High transaction amount (>10000)")
    elif amount > 5000:
        score += 0.2
        reasons.append("High transaction amount (>5000)")
    elif amount > 1000:
        score += 0.1
        reasons.append("High transaction amount (>1000)")
    
    # Check transaction channel
    channel = str(transaction.get('transaction_channel', '')).lower()
    logger.debug(f"Transaction channel: {channel}")
    if channel == 'w':
        channel = 'web'
    if channel not in ['mobile', 'web']:
        score += 0.1
        reasons.append("Unusual transaction channel")
    
    # Check anonymous scores
    anonymous_scores = {
        'transaction_payment_mode_anonymous': 0.15,
        'payment_gateway_bank_anonymous': 0.15,
        'payer_browser_anonymous': 0.1,
        'payer_email_anonymous': 0.1,
        'payee_ip_anonymous': 0.1,
        'payer_mobile_anonymous': 0.1,
        'transaction_id_anonymous': 0.1,
        'payee_id_anonymous': 0.1
    }
    
    total_anonymous_score = 0
    for field, weight in anonymous_scores.items():
        value = transaction.get(field, 0)
        logger.debug(f"Checking {field}: {value} (type: {type(value)})")
        try:
            # Convert to float and normalize to 0-1 range
            normalized_score = convert_anonymous_score(value) / 100.0
            total_anonymous_score += normalized_score
            if normalized_score < 0.5:  # If score is less than 50
                score += weight
                reasons.append(f"Low {field.replace('_', ' ').title()} score")
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing {field}: {str(e)}")
            score += weight
            reasons.append(f"Invalid {field.replace('_', ' ').title()} score")
    
    # Add risk based on average anonymous score
    avg_anonymous_score = total_anonymous_score / len(anonymous_scores)
    if avg_anonymous_score < 0.5:  # If average score is less than 50
        score += 0.2
        reasons.append("Low average anonymous score")
    
    # Determine if fraudulent based on score threshold
    is_fraud = score >= 0.7
    reason = " & ".join(reasons) if reasons else "No suspicious patterns"
    logger.debug(f"Final analysis - is_fraud: {is_fraud}, score: {score}, reason: {reason}")
    
    return is_fraud, reason, min(score, 1.0)

@app.route('/api/analyze-transaction', methods=['POST'])
def analyze_transaction():
    try:
        data = request.get_json()
        
        # Extract transaction details
        amount = float(data.get('transaction_amount', 0))
        channel = data.get('transaction_channel', '')
        if channel == 'w':
            channel = 'web'
        
        # Extract anonymous scores with conversion
        payment_mode_score = convert_anonymous_score(data.get('transaction_payment_mode_anonymous', 0))
        gateway_bank_score = convert_anonymous_score(data.get('payment_gateway_bank_anonymous', 0))
        browser_score = convert_anonymous_score(data.get('payer_browser_anonymous', 0))
        email_score = convert_anonymous_score(data.get('payer_email_anonymous', 0))
        ip_score = convert_anonymous_score(data.get('payee_ip_anonymous', 0))
        mobile_score = convert_anonymous_score(data.get('payer_mobile_anonymous', 0))
        transaction_id_score = convert_anonymous_score(data.get('transaction_id_anonymous', 0))
        payee_id_score = convert_anonymous_score(data.get('payee_id_anonymous', 0))
        
        # Calculate risk score based on amount and channel
        risk_score = 0
        if amount > 10000:
            risk_score += 30
        elif amount > 5000:
            risk_score += 20
        elif amount > 1000:
            risk_score += 10
            
        if channel.lower() == 'mobile':
            risk_score += 20
            
        # Calculate average anonymous score
        anonymous_scores = [
            payment_mode_score,
            gateway_bank_score,
            browser_score,
            email_score,
            ip_score,
            mobile_score,
            transaction_id_score,
            payee_id_score
        ]
        avg_anonymous_score = sum(anonymous_scores) / len(anonymous_scores)
        
        # Adjust risk score based on anonymous score
        if avg_anonymous_score < 50:
            risk_score += 30
        elif avg_anonymous_score < 70:
            risk_score += 20
        elif avg_anonymous_score < 90:
            risk_score += 10
            
        # Cap risk score at 100
        risk_score = min(risk_score, 100)
        
        return jsonify({
            "transaction_id": data.get('transaction_id_anonymous', ''),
            "transaction_amount": amount,
            "transaction_channel": channel,
            "is_fraud": risk_score >= 70,
            "risk_score": risk_score,
            "status": "Suspicious" if risk_score >= 70 else "Safe",
            "anonymous_scores": {
                'payment_mode': payment_mode_score,
                'gateway_bank': gateway_bank_score,
                'browser': browser_score,
                'email': email_score,
                'ip': ip_score,
                'mobile': mobile_score,
                'transaction_id': transaction_id_score,
                'payee_id': payee_id_score
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing transaction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/upload-transactions', methods=['POST'])
def upload_transactions():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400
        
    try:
        # Read CSV file
        df = pd.read_csv(file)
        logger.debug(f"Reading CSV file: {file.filename}")
        logger.debug(f"CSV columns: {df.columns.tolist()}")
        logger.debug(f"CSV data types:\n{df.dtypes}")
        logger.debug(f"First few rows:\n{df.head()}")
        
        results = []
        
        for index, row in df.iterrows():
            logger.debug(f"\nProcessing transaction {index + 1}")
            # Convert row to dictionary
            transaction = row.to_dict()
            logger.debug(f"Transaction data: {transaction}")
            
            # Analyze the transaction
            is_fraud, fraud_reason, fraud_score = analyze_fraud(transaction)
            logger.debug(f"Analysis result - is_fraud: {is_fraud}, fraud_score: {fraud_score}")
            
            # Format datetime if present
            transaction_date = transaction.get('transaction_date', '')
            if transaction_date:
                try:
                    dt = pd.to_datetime(transaction_date)
                    transaction_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                    logger.debug(f"Formatted date: {transaction_date}")
                except Exception as e:
                    logger.error(f"Error formatting date: {str(e)}")
                    pass
            
            # Calculate risk score (0-100)
            risk_score = round(fraud_score * 100, 2)
            
            result = {
                "transaction_id": transaction.get('transaction_id_anonymous', ''),
                "transaction_amount": float(transaction.get('transaction_amount', 0)),
                "transaction_date": transaction_date,
                "transaction_channel": transaction.get('transaction_channel', ''),
                "is_fraud": is_fraud,
                "risk_score": risk_score,
                "status": "Suspicious" if is_fraud or risk_score > 70 else "Safe",
                "fraud_reason": fraud_reason,
                "anonymous_scores": {
                    'payment_mode': convert_anonymous_score(transaction.get('transaction_payment_mode_anonymous', 0)),
                    'gateway_bank': convert_anonymous_score(transaction.get('payment_gateway_bank_anonymous', 0)),
                    'browser': convert_anonymous_score(transaction.get('payer_browser_anonymous', 0)),
                    'email': convert_anonymous_score(transaction.get('payer_email_anonymous', 0)),
                    'ip': convert_anonymous_score(transaction.get('payee_ip_anonymous', 0)),
                    'mobile': convert_anonymous_score(transaction.get('payer_mobile_anonymous', 0)),
                    'transaction_id': convert_anonymous_score(transaction.get('transaction_id_anonymous', 0)),
                    'payee_id': convert_anonymous_score(transaction.get('payee_id_anonymous', 0))
                }
            }
            logger.debug(f"Formatted result: {result}")
            results.append(result)
        
        logger.info(f"Successfully processed {len(results)} transactions")
        return jsonify({
            "message": f"Successfully processed {len(results)} transactions",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/api/contact', methods=['POST'])
def contact():
    data = request.get_json()
    print("Received contact form data:", data)
    
    return jsonify({
        "message": "Contact form submitted successfully",
        "data": data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 