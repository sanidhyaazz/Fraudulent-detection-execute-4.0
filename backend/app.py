from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def analyze_fraud(transaction):
    """
    Analyze a transaction for fraud based on rules.
    Returns a tuple of (is_fraud, reason, score)
    """
    score = 0.0
    reasons = []
    
    # Check transaction amount
    amount = float(transaction.get('transaction_amount', 0))
    if amount > 3000:
        score += 0.3
        reasons.append("High transaction amount")
    
    # Check transaction channel
    channel = str(transaction.get('transaction_channel', '')).lower()
    if channel not in ['mobile', 'w']:
        score += 0.1
        reasons.append("Unusual transaction channel")
    
    # Check anonymous flags
    anonymous_flags = {
        'transaction_payment_mode_anonymous': 0.15,
        'payment_gateway_bank_anonymous': 0.15,
        'payer_browser_anonymous': 0.1,
        'payer_email_anonymous': 0.1,
        'payee_ip_anonymous': 0.1,
        'payer_mobile_anonymous': 0.1,
        'transaction_id_anonymous': 0.1,
        'payee_id_anonymous': 0.1
    }
    
    for flag, weight in anonymous_flags.items():
        value = transaction.get(flag, 0)
        try:
            if float(value) > 0:
                score += weight
                reasons.append(f"{flag.replace('_', ' ').title()}")
        except (ValueError, TypeError):
            # If value is not numeric or is missing, consider it as suspicious
            score += weight
            reasons.append(f"{flag.replace('_', ' ').title()}")
    
    # Determine if fraudulent based on score threshold
    is_fraud = score >= 0.7
    reason = " & ".join(reasons) if reasons else "No suspicious patterns"
    
    return is_fraud, reason, min(score, 1.0)

@app.route('/api/analyze-transaction', methods=['POST'])
def analyze_transaction():
    data = request.get_json()
    print("Received transaction data:", data)
    
    # Analyze the transaction
    is_fraud, fraud_reason, fraud_score = analyze_fraud(data)
    
    response = {
        "transaction_id": data.get('transaction_id', ''),
        "is_fraud": is_fraud,
        "fraud_source": "rule",  # Since we're using rule-based detection
        "fraud_reason": fraud_reason,
        "fraud_score": round(fraud_score, 2)
    }
    
    return jsonify(response)

@app.route('/api/upload-transactions', methods=['POST'])
def upload_transactions():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV file"}), 400
    
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(file)
        
        # Process each transaction
        results = []
        for _, row in df.iterrows():
            # Convert row to dictionary
            transaction = row.to_dict()
            
            # Analyze the transaction
            is_fraud, fraud_reason, fraud_score = analyze_fraud(transaction)
            
            # Format datetime if present
            transaction_date = transaction.get('transaction_date', '')
            if transaction_date:
                try:
                    dt = pd.to_datetime(transaction_date)
                    transaction_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            results.append({
                "transaction_amount": float(transaction.get('transaction_amount', 0)),
                "transaction_date": transaction_date,
                "transaction_channel": transaction.get('transaction_channel', ''),
                "is_fraud": is_fraud,
                "fraud_source": "rule",
                "fraud_reason": fraud_reason,
                "fraud_score": round(fraud_score, 2),
                "original_fraud_label": bool(transaction.get('is_fraud', 0))
            })
        
        return jsonify({
            "message": f"Successfully processed {len(results)} transactions",
            "results": results
        })
        
    except Exception as e:
        print("Error processing file:", str(e))
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
    app.run(debug=True, port=5000) 