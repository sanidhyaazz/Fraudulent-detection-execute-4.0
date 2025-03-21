#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraud Detection API

This script implements a Flask API for real-time and batch fraud detection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import time
import concurrent.futures
from datetime import datetime
import sqlite3
import json
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the model and scaler
MODEL_PATH = 'models/optimized_logistic_regression_model.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'
HIST_GB_MODEL_PATH = 'models/hist_gradient_boosting_model.pkl'

# Database configuration
DB_PATH = 'data/fraud_detection.db'

# Rule engine configuration (defaults)
RULES = {
    'amount_threshold': 5000.0,  # Flag transactions above this amount
    'high_risk_channels': ['mobile', 'm'],  # High risk channels
    'high_risk_payment_modes': [3, 9],  # High risk payment modes
    'suspicious_hour_ranges': [(0, 4), (22, 24)],  # Suspicious hours (night time)
}

def initialize_db():
    """Initialize the SQLite database with required tables."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create fraud_detection table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fraud_detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            transaction_date TEXT,
            transaction_amount REAL,
            transaction_channel TEXT,
            is_fraud_predicted INTEGER,
            fraud_source TEXT,
            fraud_reason TEXT,
            fraud_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create fraud_reporting table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fraud_reporting (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            reporting_entity_id TEXT,
            fraud_details TEXT,
            is_fraud_reported INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (transaction_id) REFERENCES fraud_detection(transaction_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

def load_models():
    """Load machine learning models."""
    global model, scaler, hist_gb_model
    
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # Try to load the standard model and scaler
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info("Standard model and scaler loaded successfully")
        else:
            model = None
            scaler = None
            logger.warning("Standard model or scaler not found")
        
        # Try to load the HistGradientBoosting model as fallback
        if os.path.exists(HIST_GB_MODEL_PATH):
            hist_gb_model = joblib.load(HIST_GB_MODEL_PATH)
            logger.info("HistGradientBoosting model loaded successfully")
        else:
            hist_gb_model = None
            logger.warning("HistGradientBoosting model not found")
            
        if model is None and hist_gb_model is None:
            logger.warning("No models available. Only rule-based detection will be used.")
            
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def prepare_features(transaction_data):
    """Extract and prepare features for model prediction."""
    try:
        # Extract transaction date or use current time if not available
        if 'transaction_date' in transaction_data and transaction_data['transaction_date']:
            try:
                transaction_datetime = datetime.fromisoformat(str(transaction_data['transaction_date']).replace('Z', '+00:00'))
            except:
                transaction_datetime = datetime.strptime(str(transaction_data['transaction_date']), '%Y-%m-%d %H:%M:%S')
        else:
            transaction_datetime = datetime.now()
        
        # Extract hour, day of week, day of month
        hour = transaction_datetime.hour
        day_of_week = transaction_datetime.weekday()
        day_of_month = transaction_datetime.day
        
        # Map transaction channel to numeric
        channel = str(transaction_data.get('transaction_channel', '')).lower()
        channel_mapping = {'w': 0, 'web': 0, 'm': 1, 'mobile': 1, 'npm': 2, '': 3}
        channel_encoded = channel_mapping.get(channel, 3)  # Default to 3 for unknown channels
        
        # Extract other features with defaults
        features = [
            float(transaction_data.get('transaction_amount', 0.0)),
            hour,
            day_of_week,
            day_of_month,
            channel_encoded,
            int(transaction_data.get('transaction_payment_mode_anonymous', 0)),
            int(transaction_data.get('payment_gateway_bank_anonymous', 0)),
            int(transaction_data.get('payer_browser_anonymous', 0))
        ]
        
        return np.array(features).reshape(1, -1), transaction_datetime, channel
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None, datetime.now(), ""

def check_rule_engine(transaction_data, features, transaction_datetime, channel):
    """Check transaction against fraud detection rules."""
    try:
        # Get transaction amount
        amount = float(transaction_data.get('transaction_amount', 0))
        
        # Get payment mode
        payment_mode = int(transaction_data.get('transaction_payment_mode_anonymous', 0))
        
        # Check amount threshold
        if amount > RULES['amount_threshold']:
            return True, "High transaction amount", amount / RULES['amount_threshold']
        
        # Check high risk channels
        if channel.lower() in RULES['high_risk_channels']:
            return True, "High risk channel", 0.8
        
        # Check high risk payment modes
        if payment_mode in RULES['high_risk_payment_modes']:
            return True, "High risk payment mode", 0.9
        
        # Check suspicious hours
        hour = transaction_datetime.hour
        for start_hour, end_hour in RULES['suspicious_hour_ranges']:
            if start_hour <= hour < end_hour:
                return True, f"Suspicious transaction time: {hour}:00", 0.7
        
        return False, "", 0.0
    except Exception as e:
        logger.error(f"Error in rule engine: {str(e)}")
        return False, "", 0.0

def predict_fraud(features):
    """Use the ML model to predict fraud."""
    try:
        # If standard model is available
        if model is not None and scaler is not None:
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Check for NaN values
            if np.isnan(features_scaled).any():
                features_scaled = np.nan_to_num(features_scaled)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            
            # Get probability if available
            try:
                probability = model.predict_proba(features_scaled)[0][1]
            except:
                probability = float(prediction)
            
            return bool(prediction), "ML model detection", probability
        
        # Fallback to HistGradientBoosting model
        elif hist_gb_model is not None:
            # HistGradientBoosting can handle raw features
            prediction = hist_gb_model.predict(features)[0]
            
            # Get probability if available
            try:
                probability = hist_gb_model.predict_proba(features)[0][1]
            except:
                probability = float(prediction)
            
            return bool(prediction), "ML model detection (fallback)", probability
        
        else:
            logger.warning("No models available for prediction, using rules only")
            return False, "No ML model available", 0.0
            
    except Exception as e:
        logger.error(f"Error in ML prediction: {str(e)}")
        return False, "ML prediction error", 0.0

def store_detection_result(transaction_data, result):
    """Store the fraud detection result in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if transaction_id already exists
        cursor.execute("SELECT transaction_id FROM fraud_detection WHERE transaction_id = ?", 
                      (transaction_data.get('transaction_id'),))
        
        if cursor.fetchone() is None:
            # Insert new record
            cursor.execute('''
            INSERT INTO fraud_detection 
            (transaction_id, transaction_date, transaction_amount, transaction_channel, 
             is_fraud_predicted, fraud_source, fraud_reason, fraud_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction_data.get('transaction_id', ''),
                transaction_data.get('transaction_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                float(transaction_data.get('transaction_amount', 0.0)),
                transaction_data.get('transaction_channel', ''),
                1 if result['is_fraud'] else 0,
                result['fraud_source'],
                result['fraud_reason'],
                result['fraud_score']
            ))
        else:
            # Update existing record
            cursor.execute('''
            UPDATE fraud_detection 
            SET transaction_date = ?, transaction_amount = ?, transaction_channel = ?, 
                is_fraud_predicted = ?, fraud_source = ?, fraud_reason = ?, fraud_score = ?
            WHERE transaction_id = ?
            ''', (
                transaction_data.get('transaction_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                float(transaction_data.get('transaction_amount', 0.0)),
                transaction_data.get('transaction_channel', ''),
                1 if result['is_fraud'] else 0,
                result['fraud_source'],
                result['fraud_reason'],
                result['fraud_score'],
                transaction_data.get('transaction_id', '')
            ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error storing detection result: {str(e)}")
        return False

def process_transaction(transaction_data):
    """Process a single transaction for fraud detection."""
    start_time = time.time()
    
    # Validate transaction data
    if not transaction_data or not isinstance(transaction_data, dict):
        return {
            "transaction_id": "unknown",
            "is_fraud": False,
            "fraud_source": "error",
            "fraud_reason": "Invalid transaction data",
            "fraud_score": 0.0
        }
    
    # Get transaction ID
    transaction_id = transaction_data.get('transaction_id', 'unknown')
    
    try:
        # Prepare features for model
        features, transaction_datetime, channel = prepare_features(transaction_data)
        
        if features is None:
            return {
                "transaction_id": transaction_id,
                "is_fraud": False,
                "fraud_source": "error",
                "fraud_reason": "Failed to process transaction features",
                "fraud_score": 0.0
            }
        
        # Check rules first
        is_fraud_rule, rule_reason, rule_score = check_rule_engine(
            transaction_data, features, transaction_datetime, channel)
        
        # If rule detected fraud, return early
        if is_fraud_rule:
            result = {
                "transaction_id": transaction_id,
                "is_fraud": True,
                "fraud_source": "rule",
                "fraud_reason": rule_reason,
                "fraud_score": rule_score
            }
            # Store result
            store_detection_result(transaction_data, result)
            return result
        
        # Otherwise, use ML model
        is_fraud_ml, ml_reason, ml_score = predict_fraud(features)
        
        result = {
            "transaction_id": transaction_id,
            "is_fraud": is_fraud_ml,
            "fraud_source": "model",
            "fraud_reason": ml_reason,
            "fraud_score": ml_score
        }
        
        # Store result
        store_detection_result(transaction_data, result)
        
        # Calculate and log latency
        latency = (time.time() - start_time) * 1000  # in ms
        logger.info(f"Transaction {transaction_id} processed in {latency:.2f} ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing transaction {transaction_id}: {str(e)}")
        return {
            "transaction_id": transaction_id,
            "is_fraud": False,
            "fraud_source": "error",
            "fraud_reason": f"Processing error: {str(e)}",
            "fraud_score": 0.0
        }

@app.route('/api/fraud-detection', methods=['POST'])
def fraud_detection_api():
    """Endpoint for real-time fraud detection of a single transaction."""
    try:
        # Get transaction data from request
        transaction_data = request.json
        
        if not transaction_data:
            return jsonify({
                "error": "No transaction data provided",
                "status": "failed"
            }), 400
        
        # Process the transaction
        result = process_transaction(transaction_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/api/fraud-detection/batch', methods=['POST'])
def batch_fraud_detection_api():
    """Endpoint for batch fraud detection of multiple transactions."""
    try:
        # Get batch of transactions
        batch_data = request.json
        
        if not batch_data or not isinstance(batch_data, list):
            return jsonify({
                "error": "Invalid batch data format. Expected a list of transactions.",
                "status": "failed"
            }), 400
        
        # Process transactions in parallel
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all transactions for processing
            future_to_transaction = {
                executor.submit(process_transaction, transaction): transaction 
                for transaction in batch_data
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_transaction):
                transaction = future_to_transaction[future]
                try:
                    result = future.result()
                    transaction_id = result.get('transaction_id')
                    if transaction_id:
                        results[transaction_id] = {
                            "is_fraud": result.get('is_fraud', False),
                            "fraud_reason": result.get('fraud_reason', ''),
                            "fraud_score": result.get('fraud_score', 0.0)
                        }
                except Exception as e:
                    transaction_id = transaction.get('transaction_id', 'unknown')
                    results[transaction_id] = {
                        "is_fraud": False,
                        "fraud_reason": f"Processing error: {str(e)}",
                        "fraud_score": 0.0
                    }
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Batch API error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/')
def index():
    """Root endpoint to check if the API is up."""
    return jsonify({
        "status": "online",
        "message": "Fraud Detection API is running",
        "version": "1.0.0",
        "endpoints": {
            "/api/fraud-detection": "Real-time fraud detection for a single transaction",
            "/api/fraud-detection/batch": "Batch fraud detection for multiple transactions"
        }
    }), 200

@app.route('/api/rules', methods=['GET'])
def get_rules():
    """Endpoint to get current rule engine configuration."""
    return jsonify(RULES), 200

@app.route('/api/rules', methods=['POST'])
def update_rules():
    """Endpoint to update rule engine configuration."""
    global RULES
    try:
        new_rules = request.json
        
        if not new_rules or not isinstance(new_rules, dict):
            return jsonify({
                "error": "Invalid rules format",
                "status": "failed"
            }), 400
        
        # Update only the provided rules
        for key, value in new_rules.items():
            if key in RULES:
                RULES[key] = value
        
        return jsonify({
            "status": "success",
            "message": "Rules updated successfully",
            "rules": RULES
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating rules: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

if __name__ == '__main__':
    # Initialize the database
    if not initialize_db():
        logger.error("Failed to initialize database. Exiting.")
        exit(1)
    
    # Load the models
    if not load_models():
        logger.warning("Failed to load models. Using rule engine only.")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 