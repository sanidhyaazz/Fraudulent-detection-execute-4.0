#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for fraud detection API

This script tests both the single transaction and batch APIs.
"""

import requests
import json
import time
import uuid
from datetime import datetime
import random

# API base URL - update this to match your deployment
BASE_URL = "http://localhost:5000"

def generate_transaction(fraud_risk='random'):
    """Generate a random transaction for testing."""
    # Create a unique transaction ID
    transaction_id = str(uuid.uuid4())
    
    # Current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Set amount based on fraud risk
    if fraud_risk == 'high':
        # High amount to trigger rule
        amount = random.uniform(5001, 10000)
    elif fraud_risk == 'low':
        # Low amount to avoid rule
        amount = random.uniform(10, 1000)
    else:
        # Random amount
        amount = random.uniform(10, 10000)
    
    # Transaction channel
    channels = ['web', 'mobile', 'npm']
    channel_weights = [0.6, 0.3, 0.1]  # Probability weights
    channel = random.choices(channels, weights=channel_weights, k=1)[0]
    
    # Generate transaction
    transaction = {
        "transaction_id": transaction_id,
        "transaction_date": timestamp,
        "transaction_amount": amount,
        "transaction_channel": channel,
        "transaction_payment_mode_anonymous": random.randint(0, 15),
        "payment_gateway_bank_anonymous": random.randint(0, 10),
        "payer_browser_anonymous": random.randint(1000, 5000),
        "payer_email_anonymous": f"user_{random.randint(1000, 9999)}@example.com",
        "payer_mobile_anonymous": f"XXXXX{random.randint(100, 999)}.0",
        "payee_id_anonymous": f"ANON_{random.randint(0, 100)}"
    }
    
    return transaction

def test_single_transaction_api():
    """Test the real-time fraud detection API with a single transaction."""
    print("\n=== Testing Real-time Fraud Detection API ===")
    
    # Test with a low-risk transaction
    low_risk_transaction = generate_transaction(fraud_risk='low')
    print(f"Sending low-risk transaction: {low_risk_transaction['transaction_id']}")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/fraud-detection",
        json=low_risk_transaction,
        headers={"Content-Type": "application/json"}
    )
    latency = (time.time() - start_time) * 1000  # ms
    
    print(f"Response status code: {response.status_code}")
    print(f"Latency: {latency:.2f} ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.text}")
    
    # Test with a high-risk transaction
    high_risk_transaction = generate_transaction(fraud_risk='high')
    print(f"\nSending high-risk transaction: {high_risk_transaction['transaction_id']}")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/fraud-detection",
        json=high_risk_transaction,
        headers={"Content-Type": "application/json"}
    )
    latency = (time.time() - start_time) * 1000  # ms
    
    print(f"Response status code: {response.status_code}")
    print(f"Latency: {latency:.2f} ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.text}")

def test_batch_transaction_api(batch_size=5):
    """Test the batch fraud detection API with multiple transactions."""
    print(f"\n=== Testing Batch Fraud Detection API (batch size: {batch_size}) ===")
    
    # Generate a batch of transactions
    batch_transactions = [generate_transaction() for _ in range(batch_size)]
    
    print(f"Sending batch of {batch_size} transactions...")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/fraud-detection/batch",
        json=batch_transactions,
        headers={"Content-Type": "application/json"}
    )
    latency = (time.time() - start_time) * 1000  # ms
    
    print(f"Response status code: {response.status_code}")
    print(f"Total batch latency: {latency:.2f} ms")
    print(f"Average latency per transaction: {latency / batch_size:.2f} ms")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Received {len(results)} results")
        print("Sample results:")
        
        # Print first 2 results as samples
        sample_count = min(2, len(results))
        sample_keys = list(results.keys())[:sample_count]
        
        for key in sample_keys:
            print(f"  Transaction {key}: {json.dumps(results[key], indent=2)}")
            
        # Count frauds detected
        fraud_count = sum(1 for result in results.values() if result['is_fraud'])
        print(f"\nFrauds detected: {fraud_count}/{batch_size} ({fraud_count/batch_size*100:.1f}%)")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    try:
        # Check if API is available
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print(f"Error: API not available at {BASE_URL}")
            exit(1)
            
        print(f"API is available at {BASE_URL}")
        
        # Test APIs
        test_single_transaction_api()
        test_batch_transaction_api(batch_size=10)
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {BASE_URL}")
        print("Make sure the API server is running and accessible.")
        exit(1) 