#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraud Detection System with Low Latency

This script implements a simple, low-latency fraud detection system
that classifies transactions as fraudulent or legitimate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import time
import joblib
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path):
    """Load and return the transaction dataset."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data for model training."""
    print("Preprocessing data...")
    
    # Check for missing values before preprocessing
    missing_values = df.isnull().sum()
    print("\nMissing values in original data:")
    print(missing_values[missing_values > 0])
    
    # Convert transaction_date to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Extract useful time features
    df['hour'] = df['transaction_date'].dt.hour
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['day_of_month'] = df['transaction_date'].dt.day
    
    # Convert transaction_channel to numeric
    # Clean and standardize transaction channels first
    df['transaction_channel'] = df['transaction_channel'].str.lower().fillna('unknown')
    channel_mapping = {channel: idx for idx, channel in enumerate(df['transaction_channel'].unique())}
    df['transaction_channel_encoded'] = df['transaction_channel'].map(channel_mapping)
    
    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        # For numeric columns, fill missing values with median
        if df[col].isnull().sum() > 0:
            print(f"Filling missing values in {col} with median")
            df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values in other columns
    for col in df.columns:
        if col not in numeric_cols and df[col].isnull().sum() > 0:
            # For categorical columns, fill with the most frequent value
            print(f"Filling missing values in {col} with most frequent value")
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Check for any remaining missing values
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print("\nRemaining missing values after preprocessing:")
        print(missing_after[missing_after > 0])
    else:
        print("\nAll missing values have been handled.")
    
    return df

def feature_engineering(df):
    """Create new features that might be useful for fraud detection."""
    print("Performing feature engineering...")
    
    # First verify that all the features we need are present
    required_features = [
        'transaction_amount', 
        'hour', 
        'day_of_week', 
        'day_of_month',
        'transaction_channel_encoded',
        'transaction_payment_mode_anonymous',
        'payment_gateway_bank_anonymous',
        'payer_browser_anonymous'
    ]
    
    # Check if all required features exist in the dataframe
    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        print(f"Warning: Missing required features: {missing_features}")
        # If some features are missing, we'll only use what's available
        required_features = [f for f in required_features if f in df.columns]
    
    # Check if any selected feature has NaN values
    for feature in required_features:
        if feature in df.columns and df[feature].isnull().sum() > 0:
            print(f"Warning: Feature {feature} has {df[feature].isnull().sum()} NaN values")
            print(f"Filling NaN values in {feature} with median/mode")
            if df[feature].dtype in ['int64', 'float64']:
                df[feature] = df[feature].fillna(df[feature].median())
            else:
                df[feature] = df[feature].fillna(df[feature].mode()[0])
    
    # Simple target
    target = 'is_fraud'
    
    # Verify target exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the dataframe")
    
    # Double check that X has no NaN values
    X = df[required_features]
    if X.isnull().values.any():
        print("Warning: There are still NaN values in the feature matrix!")
        print("NaN counts by feature:")
        print(X.isnull().sum())
        
        # Fill any remaining NaN values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0])
    
    # Final check to ensure no NaN values remain
    assert not X.isnull().values.any(), "NaN values remain in feature matrix!"
    
    return X, df[target]

def perform_eda(df):
    """Perform exploratory data analysis on the dataset."""
    print("\nPerforming EDA...")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.4f}%")
    
    # Transaction amount statistics
    print("\nTransaction amount statistics:")
    print(df['transaction_amount'].describe())
    
    # Transaction channel distribution
    print("\nTransaction channel distribution:")
    print(df['transaction_channel'].value_counts())
    
    # Payment mode distribution
    print("\nPayment mode distribution:")
    print(df['transaction_payment_mode_anonymous'].value_counts())
    
    # Distribution of transactions by hour
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='hour', hue='is_fraud', multiple='stack', bins=24)
    plt.title('Distribution of Transactions by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Transactions')
    plt.tight_layout()
    plt.savefig('transactions_by_hour.png')
    
    # Distribution of transaction amounts
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='transaction_amount', hue='is_fraud', 
                 multiple='stack', bins=30, log_scale=True)
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Transaction Amount (log scale)')
    plt.ylabel('Number of Transactions')
    plt.tight_layout()
    plt.savefig('transaction_amounts.png')
    
    # Correlation heatmap of numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_features].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    
    print("EDA completed. Visualizations saved as PNG files.")

def benchmark_models(X_train, X_test, y_train, y_test):
    """Benchmark different models for performance and latency."""
    print("\nBenchmarking models for performance and latency...")
    
    # Define models to benchmark
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, 
                                                 solver='liblinear'),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, 
                                              class_weight='balanced', n_jobs=-1),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(max_depth=5, learning_rate=0.1)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Measure training time
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # Measure prediction time
        pred_start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - pred_start
        avg_pred_time_ms = (pred_time / len(X_test)) * 1000  # Average prediction time in ms
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Try to get ROC-AUC if the model supports predict_proba
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = float('nan')
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc,
            'Training Time (s)': train_time,
            'Avg Prediction Time (ms)': avg_pred_time_ms
        })
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Training Time: {train_time:.4f} seconds")
        print(f"  Avg Prediction Time: {avg_pred_time_ms:.4f} ms")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save the model
        joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")
    
    return pd.DataFrame(results)

def train_optimized_model(X_train, y_train, model_type='logistic_regression'):
    """Train the selected model with optimized parameters for low latency."""
    print(f"\nTraining optimized {model_type} model...")
    
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            class_weight='balanced',
            solver='liblinear',  # Fast solver for small datasets
            C=0.1,               # Regularization strength
            max_iter=100,        # Limit iterations for faster convergence
            tol=1e-3             # Tolerance for stopping criteria
        )
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=3,         # Shallow tree for faster inference
            min_samples_split=10,
            class_weight='balanced',
            splitter='best'      # Use 'random' for faster training
        )
    elif model_type == 'hist_gradient_boosting':
        model = HistGradientBoostingClassifier(
            max_depth=3,          # Shallow trees for faster inference
            learning_rate=0.1,    # Moderate learning rate
            max_iter=100,         # Limit iterations for faster convergence
            early_stopping=True,  # Stop early if validation score doesn't improve
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=1
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Model training completed in {training_time:.4f} seconds")
    
    # Save the optimized model
    model_filename = f"optimized_{model_type}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    print("\nEvaluating model performance...")
    
    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    avg_pred_time_ms = (prediction_time / len(X_test)) * 1000
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Prediction Time: {avg_pred_time_ms:.4f} ms per transaction")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Try to create ROC curve if model supports predict_proba
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png')
        print(f"ROC-AUC: {roc_auc:.4f}")
    except:
        print("Model doesn't support probability predictions, skipping ROC curve.")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_pred_time_ms': avg_pred_time_ms
    }

def main():
    """Main function to run the fraud detection pipeline."""
    print("Starting fraud detection pipeline...")
    
    try:
        # Load data
        df = load_data('transactions_train.csv')
        
        # Preprocess data
        df = preprocess_data(df)
        
        # Perform exploratory data analysis
        perform_eda(df)
        
        # Feature engineering
        X, y = feature_engineering(df)
        
        # Display basic info about the preprocessed data
        print("\nFeature matrix shape:", X.shape)
        print("Target vector shape:", y.shape)
        print("Fraud cases in dataset:", y.sum())
        print(f"Fraud ratio: {y.mean() * 100:.4f}%")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        print("\nTraining set shape:", X_train.shape)
        print("Testing set shape:", X_test.shape)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for future use
        joblib.dump(scaler, 'feature_scaler.pkl')
        print("Feature scaler saved to 'feature_scaler.pkl'")
        
        # Check for NaN values after scaling
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            print("\nWarning: NaN values detected after scaling!")
            print(f"NaN values in X_train_scaled: {np.isnan(X_train_scaled).sum()}")
            print(f"NaN values in X_test_scaled: {np.isnan(X_test_scaled).sum()}")
            
            # Replace any remaining NaN values with 0
            X_train_scaled = np.nan_to_num(X_train_scaled)
            X_test_scaled = np.nan_to_num(X_test_scaled)
            print("NaN values have been replaced with 0.")
        
        try:
            # Benchmark different models
            results_df = benchmark_models(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Display results as a sorted table
            print("\nModel Performance Summary (sorted by latency):")
            print(results_df.sort_values('Avg Prediction Time (ms)').to_string(index=False))
            
            # Based on the benchmarking results, select and train the optimized model
            # For low-latency, we typically choose logistic regression or a small decision tree
            model = train_optimized_model(X_train_scaled, y_train, model_type='logistic_regression')
        except ValueError as e:
            if "NaN" in str(e):
                print(f"\nError with standard models due to missing values: {str(e)}")
                print("Falling back to HistGradientBoosting which can handle missing values natively.")
                
                # Use HistGradientBoosting which can handle missing values natively
                model = train_optimized_model(X_train, y_train, model_type='hist_gradient_boosting')
                X_test_for_eval = X_test  # Use unscaled data for HistGradientBoosting
            else:
                raise e
        
        # Evaluate the optimized model
        metrics = evaluate_model(model, X_test_scaled if 'X_test_for_eval' not in locals() else X_test_for_eval, y_test)
        
        print("\nFraud Detection Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in fraud detection pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTrying to continue with HistGradientBoosting model which can handle missing values...")
        
        try:
            # Attempt to salvage by creating a model that can handle missing values
            if 'X_train' in locals() and 'y_train' in locals():
                print("Training a HistGradientBoosting model with available data...")
                model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=100)
                model.fit(X_train, y_train)
                joblib.dump(model, "optimized_hist_gradient_boosting_model.pkl")
                print("HistGradientBoosting model saved successfully!")
                
                if 'X_test' in locals() and 'y_test' in locals():
                    print("Evaluating the fallback model...")
                    metrics = evaluate_model(model, X_test, y_test)
            else:
                print("Cannot salvage - insufficient data processed.")
        except Exception as e2:
            print(f"Failed to create fallback model: {str(e2)}")
            return False
    
    return True

def predict_transaction(transaction_data, model_path='optimized_logistic_regression_model.pkl', 
                       scaler_path='feature_scaler.pkl'):
    """
    Make a real-time prediction for a single transaction.
    
    Args:
        transaction_data: Dictionary containing transaction features
        model_path: Path to the saved model
        scaler_path: Path to the saved feature scaler
        
    Returns:
        A tuple of (prediction, probability) where prediction is 0 or 1
    """
    try:
        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Extract features from transaction data with defaults for missing values
        # Convert transaction date to features
        try:
            transaction_datetime = datetime.strptime(transaction_data.get('transaction_date', ''), 
                                                 '%Y-%m-%d %H:%M:%S')
            hour = transaction_datetime.hour
            weekday = transaction_datetime.weekday()
            day = transaction_datetime.day
        except ValueError:
            # If date parsing fails, use default values
            print("Warning: Invalid transaction date format. Using default values.")
            hour = 12  # Noon
            weekday = 0  # Monday
            day = 15  # Middle of month
        
        # Prepare feature vector with defaults for missing values
        features = [
            transaction_data.get('transaction_amount', 0.0),
            hour,
            weekday,
            day,
            transaction_data.get('transaction_channel_encoded', 0),
            transaction_data.get('transaction_payment_mode_anonymous', 0),
            transaction_data.get('payment_gateway_bank_anonymous', 0),
            transaction_data.get('payer_browser_anonymous', 0)
        ]
        
        # Convert to numpy array and reshape for single prediction
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Check for NaN values after scaling
        if np.isnan(features_scaled).any():
            print("Warning: NaN values detected in scaled features. Replacing with 0.")
            features_scaled = np.nan_to_num(features_scaled)
        
        # Make prediction
        start_time = time.time()
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(features_scaled)[0][1]
        except:
            probability = None
        
        prediction_time_ms = (time.time() - start_time) * 1000
        
        print(f"Prediction made in {prediction_time_ms:.2f} ms")
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        # Return a safe default - assume not fraud
        return 0, 0.0

if __name__ == "__main__":
    main() 