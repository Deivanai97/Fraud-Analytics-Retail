import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)

def load_and_process_data(filepath='retail_fraud_detection_100k.csv'):
    """Backend Data Pipeline: Loads and engineers features."""
    raw_df = pd.read_csv(filepath)
    
    df = raw_df.copy()
    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    df['transaction_hour'] = df['transaction_timestamp'].dt.hour
    df['transaction_day_of_week'] = df['transaction_timestamp'].dt.dayofweek
    df['amount_to_avg_7d_ratio'] = df['transaction_amount'] / (df['avg_transaction_amount_7d'] + 0.001)
    
    cat_cols = ['payment_method', 'device_type', 'location', 'merchant_category']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            
    return raw_df, df

def train_and_evaluate(df):
    """Backend ML Pipeline: Trains model, calculates metrics, and builds explainers."""
    drop_cols = ['transaction_id', 'customer_id', 'transaction_timestamp', 'fraud_flag', 'fraud_risk','previous_fraud_flag',
        'unusual_location_flag',
        'unusual_amount_flag']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['fraud_flag']
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Generate Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate all requested metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'cm': confusion_matrix(y_test, y_pred),
        'pr_curve': precision_recall_curve(y_test, y_prob)
    }
    
    # Build SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, X_test, y_test, idx_test, metrics, features