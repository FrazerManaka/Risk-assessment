import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Preprocessing
def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Handle missing values and encode categorical variables."""
    # For simplicity, we'll drop rows with missing values
    data = data.dropna()
    
    # Encode categorical variables
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan']
    return pd.get_dummies(data, columns=categorical_columns)

# 2. Feature Engineering
def calculate_financial_ratios(data):
    """Calculate relevant financial ratios."""
    data['debt_to_income'] = data['balance'] / (data['income'] + 1)  # Adding 1 to avoid division by zero
    data['age_to_income'] = data['age'] / (data['income'] + 1)
    return data

def create_risk_indicators(data):
    """Create binary risk indicators based on thresholds."""
    data['high_debt'] = (data['debt_to_income'] > 0.5).astype(int)
    data['low_balance'] = (data['balance'] < 0).astype(int)
    return data

# 3. Risk Scoring Model
def train_risk_model(X, y):
    """Train a Random Forest model for risk assessment."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_risk_score(model, X):
    """Use the trained model to predict risk scores."""
    return model.predict_proba(X)[:, 1]  # Probability of the positive class

# 4. Reporting and Visualization
def generate_risk_report(data, risk_scores):
    """Create a summary report of risk assessment."""
    data['risk_score'] = risk_scores
    high_risk = data[data['risk_score'] > 0.7]
    print("Risk Assessment Report")
    print("----------------------")
    print(f"Total customers assessed: {len(data)}")
    print(f"High-risk customers (score > 0.7): {len(high_risk)}")
    print("\nTop 10 highest-risk customers:")
    print(high_risk.sort_values('risk_score', ascending=False)[['age', 'job', 'income', 'balance', 'risk_score']].head(10))

def visualize_risk_distribution(risk_scores):
    """Create visualizations of risk score distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(risk_scores, kde=True)
    plt.title('Distribution of Risk Scores')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.savefig('risk_distribution.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # 1. Load and preprocess data
    raw_data = load_data("bank_data.csv")  # You'll need to provide this CSV file
    processed_data = preprocess_data(raw_data)

    # 2. Perform feature engineering
    processed_data = calculate_financial_ratios(processed_data)
    processed_data = create_risk_indicators(processed_data)

    # 3. Train and apply risk scoring model
    X = processed_data.drop(["y"], axis=1)  # Assuming 'y' is the target variable
    y = processed_data["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_risk_model(X_train, y_train)
    risk_scores = predict_risk_score(model, X_test)

    # Print model performance
    y_pred = (risk_scores > 0.5).astype(int)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))

    # 4. Generate reports and visualizations
    generate_risk_report(X_test, risk_scores)
    visualize_risk_distribution(risk_scores)

    print("\nRisk assessment completed. Check 'risk_distribution.png' for the risk score distribution visualization.")