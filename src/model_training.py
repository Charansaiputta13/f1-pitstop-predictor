import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_features(path='../data/processed/features.csv'):
    """Load engineered features and labels."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found! Run feature_engineering.py first.")
    df = pd.read_csv(path)
    X = df.drop(columns=['will_pit_next_lap'])
    y = df['will_pit_next_lap']
    return X, y


def train_model(X, y):
    """Train and evaluate a Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }

    print("✅ Model Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Pit Stop Prediction Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return model, scaler


def save_model(model, scaler, path='../model/'):
    """Save the model and scaler."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, 'pitstop_model.pkl'))
    joblib.dump(scaler, os.path.join(path, 'scaler.pkl'))
    print(f"✅ Model and scaler saved to {path}")


if __name__ == "__main__":
    X, y = load_features()
    model, scaler = train_model(X, y)
    save_model(model, scaler)
