# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Original/Raw Dataset
raw_data_path = 'C:/Users/user/OneDrive/Desktop/teleconnect - raw dataset.csv'
df = pd.read_csv(raw_data_path)

# Data Pre-processing Steps: (i) Drop Customer ID
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# (ii) Clean 'TotalCharges' (convert to numeric and drop rows with issues)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# (iii) Encode Binary Columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# (iv) Encode 'gender' (Female = 0, Male = 1)
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# (v) One-Hot Encode Categorical Columns
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature Engineering (e.g., TotalSpend)
df['TotalSpend'] = df['MonthlyCharges'] * df['tenure']

# Scaling (Standardize numeric features)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalSpend']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save Cleaned Dataset
cleaned_path = 'C:/Users/user/PycharmProjects/BAN6800/cleaned_telco_customer_churn.csv'
df.to_csv(cleaned_path, index=False)

print("✅ Cleaned dataset saved successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the cleaned dataset
data_path = 'C:/Users/user/PycharmProjects/BAN6800/cleaned_telco_customer_churn.csv'
df = pd.read_csv(data_path)

# Step 2: Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Step 3: Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = log_model.predict(X_test_scaled)
y_proba = log_model.predict_proba(X_test_scaled)[:, 1]

# Step 7: Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Step 8: ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()












