# Experiment No. 8
# Customer Churn Prediction using Decision Tree Classification

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Step 2: Load the dataset
# Keep Telco-Customer-Churn.csv in the same folder as this notebook
df = pd.read_csv("Telco-Customer-Churn.csv")

# Display first few records
print("First 5 records:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

# Step 3: Data preprocessing

# Remove customer ID because it does not contribute to prediction
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
# Invalid/blank values become NaN
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert target variable: No = 0, Yes = 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Encode categorical variables
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# Convert Boolean columns to integers
df = df.astype(int)

print("\nPreprocessed Dataset:")
print(df.head())

# Step 4: Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 6: Train Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Step 7: Predict customer churn
y_pred = model.predict(X_test)

# Step 8: Evaluate the model

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nEvaluation Metrics:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Visualize the Decision Tree
plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Customer Churn Prediction")
plt.show()

# Step 10: Find important features
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Factors Influencing Customer Churn:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Features Influencing Customer Churn")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
