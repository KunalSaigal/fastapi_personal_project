from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Recreating our clean, imputed dataset from the previous step
data_clean = {
    "Age": [25.0, 47.0, 39.5, 32.0, 55.0],
    "Blood_Pressure": [80.0, 90.6, 72.0, 120.0, 90.6],
    "Glucose": [110.0, 140.0, 95.0, 131.25, 180.0],
}
df = pd.DataFrame(data_clean)

# Let's add a Target column 'Has_Diabetes' (1 = Yes, 0 = No)
df["Has_Diabetes"] = [0, 1, 0, 0, 1]

print("--- Clean DataFrame Ready for ML ---")
print(df)


# STEP 1: Separate Features (X) and Target (y)
# X contains all the columns we use to predict (Age, BP, Glucose)
X = df.drop(columns=["Has_Diabetes"])

# y contains the single column we want the model to predict
y = df["Has_Diabetes"]


# STEP 2: Split into Training and Testing Sets
# We use 80% of data for training and reserve 20% for testing.
# random_state acts like a "seed" ensuring we get the exact same split every time we run it.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)

print(
    "\nTraining Features Shape:", X_train.shape
)  # Output: (4, 3) -> 4 patients, 3 features
print("Testing Features Shape:", X_test.shape)  # Output: (1, 3) -> 1 patient held back


# STEP 3: Initialize and Train the Model
# 1. Create the model object
model = LogisticRegression()

# 2. Train the model using the .fit() method
# Under the hood, Scikit-learn extracts the underlying NumPy arrays from our DataFrames
# and runs matrix calculus to determine the ideal weights for Age, BP, and Glucose.
model.fit(X_train, y_train)

print("\nModel training complete!")


# BONUS: Making a Prediction
# Let's pass the unseen test patient through the trained model
prediction = model.predict(X_train)
print(f"Prediction for the test patient: {prediction[0]} (Actual: {y_test.values[0]})")

print(f"Prediction: {prediction}")

# Let's simulate the actual true answers vs what our model predicted
# 1 = Has Diabetes, 0 = Healthy
y_pred = prediction  # Actual reality
y_true = y_train  # Model's guesses


# 1. Individual Metrics
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2f}")  # 0.80
print(f"Precision: {precision_score(y_true, y_pred):.2f}")  # 0.80
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")  # 0.80


# 2. The Confusion Matrix Matrix
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred))
# Output:
# [[4 1]   -> [True Negatives, False Positives]
#  [1 4]]  -> [False Negatives, True Positives]


# 3. The Holy Grail: Classification Report
# This gives you a complete summary of all metrics for both classes at once
print("\n--- Full Classification Report ---")
print(
    classification_report(y_true, y_pred, target_names=["Healthy (0)", "Diabetes (1)"])
)
