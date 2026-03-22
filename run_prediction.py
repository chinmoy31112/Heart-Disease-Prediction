import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the data
print("Loading data from heart.csv...")
df = pd.read_csv('heart.csv')

# Drop Target from features and define target
x = df.drop(columns='target')
y = df['target']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Feature Scaling (Standardizing the data)
print("Scaling features...")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and train model
print("Training Logistic Regression model...")
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate
x_train_pred = model.predict(x_train)
train_acc = accuracy_score(x_train_pred, y_train)
x_test_pred = model.predict(x_test)
test_acc = accuracy_score(x_test_pred, y_test)

# Prediction System Examples
# Example 1: Expected Heart Disease
input_data_1 = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
# Scale the input data before predicting
input_array_1 = np.asarray(input_data_1).reshape(1, -1)
input_array_1_scaled = scaler.transform(input_array_1)
prediction_1 = model.predict(input_array_1_scaled)

# Example 2: Expected No Heart Disease
input_data_2 = (69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0)
input_array_2 = np.asarray(input_data_2).reshape(1, -1)
input_array_2_scaled = scaler.transform(input_array_2)
prediction_2 = model.predict(input_array_2_scaled)

# Write results to output file
with open('results.txt', 'w') as f:
    f.write("Heart Disease Prediction Model Results (with Scaling)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Model Testing Accuracy: {test_acc:.4f}\n")
    f.write("-" * 30 + "\n")
    f.write("Sample Patient Predictions:\n")
    f.write(f"\nPatient 1 Data: {input_data_1}\n")
    f.write(f"Prediction: {'Heart Disease Detected' if prediction_1[0] == 1 else 'No Heart Disease'}\n")
    f.write(f"\nPatient 2 Data: {input_data_2}\n")
    f.write(f"Prediction: {'Heart Disease Detected' if prediction_2[0] == 1 else 'No Heart Disease'}\n")

print("Improved results written to results.txt")
