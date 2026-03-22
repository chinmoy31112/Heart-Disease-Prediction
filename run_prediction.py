import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
print("Loading data from heart.csv...")
df = pd.read_csv('heart.csv')

# Drop Target from features and define target
x = df.drop(columns='target')
y = df['target']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Create and train model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate
x_train_pred = model.predict(x_train)
train_acc = accuracy_score(x_train_pred, y_train)
x_test_pred = model.predict(x_test)
test_acc = accuracy_score(x_test_pred, y_test)

# Prediction System Examples
# Example 1: Expected Heart Disease
input_data_1 = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_array_1 = np.asarray(input_data_1).reshape(1, -1)
prediction_1 = model.predict(input_array_1)

# Example 2: Expected No Heart Disease
input_data_2 = (69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0)
input_array_2 = np.asarray(input_data_2).reshape(1, -1)
prediction_2 = model.predict(input_array_2)

# Write results to output file
with open('results.txt', 'w') as f:
    f.write("Heart Disease Prediction Model Results\n")
    f.write("=" * 40 + "\n")
    f.write(f"Model Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Model Testing Accuracy: {test_acc:.4f}\n")
    f.write("-" * 30 + "\n")
    f.write("Sample Patient Predictions:\n")
    f.write(f"\nPatient 1 Data: {input_data_1}\n")
    f.write(f"Prediction: {'Heart Disease Detected' if prediction_1[0] == 1 else 'No Heart Disease'}\n")
    f.write(f"\nPatient 2 Data: {input_data_2}\n")
    f.write(f"Prediction: {'Heart Disease Detected' if prediction_2[0] == 1 else 'No Heart Disease'}\n")

print("Results written to results.txt")
