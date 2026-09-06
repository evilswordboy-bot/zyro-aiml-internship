import sys
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("=" * 60)
print("AI/ML ENVIRONMENT TEST - WEEK 01")
print("ENVIRONMENT & MACHINE LEARNING TEST")
print("=" * 60)

# Python
print("\n[1] Python Version")
print(sys.version)

# Libraries
print("\n[2] AI/ML Libraries")
print("NumPy       :", np.__version__)
print("Pandas      :", pd.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("Seaborn     :", sns.__version__)
print("Scikit-learn:", sklearn.__version__)

# Load dataset
print("\n[3] Loading Iris Dataset")

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

# Pandas
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["target"] = y

print("\n[4] Dataset Preview")
print(df.head())

# Split
print("\n[5] Train/Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# Model
print("\n[6] Training Machine Learning Model")

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("Model training completed successfully.")

# Prediction
print("\n[7] Making Predictions")

predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n[8] Model Performance")
print(f"Accuracy: {accuracy:.2%}")

print("\n" + "=" * 60)

if accuracy >= 0.90:
    print("SUCCESS: AI/ML ENVIRONMENT TEST PASSED")
else:
    print("ML PROGRAM RAN SUCCESSFULLY")

print("=" * 60)
