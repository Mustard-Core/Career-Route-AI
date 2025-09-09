import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dataset
df = pd.read_csv("programming_language_dataset.csv")
df = df.fillna("None")

# Features and Target
X = df.drop(columns=['Best_fit_Job'])
y = df['Best_fit_Job']

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Encode target labels
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42
)

# Decision Tree Classifier setup
model = DecisionTreeClassifier(random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15]
}

# Fine-tune the model using GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predictions on the test set with the fine-tuned model
preds_tuned = best_model.predict(X_test)

# Comprehensive validation of the fine-tuned model
print("Best Parameters Found: ", grid_search.best_params_)
print("\nFine-Tuned Model Results:")
print(f"Accuracy: {accuracy_score(y_test, preds_tuned) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, preds_tuned, target_names=label_enc.classes_))
