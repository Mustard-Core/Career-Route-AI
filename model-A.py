import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from settings import *
# Load dataset
df = pd.read_csv(r"programming_language_dataset.csv")
df = df.fillna("None")

sdf = df.head(1000)
#df = df.drop(columns=['Gender'])  # Gender not needed

# Features (languages) and Target (job)
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


# Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Results
print("Predicted Jobs:", label_enc.inverse_transform(preds))
print("Accuracy:", accuracy_score(y_test, preds) * 100, "%")

    
