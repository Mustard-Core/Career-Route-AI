import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"cert_job_dataset.csv")
df = df.fillna("None")

# Separate features and target
X = df[['networking', 'cloud', 'hybrid']]
y = df['Best_fit_Job']

# One-hot encode networking, cloud, hybrid
onehot = OneHotEncoder(handle_unknown='ignore')
X = onehot.fit_transform(X).toarray()

# Label encode jobs (target)
job_encoder = LabelEncoder()
y = job_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=42
)

# Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=6, random_state=42)  # limit depth to reduce overfitting
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Results
print("Predicted Jobs:", job_encoder.inverse_transform(preds))
print("Accuracy:", accuracy_score(y_test, preds) * 100, "%")
