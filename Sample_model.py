import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load CSV files
normal = pd.read_csv("C:\\Users\\Rohan\\OneDrive\\Documents\\normal.csv")
attack = pd.read_csv("C:\\Users\\Rohan\\OneDrive\\Documents\\attack.csv")

# Add labels
normal["label"] = 0   # 0 = Normal
attack["label"] = 1   # 1 = Attack

# Combine datasets
data = pd.concat([normal, attack], ignore_index=True)

# Select useful features (check your column names)
features = ["Length", "Time"]

X = data[features]
y = data["label"]

# Split data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create model
model = RandomForestClassifier()
model = RandomForestClassifier(class_weight="balanced", random_state=42)
# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
