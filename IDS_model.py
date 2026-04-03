import pandas as pd
from sklearn.utils import resample
import pickle

normal = pd.read_csv("C:\\Users\\Rohan\\OneDrive\\Documents\\normal.csv")
attack = pd.read_csv("C:\\Users\\Rohan\\OneDrive\\Documents\\attack.csv")

# Clean column names
normal.columns = normal.columns.str.strip()
attack.columns = attack.columns.str.strip()

def create_features(df):
    df["is_mqtt"] = (df["Protocol"] == "MQTT").astype(int)
    df["is_attack_proto"] = df["Protocol"].isin(["ICMP", "ICMPv6"]).astype(int)
    df["time_delta"] = df["Time"].diff().fillna(0)

    df["length_bucket"] = pd.cut(
        df["Length"],
        bins=[0, 100, 500, 1000, 2000],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df

normal = create_features(normal)
attack = create_features(attack)

normal["label"] = 0   # Normal
attack["label"] = 1   # Attack

data = pd.concat([normal, attack], ignore_index=True)

features = ["Length", "time_delta", "is_mqtt", "is_attack_proto", "length_bucket"]

X = data[features]
y = data["label"]

print(X.head())
print(y.value_counts())

data["label"] = data["label"]  # already exists

majority = data[data["label"] == 0]
minority = data[data["label"] == 1]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

data_balanced = pd.concat([majority, minority_upsampled])
data_balanced = data_balanced.sample(frac=1, random_state=42)

print(data_balanced["label"].value_counts())

from sklearn.model_selection import train_test_split

features = ["Length", "time_delta", "is_mqtt", "is_attack_proto", "length_bucket"]

X = data_balanced[features]
y = data_balanced["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(len(X_train), len(X_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


model_bundle = {
    "model": model,
    "features": ["Length", "time_delta", "is_mqtt", "is_attack_proto", "length_bucket"]
}

with open("iot_ids_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Model saved successfully!")
