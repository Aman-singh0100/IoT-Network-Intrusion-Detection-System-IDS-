import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
attack_df = pd.read_csv(r"C:\Users\arnav\Downloads\attack.csv")
normal_df = pd.read_csv(r"C:\Users\arnav\Downloads\normal.csv")
attack_df['label'] = 1
normal_df['label'] = 0
df = pd.concat([attack_df, normal_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.drop(['No.','Source','Destination'], axis=1)
df['Protocol'] = df['Protocol'].astype('category').cat.codes
df['is_syn'] = df['Info'].str.contains('SYN', case=False, na=False).astype(int)
df['is_ack'] = df['Info'].str.contains('ACK', case=False, na=False).astype(int)
df['is_rst'] = df['Info'].str.contains('RST', case=False, na=False).astype(int)
df['Time_diff'] = df['Time'].diff().fillna(0)
df = df.drop(['Info'], axis=1)
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

