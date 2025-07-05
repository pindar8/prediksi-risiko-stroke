import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.drop(columns=["id"], inplace=True)
df["bmi"].fillna(df["bmi"].median(), inplace=True)

# Encode kolom kategorikal
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split data
X = df.drop("stroke", axis=1)
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Simpan model dan encoder
joblib.dump(model, "model_stroke.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model dan encoders berhasil disimpan.")
