# hr_attrition_ui.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)

# Encode categorical
cat_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(probability=True, kernel='rbf'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Voting Ensemble": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ], voting='soft')
}

# Streamlit App
st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("👩‍💼 HR Attrition Predictor (Multi-Model)")
st.markdown("_Will this employee leave the company?_ 🤔")

# Model selection
model_name = st.selectbox("Select Model:", list(models.keys()))
model = models[model_name]
model.fit(X_resampled, y_resampled)

# User Input UI
st.header("📥 Enter Employee Details:")
user_input = []
for col in X.columns:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        val = st.selectbox(f"{col}", options)
        enc_val = label_encoders[col].transform([val])[0]
        user_input.append(enc_val)
    else:
        val = st.number_input(f"{col}", value=float(X[col].mean()))
        user_input.append(val)

if st.button("🔮 Predict Attrition"):
    user_arr = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_arr)
    pred = model.predict(user_scaled)[0]
    proba = model.predict_proba(user_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Prediction: Likely to Leave | Probability: {proba:.2%}")
    else:
        st.success(f"✅ Prediction: Likely to Stay | Probability: {1 - proba:.2%}")
