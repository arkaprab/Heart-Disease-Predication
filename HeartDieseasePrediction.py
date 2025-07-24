import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

@st.cache_data
def generate_data(n_samples=1000):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=13,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        flip_y=0.03,
        class_sep=1.5,
        random_state=42
    )
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df = df.apply(lambda col: (col - col.min()) / (col.max() - col.min()) if col.name != "sex" else col)
    df["age"] = (df["age"] * 50 + 30).astype(int)
    df["trestbps"] = (df["trestbps"] * 80 + 90).astype(int)
    df["chol"] = (df["chol"] * 200 + 100).astype(int)
    df["thalach"] = (df["thalach"] * 100 + 70).astype(int)
    df["oldpeak"] = (df["oldpeak"] * 4).round(1)
    df["target"] = y
    return df

data = generate_data()
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

st.title("💓 Heart Disease Prediction App")
st.write("This app uses machine learning to predict the presence of heart disease based on patient features.")

if st.checkbox("🔍 Show Sample Data"):
    st.dataframe(data.head(20))

st.subheader("📊 Model Comparison (Cross-Validation Accuracy)")
model_choice = st.selectbox("Choose a Model", list(models.keys()))
if st.button("Run Cross-Validation"):
    score = cross_val_score(models[model_choice], X_train, y_train, cv=5, scoring='accuracy').mean()
    st.success(f"{model_choice} Average Accuracy: {score:.4f}")

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

st.subheader("✅ Model Performance on Test Set")
st.write(f"**Accuracy:** {acc:.4f} | **Precision:** {prec:.4f} | **Recall:** {rec:.4f}")
st.json(report)

st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.subheader("🧑‍⚕️ Predict Heart Disease from Patient Info")

input_data = []
slider_config = {
    "age": (30, 80, 50),
    "sex": (0, 1, 0),
    "cp": (0, 3, 1),
    "trestbps": (90, 200, 120),
    "chol": (100, 400, 200),
    "fbs": (0, 1, 0),
    "restecg": (0, 2, 1),
    "thalach": (70, 200, 120),
    "exang": (0, 1, 0),
    "oldpeak": (0.0, 6.0, 1.0),
    "slope": (0, 2, 1),
    "ca": (0, 4, 0),
    "thal": (0, 3, 1)
}

for col in X.columns:
    config = slider_config[col]
    val = st.slider(f"{col}", float(config[0]), float(config[1]), float(config[2]))
    input_data.append(val)

if st.button("Predict Now"):
    scaled_input = scaler.transform([input_data])
    result = final_model.predict(scaled_input)[0]
    if result == 1:
        st.error("⚠️ High Risk of Heart Disease Detected!")
    else:
        st.success("✅ Low Risk of Heart Disease")
