# ML Assignment 2 - 2026AK05727
# Machine Learning (AIMLZG565), M.Tech AIML, BITS WILP

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def load_files():
    models = {}
    for name in model_files:
        models[name] = joblib.load(model_files[name])
    scaler = joblib.load("model/scaler.joblib")
    le = joblib.load("model/label_encoder.joblib")
    return models, scaler, le


def get_metrics(y_true, pred, prob):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, prob, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_true, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }


models, scaler, le = load_files()

st.title("Dry Bean Classification")
st.write("Upload the test data, pick a model and see how it performs. "
         "The models are already trained, nothing is trained here.")

st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.sidebar.selectbox("Choose a model", list(model_files.keys()))
st.sidebar.write("---")
st.sidebar.caption("2026AK05727 | ML (AIMLZG565) | Assignment 2")

if uploaded is None:
    st.info("Please upload test_data.csv from the GitHub repo to continue.")
    st.stop()

df = pd.read_csv(uploaded)

if "Class" not in df.columns:
    st.error("The file must have the 'Class' column.")
    st.stop()

feature_cols = list(scaler.feature_names_in_)
missing = [c for c in feature_cols if c not in df.columns]
if len(missing) > 0:
    st.error("These columns are missing in the file: " + ", ".join(missing))
    st.stop()

st.subheader("Uploaded data")
st.write("Rows:", df.shape[0], " Columns:", df.shape[1])
st.dataframe(df.head())

# prepare the data the same way as in training
X = scaler.transform(df[feature_cols])
y_true = le.transform(df["Class"])

# selected model
model = models[model_name]
pred = model.predict(X)
prob = model.predict_proba(X)
m = get_metrics(y_true, pred, prob)

st.subheader("Results - " + model_name)
cols = st.columns(6)
for i, key in enumerate(m):
    cols[i].metric(key, round(m[key], 4))

col1, col2 = st.columns(2)

with col1:
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.write("Classification Report")
    report = classification_report(y_true, pred, target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

# all models together
st.subheader("All models on this test data")
rows = []
for name in models:
    p = models[name].predict(X)
    pr = models[name].predict_proba(X)
    r = get_metrics(y_true, p, pr)
    r["ML Model Name"] = name
    rows.append(r)
table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(table)
