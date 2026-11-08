# ML Assignment 2 - 2026AK05727
# Machine Learning (AIMLZG565), M.Tech AIML, BITS WILP
# Trains 5 classification models on the Dry Bean dataset and saves them.

import os
import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

os.makedirs("model", exist_ok=True)

# load data
data = pd.read_csv("data/dataset.csv")
print("original shape:", data.shape)
data = data.drop_duplicates()
print("after removing duplicates:", data.shape)

X = data.drop("Class", axis=1)
y_text = data["Class"]

# class names -> numbers
le = LabelEncoder()
y = le.fit_transform(y_text)
print("classes:", list(le.classes_))

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y)

# scaling (fit only on train data)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=17),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=17),
}

rows = []
for name, model in models.items():
    print("training", name, "...")
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_test, pred, average="weighted"),
        "Recall": recall_score(y_test, pred, average="weighted"),
        "F1": f1_score(y_test, pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    file_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, "model/" + file_name)

results = pd.DataFrame(rows).round(4)
print()
print(results.to_string(index=False))
results.to_csv("model_results.csv", index=False)

# save scaler, label encoder and the test data (with original class names)
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(le, "model/label_encoder.joblib")
test_data = X_test.copy()
test_data["Class"] = le.inverse_transform(y_test)
test_data.to_csv("test_data.csv", index=False)

print("\ntest_data.csv saved with", len(test_data), "rows")
print("scikit-learn version used:", sklearn.__version__)
