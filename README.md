# Dry Bean Classification - ML Assignment 2

**Student ID:** 2026AK05727
**Course:** Machine Learning (AIMLZG565), M.Tech AIML, BITS WILP

## a. Problem Statement
(write 3-4 lines in your own words: what is being predicted and why)

## b. Dataset Description
- Source: UCI Machine Learning Repository - Dry Bean Dataset (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- Instances: 13,611 (xxxx after removing duplicates)
- Features: 16 numeric features (shape and size measurements of bean grains)
- Target: Class - 7 bean types (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira)
- Preprocessing: removed duplicates, StandardScaler, 80/20 stratified split
- (add class distribution here)

## c. GitHub Repository
(paste link)

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

Precision, Recall and F1 are weighted averages (7 classes). AUC is one-vs-rest weighted.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | |

## Streamlit App
(paste live link)

## How to run
```
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```
