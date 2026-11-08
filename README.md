Dry Bean Classification - ML Assignment 2
Student ID: 2026AK05727  
Course: Machine Learning (AIMLZG565), M.Tech AIML, BITS WILP
a. Problem Statement
The goal is to identify the variety of a dry bean grain from 16 shape and size measurements taken from images of the grain. There are 7 varieties, so this is a multi-class classification problem with the column Class as the target. A model like this can support automatic seed sorting and quality control, where doing it by hand is slow and inconsistent. Five classifiers are trained on the same data and compared with six evaluation metrics.
b. Dataset Description
Source: UCI Machine Learning Repository - Dry Bean Dataset (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), CC BY 4.0
Instances: 13,611 rows. 68 duplicate rows (all of the Horoz class) were removed, leaving 13,543 rows.
Features: 16 numeric features: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1 to ShapeFactor4. No missing values.
Target: Class, 7 types: Dermason 3546, Sira 2636, Seker 2027, Horoz 1860, Cali 1630, Barbunya 1322, Bombay 522 (after removing duplicates). The classes are imbalanced.
Preprocessing: Duplicates removed, class names converted to numbers with LabelEncoder, features scaled with StandardScaler (fitted on the training data only).
Split: 80/20 stratified split with random_state=17: 10,834 training rows and 2,709 test rows. The test rows are saved as test_data.csv in the repository.
Metrics: Precision, Recall and F1 are weighted averages over the 7 classes. AUC is one-vs-rest, weighted. MCC is the multi-class Matthews correlation coefficient.
c. GitHub Repository
https://github.com/avinashsingh30101985/drybean-ml-assignment2
d. Models Used
Comparison table (test set, 2,709 rows)
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9228	0.9932	0.9227	0.9228	0.9227	0.9067
Decision Tree	0.9103	0.9772	0.9103	0.9103	0.9102	0.8914
KNN	0.9254	0.9825	0.9254	0.9254	0.9253	0.9098
Naive Bayes	0.8944	0.9898	0.8948	0.8944	0.8944	0.8726
Support Vector Machine	0.9299	0.9943	0.9299	0.9299	0.9296	0.9152
Observations
ML Model Name	Observation about model performance
Logistic Regression	Accuracy 0.9228 with the second best AUC (0.9932). Training accuracy (0.926) and test accuracy (0.923) are almost the same, so it does not overfit. It works well because the scaled features separate the beans reasonably well with linear boundaries. It is weakest on Sira (F1 0.862).
Decision Tree	Accuracy 0.9103 and the lowest AUC (0.9772). Training accuracy is 0.9465 against 0.9103 on the test set, a gap of about 3.6 points, so it overfits a little even with max_depth=8. Its probabilities are coarse because each leaf gives only a few distinct values, which lowers the AUC. Its advantage is that the rules are easy to read.
KNN	Accuracy 0.9254, the second best, and MCC 0.9098. Its AUC (0.9825) is lower than Logistic Regression and SVM because with k=5 the predicted probabilities can only take a few values. It needs scaled features to work properly, and it is slower at prediction time because it stores all the training rows.
Naive Bayes	Lowest accuracy (0.8944) and MCC (0.8726), but a high AUC (0.9898), so it ranks the classes well while making more wrong final decisions. It assumes the features are independent, but 15 pairs of features have a correlation above 0.95 (for example Area and ConvexArea are about 1.0, Area and EquivDiameter 0.985), so the assumption does not hold. It is weakest on Barbunya (F1 0.83). It is very fast to train.
Support Vector Machine	Best result on every metric: accuracy 0.9299, AUC 0.9943, F1 0.9296, MCC 0.9152. Training accuracy (0.932) and test accuracy (0.930) are close, so it generalises well. The RBF kernel can follow the curved boundaries between the bean types. The cost is slower training and prediction than the simpler models.
Overall winner	Support Vector Machine is the winner. It has the highest Accuracy, AUC, Precision, Recall, F1 and MCC, and it does not overfit. The lead is small (about 0.5 points of accuracy over KNN and 0.7 points over Logistic Regression) and comes from a single train/test split, so Logistic Regression and KNN are close alternatives. All models find Bombay easy (F1 1.0), while Sira is the hardest class for most of them and is often mistaken for Dermason.
Streamlit App
Live app: https://drybean-ml-assignment2-g79wvgddb7hcxq4qwmwwwc.streamlit.app/
Features: CSV upload (test data), model dropdown, evaluation metrics, confusion matrix, classification report, and a table comparing all 5 models.
How to run
```
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```
