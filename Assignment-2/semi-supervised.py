# https://machinelearningmastery.com/semi-supervised-learning-with-label-propagation/ 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
from sklearn.semi_supervised import LabelPropagation



data = pd.read_csv('Data-PR-As2\creditcard.csv')

X = data.drop(columns = ['Time','Amount','Class'])
y = data['Class']

print(f"Original class distribution: {Counter(y)}")

baseline_accuracies, semi_supervised_accuracies, retrained_accuracies = [], [], []
baseline_roc_aucs, semi_supervised_roc_aucs, retrained_roc_aucs = [], [], []


# Balancing the data 
# https://www.turing.com/kb/smote-for-an-imbalanced-dataset 

smote = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = 12) 
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"Balanced class distribution: {Counter(y_balanced)}")


# Split into train / test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size = 0.2, stratify = y_balanced, random_state = 12)

# Split train set into labled / unlabeled (30-70)
X_train_lab, X_train_unlab, y_train_lab, y_train_unlab = train_test_split(X_train, y_train, test_size=0.7, stratify=y_train, random_state=12)


print(f"Train set class distribution: {Counter(y_train)}")
print(f"Test set class distribution: {Counter(y_test)}")


###################################### BASELINE MODEL - step 2 #####################################

rf = RandomForestClassifier(n_estimators=100, random_state=12)
rf.fit(X_train_lab, y_train_lab)


y_pred = rf.predict(X_test)
accuracy_baseline = accuracy_score(y_test, y_pred)
print("Accuracy of baseline:", accuracy_baseline)

f1_baseline = f1_score(y_test, y_pred)
print("F1 Score of baseline:", f1_baseline)


baseline_accuracies.append(accuracy_score(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
baseline_roc_aucs.append(auc(fpr, tpr))


###################################### SEMI-SUPERVISED MODEL - step 3 #####################################
# https://medium.com/geekculture/semi-supervised-learning-label-propagation-for-classification-1963439531cb

y_train_unlab[:] = -1

X_train_semi_sup = np.vstack((X_train_lab, X_train_unlab))
y_train_semi_sup = np.concatenate((y_train_lab, y_train_unlab))

semi_sup_model = LabelPropagation(kernel='knn')
semi_sup_model.fit(X_train_semi_sup, y_train_semi_sup)

y_pred = semi_sup_model.predict(X_test)


accuracy_semi_sup = accuracy_score(y_test, y_pred)
f1_semi_sup = f1_score(y_test, y_pred)

print("Accuracy of Semi-supervised model: ", accuracy_semi_sup)
print("F1 Score of Semi-supervised model: ", f1_semi_sup)


class_distribution = Counter(y_train_semi_sup.flatten())
print("Class distribution in semi-supervised training set: ", class_distribution)

semi_supervised_accuracies.append(accuracy_score(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, semi_sup_model.predict_proba(X_test)[:, 1])
semi_supervised_roc_aucs.append(auc(fpr, tpr))


############################## RETRAINED BASELINE WITH RETRIEVED LABELS - step 4 #############################

retrieved_labels = semi_sup_model.transduction_

retrained_baseline = RandomForestClassifier(n_estimators=100, random_state=12)
retrained_baseline.fit(X_train, retrieved_labels)

y_pred_retrained = retrained_baseline.predict(X_test)

accuracy_retrained_rf = accuracy_score(y_test, y_pred_retrained)
f1_retrained_rf = f1_score(y_test, y_pred_retrained)

print("Accuracy: of retrained baseline ", accuracy_retrained_rf)
print("F1 Score of retrained baseline: ", f1_retrained_rf)

retrained_accuracies.append(accuracy_score(y_test, y_pred_retrained))
fpr, tpr, _ = roc_curve(y_test, retrained_rf.predict_proba(X_test)[:, 1])
retrained_roc_aucs.append(auc(fpr, tpr))


