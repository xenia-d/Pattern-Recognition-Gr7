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
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
import os



output_dir = 'task2_plots'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv('Data-PR-As2/creditcard.csv')

X = data.drop(columns = ['Time','Amount','Class'])
y = data['Class']

print(f"Original class distribution: {Counter(y)}")

baseline_accuracies, semi_supervised_accuracies, retrained_accuracies = [], [], []
baseline_f1s, semi_supervised_f1s, retrained_f1s = [], [], []


# Balancing the data 
# https://www.turing.com/kb/smote-for-an-imbalanced-dataset 

smote = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state = 12) 
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"Balanced class distribution: {Counter(y_balanced)}")


for i in range(1):

    print("This is iteration: ", i)

    # Split into train / test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size = 0.2, stratify = y_balanced, random_state = 12)

    # Split train set into labled / unlabeled (30-70)
    X_train_lab, X_train_unlab, y_train_lab, y_train_unlab = train_test_split(X_train, y_train, test_size=0.7, stratify=y_train, random_state=12)


    print(f"Train set class distribution: {Counter(y_train)}")
    print(f"Test set class distribution: {Counter(y_test)}")


    ###################################### BASELINE MODEL - step 2 #####################################

    baseline_knn = KNeighborsClassifier()
    baseline_knn.fit(X_train_lab, y_train_lab)


    y_pred = baseline_knn.predict(X_test)
    accuracy_baseline = accuracy_score(y_test, y_pred)
    print("Accuracy of baseline:", accuracy_baseline)

    f1_baseline = f1_score(y_test, y_pred)
    print("F1 Score of baseline:", f1_baseline)


    baseline_accuracies.append(accuracy_score(y_test, y_pred))
    baseline_f1s.append(f1_baseline)


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
    semi_supervised_f1s.append(f1_semi_sup)



    ############################## RETRAINED BASELINE WITH RETRIEVED LABELS - step 4 #############################

    retrieved_labels = semi_sup_model.transduction_

    retrained_baseline = KNeighborsClassifier()
    retrained_baseline.fit(X_train, retrieved_labels)

    y_pred_retrained = retrained_baseline.predict(X_test)

    accuracy_retrained_baseline = accuracy_score(y_test, y_pred_retrained)
    f1_retrained_baseline = f1_score(y_test, y_pred_retrained)

    print("Accuracy: of retrained baseline ", accuracy_retrained_baseline)
    print("F1 Score of retrained baseline: ", f1_retrained_baseline)

    retrained_accuracies.append(accuracy_score(y_test, y_pred_retrained))
    retrained_f1s.append(f1_retrained_baseline)



# Calculate mean accuracy and F1 score for each model
mean_accuracy_baseline = np.mean(baseline_accuracies)
mean_f1_baseline = np.mean(baseline_f1s)

mean_accuracy_semi_supervised = np.mean(semi_supervised_accuracies)
mean_f1_semi_supervised = np.mean(semi_supervised_f1s)

mean_accuracy_retrained = np.mean(retrained_accuracies)
mean_f1_retrained = np.mean(retrained_f1s)



# Print the mean values
print(f"\nMean accuracy and F1 scores over 100 iterations:\n")
print(f"Baseline KNN - Mean Accuracy: {mean_accuracy_baseline:.4f}, Mean F1 Score: {mean_f1_baseline:.4f}")
print(f"Semi-supervised - Mean Accuracy: {mean_accuracy_semi_supervised:.4f}, Mean F1 Score: {mean_f1_semi_supervised:.4f}")
print(f"Retrained KNN - Mean Accuracy: {mean_accuracy_retrained:.4f}, Mean F1 Score: {mean_f1_retrained:.4f}")

# Plot accuracy across iterations
iterations = range(1)
plt.plot(iterations, baseline_accuracies, label='Baseline KNN', color='blue')
plt.plot(iterations, semi_supervised_accuracies, label='Semi-supervised', color='orange')
plt.plot(iterations, retrained_accuracies, label='Retrained KNN', color='green')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Iteration vs. Accuracy for Baseline, Semi-Supervised, and Retrained KNN Models')
plt.legend()
plt.show()


# Save the figure
acc_iter_path = os.path.join(output_dir, 'acc_vs_iter.png')
plt.savefig(acc_iter_path)
plt.close()



# Prepare to plot ROC curves on the same axis
plt.figure(figsize=(10, 8))

# Plot ROC curve for Baseline KNN
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, baseline_knn.predict_proba(X_test)[:, 1])
roc_auc_baseline = auc(fpr_baseline, tpr_baseline)
plt.plot(fpr_baseline, tpr_baseline, color='blue', lw=2, label='Baseline KNN AUC = {:.2f}'.format(roc_auc_baseline))

# Plot ROC curve for Semi-supervised model (using one-vs-rest approximation)
y_test_binarized = label_binarize(y_test, classes=[0, 1])
y_pred_semi_sup = label_binarize(semi_sup_model.predict(X_test), classes=[0, 1])
fpr_semi, tpr_semi, _ = roc_curve(y_test_binarized.ravel(), y_pred_semi_sup.ravel())
roc_auc_semi = roc_auc_score(y_test_binarized, y_pred_semi_sup, average="macro")
plt.plot(fpr_semi, tpr_semi, color='orange', lw=2, label='Semi-supervised AUC (OvR) = {:.2f}'.format(roc_auc_semi))

# Plot ROC curve for Retrained KNN
fpr_retrained, tpr_retrained, _ = roc_curve(y_test, retrained_baseline.predict_proba(X_test)[:, 1])
roc_auc_retrained = auc(fpr_retrained, tpr_retrained)
plt.plot(fpr_retrained, tpr_retrained, color='green', lw=2, label='Retrained KNN AUC = {:.2f}'.format(roc_auc_retrained))

# Plot the diagonal line for random guessing
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

# Customize the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Baseline, Semi-supervised, and Retrained KNN Models')
plt.legend(loc='lower right')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.tight_layout()


# Save the figure
roc_plot_path = os.path.join(output_dir, 'roc_curves.png')
plt.savefig(roc_plot_path)
plt.close()