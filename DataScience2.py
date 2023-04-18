import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import joblib


train_data = pd.read_csv("./train.csv")
train_data.head()
# print(train_data)

test_data = pd.read_csv("./test.csv")
test_data.head()
# print(test_data)

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
# print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

# print("% of men who survived:", rate_men)

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

y_test = pd.read_csv("gender_submission.csv")["Survived"]
X_test = pd.get_dummies(test_data[features])


#-------------------------------------------------------------
print("------------------------------------------- ")
print("Support Vector: ")
"""
param_grid = {
    'C': [ 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4]
    #'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, num=6, base=2)),
   # 'class_weight': [None, 'balanced']
}

svm = SVC()


grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)


grid_search.fit(X, y)


results = grid_search.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print("Score: {:.3f} for params: {}".format(mean_score, params))


print("Mejor Hyperparametros:", grid_search.best_params_)
print("Mejor Score:", grid_search.best_score_)

best_params = grid_search.best_params_
svc = SVC(random_state=1, **best_params)
svc.fit(X, y)


joblib.dump(svc, "SVC.pkl")
"""
SVC = joblib.load("SVC.pkl")

accuracy_scores = cross_val_score(SVC, X_test, y_test, cv=5)
print("Accuracy: {:.5f} ".format(np.mean(accuracy_scores)))

auc_scores = cross_val_score(SVC,  X_test, y_test, cv=5, scoring="roc_auc")
print("AUC-ROC score: {:.5f} ".format(np.mean(auc_scores)))

precision_scores = cross_val_score(SVC, X_test, y_test, cv=5, scoring="precision")
print("Precision: {:.5f}".format(np.mean(precision_scores)))

recall_scores = cross_val_score(SVC, X_test, y_test, cv=5, scoring="recall")
print("Recall: {:.5f} ".format(np.mean(recall_scores)))

f1_scores = cross_val_score(SVC, X_test, y_test, cv=5, scoring="f1")
print("F1 score: {:.5f} ".format(np.mean(f1_scores)))

#-------------------------------------------------------------
print("------------------------------------------- ")
print("Random Forest: ")

"""""
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

model = RandomForestClassifier(random_state=1)


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)


grid_search.fit(X, y)


results = grid_search.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print("Score: {:.3f} for params: {}".format(mean_score, params))


print("Mejor Hyperparametros:", grid_search.best_params_)
print("Mejor Score:", grid_search.best_score_)

best_params = grid_search.best_params_
RandomForest = RandomForestClassifier(random_state=1, **best_params)
RandomForest.fit(X, y)

joblib.dump(RandomForest, "RandomForest.pkl")
"""""
RandomForest = joblib.load("RandomForest.pkl")

accuracy_scores = cross_val_score(RandomForest, X_test, y_test, cv=5)
print("Accuracy: {:.5f} ".format(np.mean(accuracy_scores)))

auc_scores = cross_val_score(RandomForest, X_test, y_test, cv=5, scoring="roc_auc")
print("AUC-ROC score: {:.5f} ".format(np.mean(auc_scores)))

precision_scores = cross_val_score(RandomForest, X_test, y_test, cv=5, scoring="precision")
print("Precision: {:.5f}".format(np.mean(precision_scores)))

recall_scores = cross_val_score(RandomForest, X_test, y_test, cv=5, scoring="recall")
print("Recall: {:.5f} ".format(np.mean(recall_scores)))

f1_scores = cross_val_score(RandomForest, X_test, y_test, cv=5, scoring="f1")
print("F1 score: {:.5f} ".format(np.mean(f1_scores)))

#-------------------------------------------------------------
print("------------------------------------------- ")
print("Extra Trees: ")
""""
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

model = ExtraTreesClassifier()

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

grid_search.fit(X, y)


results = grid_search.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print("Score: {:.3f} for params: {}".format(mean_score, params))


print("Mejor Hyperparametros:", grid_search.best_params_)
print("Mejor Score:", grid_search.best_score_)

best_params = grid_search.best_params_
ExtraTrees = ExtraTreesClassifier(random_state=1, **best_params)
ExtraTrees.fit(X, y)

joblib.dump(ExtraTrees, "ExtraTrees.pkl")
"""

ExtraTrees = joblib.load("ExtraTrees.pkl")

accuracy_scores = cross_val_score(ExtraTrees, X_test, y_test, cv=5)
print("Accuracy: {:.5f} ".format(np.mean(accuracy_scores)))

auc_scores = cross_val_score(ExtraTrees, X_test, y_test, cv=5, scoring="roc_auc")
print("AUC-ROC score: {:.5f} ".format(np.mean(auc_scores)))

precision_scores = cross_val_score(ExtraTrees, X_test, y_test, cv=5, scoring="precision")
print("Precision: {:.5f}".format(np.mean(precision_scores)))

recall_scores = cross_val_score(ExtraTrees, X_test, y_test, cv=5, scoring="recall")
print("Recall: {:.5f} ".format(np.mean(recall_scores)))

f1_scores = cross_val_score(ExtraTrees, X_test, y_test, cv=5, scoring="f1")
print("F1 score: {:.5f} ".format(np.mean(f1_scores)))

print("------------------------------------------- ")
