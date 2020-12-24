#####################################################################
# Importing Dataset
#####################################################################

import pandas as pd
dataset = pd.read_csv("creditcard.csv")
# print(dataset.head())


#####################################################################
# Data Preprocessing
#####################################################################

import numpy as np

dataset.iloc[:,:-1] = dataset.iloc[:,:-1].replace(np.nan,dataset.iloc[:,:-1].mean())
dataset["Class"] = dataset['Class'].replace(np.nan,0)

# dataset.isnull().sum()

# labels= dataset.columns
# labels=labels.drop('Class')
# labels
# for i in labels:
  # dataset[i] = dataset[i].replace(np.nan,dataset[i].mean())
#####################################################################
# Extracting XY-Features
#####################################################################

x = dataset.iloc[:,:-1].values
y = dataset['Class'].values

#####################################################################
# Splitting Datset into Training and Testing
#####################################################################

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .20, random_state = 5)

#####################################################################
# Importing and training  Models
#####################################################################

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#
# rf_Cl = RandomForestClassifier(bootstrap=True,class_weight=None,criterion='gini',
#                                max_depth=None, max_features='auto',max_leaf_nodes=None,
#                                min_impurity_decrease=0.0, min_impurity_split=None,
#                                min_samples_leaf=1,min_samples_split=2,
#                                min_weight_fraction_leaf=0.0,n_estimators=100,n_jobs=1,
#                                oob_score=False,random_state=5,verbose=0,
#                                warm_start=False)

print("Training Random Forest Model.....")
rf_Cl = RandomForestClassifier(n_estimators=100)
rf_Cl.fit(x_train,y_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# lg_Cl = LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,
#                            intercept_scaling=1,max_iter=100, multi_class='ovr',n_jobs=1,
#                            penalty='l2',random_state=5,solver='liblinear',tol=0.0001,
#                            verbose=0,warm_start=False)

print("Training Logistic Regression Model.....")
lg_Cl = LogisticRegression()
lg_Cl.fit(x_train,y_train)

# Naive Bayes  Model

from sklearn.naive_bayes import GaussianNB

print("Training Naive Bayes Model.....")
nv_Cl = GaussianNB()
nv_Cl.fit(x_train,y_train)

# Linear SVM Model
from sklearn import svm

print("Training SVM Model.....")
svm_Cl=svm.SVC(kernel='linear',C=10,gamma=1).fit(x_train,y_train)
svm_Cl.fit(x_train,y_train)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

print("Training Decision Tree Model.....")
dt_Cl = DecisionTreeClassifier(criterion = 'entropy', random_state = 5)
dt_Cl.fit(x_train, y_train)

# KNN Model
from sklearn.neighbors import KNeighborsClassifier

print("Training KNN Model.....")
knn_Cl= KNeighborsClassifier(n_neighbors = 51)
knn_Cl.fit(x_train,y_train)


#####################################################################
# Model Testing (Predicting Results)
#####################################################################

y_pred_rf_cl = rf_Cl.predict(x_test)                    # Random Forest Classifier
y_pred_lg_Cl = lg_Cl.predict(x_test)                    # Logistic Regression
y_pred_nv_cl = nv_Cl.predict(x_test)                    # Naive Bayes Classifier
y_pred_svm_cl = svm_Cl.predict(x_test)                  # Linear SVM Classifier
y_pred_decision_tree = dt_Cl.predict(x_test)    # Decision Tree Classifier
y_pred_knn=knn_Cl.predict(x_test)                      # KNN Classifier

#####################################################################
# Predicting Accuracy
#####################################################################

from sklearn.metrics import accuracy_score

accuracy = {}

rf_Cl_accuracy = round(accuracy_score(y_test,y_pred_rf_cl)*100,3)
# print("Accuracy_Score (Random Forest): ",(rf_Cl_accuracy))
accuracy['Random Forest'] = rf_Cl_accuracy

lg_Cl_accuracy = round(accuracy_score(y_test,y_pred_lg_Cl)*100,3)
# print("Accuracy_Score (Logistic Regression): ", (lg_Cl_accuracy))
accuracy['Logistic Regression'] = lg_Cl_accuracy

svm_Cl_accuracy = round(accuracy_score(y_test,y_pred_svm_cl)*100,3)
# print("Accuracy_Score (SVM): "%,(svm_Cl_accuracy))
accuracy['Linear SVM'] = svm_Cl_accuracy

nv_Cl_accuracy = round(accuracy_score(y_test,y_pred_nv_cl)*100,3)
# print("Accuracy_Score (Naive_Bayes): ", (nv_Cl_accuracy))
accuracy['Naive Bayes'] = nv_Cl_accuracy

dt_Cl_accuracy = round(accuracy_score(y_test,y_pred_decision_tree)*100,3)
# print("Accuracy_Score (Decision Tree):", (dt_Cl_accuracy))
accuracy['Decision Tree'] = dt_Cl_accuracy

knn_Cl_accuracy = round(accuracy_score(y_test,y_pred_knn)*100,3)
# print("Accuracy_Score (KNN):",(knn_Cl_accuracy))
accuracy['KNN'] = knn_Cl_accuracy

best_model = max(rf_Cl_accuracy,svm_Cl_accuracy,nv_Cl_accuracy,lg_Cl_accuracy,dt_Cl_accuracy,knn_Cl_accuracy)
print(accuracy)

import operator
best_model_name = max(accuracy.items(), key=operator.itemgetter(1))[0]

print(f"Most Accurate Model is {best_model_name} with accuracy of {best_model}.")

# from sklearn.metrics import confusion_matrix

# cm_decision = confusion_matrix(y_test, y_pred_decision_tree)
# print("confusion Marix : \n", cm_decision)
# Accuracy_Decison = ((cm_decision[0][0] + cm_decision[1][1]) / cm_decision.sum()) *100
# print("Accuracy_Decison    : ", Accuracy_Decison)

# Error_rate_Decison = ((cm_decision[0][1] + cm_decision[1][0]) / cm_decision.sum()) *100
# print("Error_rate_Decison  : ", Error_rate_Decison)

# # True Fake Recognition Rate
# Specificity_Decison = (cm_decision[1][1] / (cm_decision[1][1] + cm_decision[0][1])) *100
# print("Specificity_Decison : ", Specificity_Decison)

# # True Genuine Recognition Rate
# Sensitivity_Decison = (cm_decision[0][0] / (cm_decision[0][0] + cm_decision[1][0])) *100
# print("Sensitivity_Decison : ", Sensitivity_Decison)

#####################################################################
# Training Best Model
#####################################################################

models = {
    'Random Forest': rf_Cl,
    'Logistic Regression': lg_Cl,
    'Linear SVM': svm_Cl,
    'Naive Bayes': nv_Cl,
    'Decision Tree': dt_Cl,
    'KNN': knn_Cl
}

trained_model=models.get(best_model_name)
trained_model.fit(x,y)

#####################################################################
# Saving Model as Pickle File
#####################################################################

import pickle
filename = f'MidTerm_Model_{best_model_name}.pkl'
pickle.dump(trained_model, open(filename, 'wb'))
