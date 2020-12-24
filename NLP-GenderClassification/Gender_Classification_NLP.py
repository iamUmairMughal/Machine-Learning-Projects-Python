import pandas as pd
import numpy as np

#####################################################################
#  Creating Male Data File
#####################################################################


male_dataFile = pd.read_csv("Malefile.csv")
male_dataFile['gender']=lable_male

# print(male_dataFile.shape)
# print(male_dataFile.head(5))


#####################################################################
#  Creating Female Data File
#####################################################################

female_dataFile = pd.read_csv("Femalefile.csv")
female_dataFile['gender']=lable_female
# print(female_dataFile.shape)
# print(female_dataFile.head(5))

#####################################################################
# Merging Both Files and Creating single
#####################################################################

combinedFile = pd.concat([male_dataFile,female_dataFile],ignore_index=True)
combinedFile.to_csv('MeargedDataset.csv')

# print(combinedFile.shape)
# print(combinedFile.head(10))

#####################################################################
# Extracting XY_features
#####################################################################

x_before_cleaning=combinedFile["text"]
# print(x_before_cleaning.size)
y=combinedFile['gender']
# print(y.size)

#####################################################################
# Tokenizing and Cleaning DataSet
#####################################################################

import re
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

Data_tokens=[]

for i in x_before_cleaning:
    i=i.lower()
    # i = re.sub(r'\d+', '', i)
    i = re.sub(r'[^a-zA-Z_\s]+', '', i)
    link_list = re.findall(r'http+\w+', i)
    for link in link_list:
        i = i.replace(link, '')
    # print(i)
    temp = tokenizer.tokenize(i)
    temp = ' '.join(temp)
    temp = re.sub(r'  ', ' ', temp)
    Data_tokens.append([temp])

# Data_tokens = nltk.word_tokenize(x)
Data_tokens=np.array(Data_tokens)

# print(Data_tokens.shape)
# print(Data_tokens)

combinedFile['Text_Without_Punctuation'] = Data_tokens

# print(combinedFile.head())
# print(combinedFile.shape)

x_after_cleaning = combinedFile['Text_Without_Punctuation']

#####################################################################
# Vectorizing DataSet
#####################################################################

from sklearn.feature_extraction.text import CountVectorizer

CountVector = CountVectorizer(strip_accents='unicode',
                              analyzer='word',
                              token_pattern=r'\w{1,}',
                              stop_words='english',
                              ngram_range=(1,1),
                              max_features=3000)
print(CountVector)

X = CountVector.fit_transform(x_after_cleaning)

# print(CountVector.vocabulary_)

count_vectors = CountVector.transform(x_after_cleaning).toarray()
feature_names = CountVector.get_feature_names()

feature_file=pd.DataFrame(count_vectors,
    columns=feature_names
    )
print(feature_file.head())
feature_file.to_csv("features_File.csv")

#####################################################################
# Label Encoding for Y Features
#####################################################################

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
combinedFile['encoded_gender']=le.fit_transform(combinedFile['gender'])
# print(combinedFile.head())
# print(combinedFile.shape)

#####################################################################
# Finalized XY_Features
#####################################################################

x = feature_file.iloc[:,:].values
# print(x.shape)
y= combinedFile['encoded_gender']
# print(y.shape)

#####################################################################
# Splitting Dataset in Training and Testing
#####################################################################

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.20,random_state=5)

#####################################################################
#####################################################################
#                   Phase 1 (Training)
#####################################################################
#####################################################################


#####################################################################
# Importing and training  Models
#####################################################################

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

rf_Cl = RandomForestClassifier(bootstrap=True,class_weight=None,criterion='gini',
                               max_depth=None, max_features='auto',max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1,min_samples_split=2,
                               min_weight_fraction_leaf=0.0,n_estimators=100,n_jobs=1,
                               oob_score=False,random_state=5,verbose=0,
                               warm_start=False)
rf_Cl.fit(x_train,y_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lg_Cl = LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,
                           intercept_scaling=1,max_iter=100, multi_class='ovr',n_jobs=1,
                           penalty='l2',random_state=5,solver='liblinear',tol=0.0001,
                           verbose=0,warm_start=False)
lg_Cl.fit(x_train,y_train)

# Naive Bayes  Model
from sklearn.naive_bayes import BernoulliNB

nv_Cl = BernoulliNB(alpha=1.0, class_prior=None, fit_prior=True)
nv_Cl.fit(x_train,y_train)

# Linear SVM Model
from sklearn.svm import LinearSVC

svm_Cl = LinearSVC(C=1.0, class_weight=None, dual=True,fit_intercept=True,
                   intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                   multi_class='ovr', penalty='l2', random_state=5, tol=0.0001,
                   verbose=0)
svm_Cl.fit(x_train,y_train)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

dt_Cl = DecisionTreeClassifier(criterion = 'entropy', random_state = 5)
dt_Cl.fit(x_train, y_train)

# KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn_Cl= KNeighborsClassifier(n_neighbors=51)
knn_Cl.fit(x_train,y_train)


#####################################################################
#####################################################################
#                   Phase 2 (Testing)
#####################################################################
#####################################################################

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

from sklearn.metrics import  accuracy_score

accuracy = {}

rf_Cl_accuracy = round(accuracy_score(y_test,y_pred_rf_cl)*100,3)
accuracy['Random Forest'] = rf_Cl_accuracy

lg_Cl_accuracy = round(accuracy_score(y_test,y_pred_lg_Cl)*100,3)
accuracy['Logistic Regression'] = lg_Cl_accuracy

svm_Cl_accuracy = round(accuracy_score(y_test,y_pred_svm_cl)*100,3)
accuracy['Linear SVM'] = svm_Cl_accuracy

nv_Cl_accuracy = round(accuracy_score(y_test,y_pred_nv_cl)*100,3)
accuracy['Naive Bayes'] = nv_Cl_accuracy

dt_Cl_accuracy = round(accuracy_score(y_test,y_pred_decision_tree)*100,3)
accuracy['Decision Tree'] = dt_Cl_accuracy

knn_Cl_accuracy = round(accuracy_score(y_test,y_pred_knn)*100,3)
accuracy['KNN'] = knn_Cl_accuracy

best_model = max(rf_Cl_accuracy,svm_Cl_accuracy,nv_Cl_accuracy,lg_Cl_accuracy,dt_Cl_accuracy,knn_Cl_accuracy)
print(accuracy)

import operator
best_model_name = max(accuracy.items(), key=operator.itemgetter(1))[0]

print(f"Most Accurate Model is {best_model_name} with accuracy of {best_model}.")


#####################################################################
#####################################################################
#                   Phase 3 (Application Phase)
#####################################################################
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
import joblib
filename = f'finalized_{best_model_name}_model.pkl'
pickle.dump(trained_model, open(filename, 'wb'))

