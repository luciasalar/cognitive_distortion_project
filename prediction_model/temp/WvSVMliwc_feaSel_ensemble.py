import pandas as pd
import csv
from gensim.models import KeyedVectors
import numpy as np
import pickle
import re
import nltk
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, Normalizer
from sys import argv
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.combine import SMOTEENN

#direct to ~/cognitive_distortion/prediction_model then run the script
#This is a grid search SVC model with tfidf sentiVec and LIWC as features, feature selection 



#load the SVD feature matrix directly 
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/WikiVecTfidfSVDFea','rb')
x_vec = pickle.load(infile)
infile.close()

#get prediction labels
objects = {}
with open('../data/self_label_distortion2.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['text'])
        texthash = hash(row['text'])
        if texthash not in objects:
            objects[texthash] = MyFea(row['text'])
        objects[texthash].label.append(row['negative_yn_self'])

print('get y labels')
y = getLabel(objects)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.30, random_state=30)
#SGD### use SGD to make prediction, use predictions as features
smote_enn = SMOTEENN(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state = 0)
clf = make_pipeline(SGDClassifier(max_iter= 1000))

parameters = [{'sgdclassifier__alpha': [0.01, 0.05, 0.001, 0.005], 'sgdclassifier__class_weight':['balanced'],
              'sgdclassifier__loss': ['hinge','log','modified_huber','squared_hinge', 'perceptron'], 
               'sgdclassifier__penalty':['none','l1','l2']}]
                   
grid_search_item = GridSearchCV(clf,
                          param_grid = parameters,
                           scoring = 'accuracy',
                           cv = cv,
                           n_jobs = -1)

grid_search = grid_search_item.fit(X_train, y_train)

print('Best scores and best parameters')
print(grid_search.best_score_)
print(grid_search.best_params_)

y_true, y_pred = y_test, grid_search.predict(X_test)
print(classification_report(y_true, y_pred))
  

y_pred = grid_search.predict(X_vec)
  

####combine with liwc
X = np.concatenate((yAsFea, liwc), axis=1)

print('SMOTE again...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==2)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

#Normalize data, convert it to unit vectors
print('compute SVC with optimized parameters...')
#{'svc__C': 1.5, 'svc__class_weight': 'balanced', 'svc__gamma': 0.0001, 'svc__kernel': 'sigmoid'}
cv = StratifiedKFold(n_splits=5, random_state=0)
#svc = make_pipeline(Normalizer(),svm.SVC()) #the normalizer model has poor results
svc = make_pipeline(StandardScaler(),svm.SVC())
parameters = [{'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'svc__gamma': [0.01, 0.001, 0.0001],
                 'svc__C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0,] , 'svc__class_weight':['balanced']}]
               
grid_search_item = GridSearchCV(estimator = svc,
                      param_grid = parameters,
                       cv =  cv,
                       scoring = 'accuracy',
                       n_jobs = -1)
grid_search = grid_search_item.fit(X_train_res, y_train_res)

print('Best scores and best parameters')
print(grid_search.best_score_)
print(grid_search.best_params_)

means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']

y_true, y_pred = y_test, grid_search.predict(X_test)
print(classification_report(y_true, y_pred))

print('Done!')








