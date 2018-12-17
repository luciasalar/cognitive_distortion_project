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
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sys import argv
import gc
from sklearn.decomposition import TruncatedSVD
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.combine import SMOTEENN


#we can skip the previous step and load the SVD feature matrix directly 
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/WikiVecTfidfSVDFea','rb')
results = pickle.load(infile)
infile.close()



X_vec = svd_matrix
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.30, random_state=30)
###grid search
smote_enn = SMOTEENN(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state = 0)
svc = make_pipeline(svm.SVC())
parameters = [{'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'svc__gamma': [0.01, 0.001, 0.0001],
                     'svc__C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 10] , 'svc__class_weight':['balanced']}]
                   
grid_search_item = GridSearchCV(estimator = svc,
                          param_grid = parameters,
                           cv =  cv,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search_item.fit(X_train, y_train)

print('Best scores and best parameters')
print(grid_search.best_score_)
print(grid_search.best_params_)

y_true, y_pred = y_test, grid_search.predict(X_test)
print(classification_report(y_true, y_pred))
  

print('Done!')



#tricks to free memory
# del(X)
# tmpfile = open('tmptrainX', 'wb')
# pickle.dump(X_test, tmpfile)
# tmpfile.close()
# del(X_test)
# gc.collect()

# svc = make_pipeline(svm.SVC(gamma=0.001, class_weight='balanced', C = 2, kernel = 'rbf'))
# best_m = svc.fit(X_train, y_train)

# tmpfile = open('tmptrainX', 'rb')
# X_test = pickle.load(tmpfile)
# tmpfile.close()

# y_true, y_pred = y_test, best_m.predict(X_test)
# print(classification_report(y_true, y_pred))



















