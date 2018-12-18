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
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sys import argv
import gc
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler, Normalizer


infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/wikiVectorsBoWTfidfSparse','rb')
#infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/wikiVectorsBoW2','rb')
#infile = open(argv[2],'rb')
results = pickle.load(infile)
infile.close()


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



#we can skip the previous step and load the SVD feature matrix directly 
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/WikiVecTfidfSVDFea','rb')
results = pickle.load(infile)
infile.close()

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.30, random_state=30)


#SGD###
smote_enn = SMOTEENN(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state = 0)
clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter= 1000))

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
  

####combine with liwc
text_liwc = pd.read_csv('./data/LIWC_self_label_valence.csv')
liwc = text_liwc.loc[:,'function':'OtherP'].values


X = np.concatenate((X_vec, liwc), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

clf = make_pipeline(smote_enn, StandardScaler(),SGDClassifier(max_iter= 1000))

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
  
































