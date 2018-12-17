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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sys import argv
import gc
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# Importing the dataset
class MyFea:
    def __init__(self, text):
        self.text = text
        self.userid = hash(self.text)
        self.label = []
        self.vectors = []
        self.meanVec = []
        
    def __hash__(self):
        return self.quoteID
    


#append objects as festure matrix
def append_features(ob):
    count = 0
    proto_matrix = []
    for item in ob:
        col2 = np.append(ob[item].vectors, ob[item].meanVec)
        proto_matrix.append(col2)
        count += 1                 
    return proto_matrix


def getLabel(obj):
    labels = []
    for item in obj:
        labels.append(int(obj[item].label[0]))
    return labels



if __name__ == '__main__':
	#if len(argv) != 3:
	#	print("Usage: " + argv[0] + ' distortion_data sentence_vectors')
	#	exit(1)
	#load the word vector data
	print('is reading data...')
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/wikiVectorsBoW2','rb')
#infile = open(argv[2],'rb')
results = pickle.load(infile)
infile.close()


print('is creating feature matrix...')
	#proto_matrix = append_features(results)
dim_x = len(results.keys()) #932
dim_y = len(results[list(results.keys())[0]].vectors) #5238
dim_z = len(results[list(results.keys())[0]].vectors[0]) #300
print('X: {}, Y: {}, Z: {}'.format(dim_x, dim_y, dim_z))
X = np.zeros((dim_x,dim_y*dim_z), dtype='float32')
dim = 0
for i in results:
	X[dim] = results[i].vectors.reshape(dim_y*dim_z) #because apparently 3D features is too much to ask for
	dim = dim + 1
#X = np.nan_to_num(X)
#get y label
y = getLabel(results)
y = np.array(y)

#X = X.reshape(932,5258*300)
del(results)
gc.collect()


# print('compute SVC with optimized parameters...')

svd_model = TruncatedSVD(n_components=500, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)

svd_transformer = Pipeline([('svd', svd_model)])
svd_matrix = svd_transformer.fit_transform(X)

del(X)
gc.collect()

X_vec = svd_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

del(X)
tmpfile = open('tmptrainX', 'wb')
pickle.dump(X_test, tmpfile)
tmpfile.close()
del(X_test)
gc.collect()

svc = make_pipeline(svm.SVC(gamma=0.001, class_weight='balanced', C = 2, kernel = 'rbf'))
best_m = svc.fit(X_train, y_train)

tmpfile = open('tmptrainX', 'rb')
X_test = pickle.load(tmpfile)
tmpfile.close()

y_true, y_pred = y_test, best_m.predict(X_test)
print(classification_report(y_true, y_pred))


###grid search
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state = 0)
svc = make_pipeline(smote_enn, svm.SVC())
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


#SGD###
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
  

# >>> print(grid_search.best_score_)
# 0.6794478527607362
# >>> print(grid_search.best_params_)
# {'sgdclassifier__alpha': 0.05, 'sgdclassifier__class_weight': 'balanced', 'sgdclassifier__loss': 'log', 'sgdclassifier__penalty': 'l2'}
# >>> 
# >>> y_true, y_pred = y_test, grid_search.predict(X_test)
# >>> print(classification_report(y_true, y_pred))
#               precision    recall  f1-score   support

#            1       0.50      0.58      0.54        99
#            2       0.75      0.69      0.72       181

#    micro avg       0.65      0.65      0.65       280
#    macro avg       0.63      0.63      0.63       280
# weighted avg       0.66      0.65      0.65       280

#
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
  
#without scaler
# 0.7269938650306749
# >>> print(grid_search.best_params_)
# {'sgdclassifier__alpha': 0.005, 'sgdclassifier__class_weight': 'balanced', 'sgdclassifier__loss': 'squared_hinge', 'sgdclassifier__penalty': 'l2'}
# >>> 
# >>> y_true, y_pred = y_test, grid_search.predict(X_test)
# >>> print(classification_report(y_true, y_pred))
#               precision    recall  f1-score   support

#            1       0.56      0.54      0.55        99
#            2       0.75      0.77      0.76       181

#    micro avg       0.69      0.69      0.69       280
#    macro avg       0.66      0.65      0.66       280
# weighted avg       0.69      0.69      0.69       280


#with scaler
# >>> print(grid_search.best_score_)
# 0.7177914110429447
# >>> print(grid_search.best_params_)
# {'sgdclassifier__alpha': 0.001, 'sgdclassifier__class_weight': 'balanced', 'sgdclassifier__loss': 'perceptron', 'sgdclassifier__penalty': 'l2'}
# >>> 
# >>> y_true, y_pred = y_test, grid_search.predict(X_test)
# >>> print(classification_report(y_true, y_pred))
#               precision    recall  f1-score   support

#            1       0.58      0.57      0.57        99
#            2       0.77      0.77      0.77       181

#    micro avg       0.70      0.70      0.70       280
#    macro avg       0.67      0.67      0.67       280
# weighted avg       0.70      0.70      0.70       280

#smote + scaler

# 0.7269938650306749
# >>> print(grid_search.best_params_)
# {'sgdclassifier__alpha': 0.05, 'sgdclassifier__class_weight': 'balanced', 'sgdclassifier__loss': 'perceptron', 'sgdclassifier__penalty': 'l2'}
# >>> 
# >>> y_true, y_pred = y_test, grid_search.predict(X_test)
# >>> print(classification_report(y_true, y_pred))
#               precision    recall  f1-score   support

#            1       0.60      0.58      0.59        99
#            2       0.77      0.79      0.78       181

#    micro avg       0.71      0.71      0.71       280
#    macro avg       0.69      0.68      0.68       280
# weighted avg       0.71      0.71      0.71       280




























