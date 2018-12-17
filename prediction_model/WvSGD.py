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


def unpackFeatureMatrix(object, dim_x,dim_y,dim_z):
    X = np.zeros((dim_x,dim_y*dim_z), dtype='float32')
    dim = 0
    for i in results:
      X[dim] = results[i].vectors.reshape(dim_y*dim_z) #because apparently 3D features is too much to ask for
      dim = dim + 1
    return X



if __name__ == '__main__':
    #if len(argv) != 3:
    #   print("Usage: " + argv[0] + ' distortion_data sentence_vectors')
    #   exit(1)
    #load the word vector data
    print('is reading data...')
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/wikiVectorsBoWTfidfSparse','rb')
#infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/wikiVectorsBoW2','rb')
#infile = open(argv[2],'rb')
results = pickle.load(infile)
infile.close()

print('get y labels')
y = getLabel(results)
y = np.array(y)


print('is creating feature matrix...')
    #proto_matrix = append_features(results)

dim_x = len(results.keys()) #932
dim_y = len(results[list(results.keys())[0]].vectors) #5258
dim_z = len(results[list(results.keys())[0]].vectors[0]) #300
print('X: {}, Y: {}, Z: {}'.format(dim_x, dim_y, dim_z))
X = unpackFeatureMatrix(results)


del(results)
gc.collect()


#if the data is condensed matrix, we need to use this to unpack it 
#reshape tfidf dim 

def unpackFeatureMatrixSparse(object):
    X = np.zeros((932,5258*300), dtype='float32')
    dim = 0
    for i in results:
      X[dim] = results[i].vectors.todense().reshape(5258*300)#because apparently 3D features is too much to ask for
      dim = dim + 1
    return X

X = unpackFeatureMatrixSparse(results)


print('compute SVC with optimized parameters...')
svd_model = TruncatedSVD(n_components=500, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)

svd_matrix = svd_model.fit_transform(X)



#we can skip the previous step and load the SVD feature matrix directly 
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/WikiVecTfidfSVDFea','rb')
results = pickle.load(infile)
infile.close()


del(X)
gc.collect()

X_vec = svd_matrix

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.30, random_state=30)


#SGD###
smote_enn = SMOTEENN(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state = 0)
clf = make_pipeline(smote_enn, SGDClassifier(max_iter= 1000))

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
  































