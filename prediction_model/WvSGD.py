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

from sys import argv

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
        #print(col2)
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
    #   print("Usage: " + argv[0] + ' distortion_data sentence_vectors')
    #   exit(1)

    data = pd.read_csv('../data/self_label_distortion2.csv')
    #data = pd.read_csv(argv[1])
    data.columns
    print('loaded data')



    #load the word vector data
    print('is reading data...')
    infile = open('../sentiVectors2','rb')
    #infile = open(argv[2],'rb')
    results = pickle.load(infile)


    print('is creating feature matrix...')
    proto_matrix = append_features(results)
    X = np.matrix(proto_matrix)
    X = np.nan_to_num(X)

    y = getLabel(results)
    y = np.array(y)

    print('computing svm model...')
    #####grid search (the parameters predict everything to one class, we should use a separated 
    #sample for tuning parameters, but not enough cases so far)

    

    clf = make_pipeline(SGDClassifier(max_iter= 1000))
  
    parameters = [{'sgdclassifier__alpha': [0.01, 0.05, 0.001, 0.005], 'sgdclassifier__class_weight':['balanced'],
                  'sgdclassifier__loss': ['hinge','log','modified_huber','squared_hinge', 'perceptron'], 
                   'sgdclassifier__penalty':['none','l1','l2']}]
                       
    grid_search_item = GridSearchCV(clf,
                              param_grid = parameters,
                               scoring = 'accuracy',
                               cv = cv,
                               n_jobs = -1)
    grid_search = grid_search_item.fit(X, y)

    print(grid_search.best_score_)
    print(grid_search.best_params_)





