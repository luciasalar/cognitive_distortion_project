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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sys import argv

#direct to ~/cognitive_distortion/prediction_model then run the script
#This is a grid search SVC model with tfidf word vectors and LIWC as features
# Importing the dataset
data = pd.read_csv('../data/self_label_distortion2.csv')
data.columns
print('loaded data')

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
    if len(argv) != 2:
        print("Usage: " + argv[0] + 'language model')
        exit(1)

    #load the word vector data
    print('is reading data...')
    #'../wordEmbeddings/wikiVectors'
    #'../wordEmbeddings/sentiVectors2'
    wordEmbeddingModel = argv[1]
    infile = open(wordEmbeddingModel,'rb')
    results = pickle.load(infile)

    print('is creating feature matrix...')
    proto_matrix = append_features(results)
    fea = np.matrix(proto_matrix)
    fea = np.nan_to_num(fea)

    y = getLabel(results)
    y = np.array(y)

    print('tifidf word vectors')
    tfidf_transformer = TfidfTransformer()
    X_vec = tfidf_transformer.fit_transform(fea).toarray()


    print('load LIWC data...')
    text_liwc = pd.read_csv('../data/LIWC_self_label_valence.csv')
    liwc = text_liwc.loc[:,'function':'OtherP'].values

    ####combine with liwc
    X = np.concatenate((X_vec, liwc), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==2)))

    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

    print('compute SVC with optimized parameters...')
    #####grid search (the parameters predict everything to one class, we should use a separated 

    cv = StratifiedKFold(n_splits=5, random_state = 0)
    svc = make_pipeline(StandardScaler(),svm.SVC())
    parameters = [{'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'svc__gamma': [0.01, 0.001, 0.0001],
                         'svc__C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 10] , 'svc__class_weight':['balanced']}]
                       
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



