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

#direct to ~/cognitive_distortion/prediction_model then run the script
#This is a grid search RF model with tfidf sentiVec and LIWC as features, feature selection 
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
    wordEmbeddingModel = argv[1]
    infile = open(wordEmbeddingModel,'rb')
    results = pickle.load(infile)
    #'../wordEmbeddings/wikiVectors'


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

    #Normalize data, convert it to unit vectors
    cv = StratifiedKFold(n_splits=5, random_state = 0)
    print('computing RF (sentvec+tiidf+liwc) model...')
    rf = make_pipeline(StandardScaler(),RandomForestClassifier())

    parameters = [{'randomforestclassifier__max_features':['auto','sqrt','log2'], 'randomforestclassifier__class_weight':['balanced'], 
                   'randomforestclassifier__max_leaf_nodes':[10,50,100], 'randomforestclassifier__max_depth':[2,5,10,20], 'randomforestclassifier__n_estimators' : [50,100,200,300,400]}]
                       
    grid_search_item = GridSearchCV(rf,
                              param_grid = parameters,
                               cv = cv,
                               scoring = 'accuracy',
                               n_jobs = -1)
    grid_search = grid_search_item.fit(X, y)

    print(grid_search.best_score_)
    print(grid_search.best_params_)

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print('RF(sentvec+tfidf+liwc+feature) model Done!')

