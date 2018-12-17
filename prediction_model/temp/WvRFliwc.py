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



#direct to ~/cognitive_distortion/prediction_model then run the script
#This is a grid search RF model with tfidf sentiVec and LIWC as features, feature selection 

##just word embeddings
print('computing RF (sentvec+tiidf+liwc) model...')
rf = make_pipeline(StandardScaler(), RandomForestClassifier())

parameters = [{'randomforestclassifier__max_features':['auto','sqrt','log2'], 'randomforestclassifier__class_weight':['balanced'], 
           'randomforestclassifier__max_leaf_nodes':[10,50,100], 'randomforestclassifier__max_depth':[2,5,10,20], 'randomforestclassifier__n_estimators' : [50,100,200,300,400]}]
               
grid_search_item = GridSearchCV(rf,
                      param_grid = parameters,
                       cv = cv,
                       scoring = 'accuracy',
                       n_jobs = -1)
grid_search = grid_search_item.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)

y_true, y_pred = y_test, grid_search.predict(X_test)
print(classification_report(y_true, y_pred))
  
0.6993865030674846
{'randomforestclassifier__class_weight': 'balanced', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__max_features': 'auto', 'randomforestclassifier__max_leaf_nodes': 50, 'randomforestclassifier__n_estimators': 300}
              precision    recall  f1-score   support

           1       0.58      0.21      0.31        99
           2       0.68      0.92      0.78       181

   micro avg       0.67      0.67      0.67       280
   macro avg       0.63      0.56      0.55       280
weighted avg       0.65      0.67      0.61       280







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
    
    y_true, y_pred = y_test, grid_search.predict(X_test)
    print(classification_report(y_true, y_pred))

    print('Done!')

    print('RF(sentvec+tfidf+liwc+feature selection) model Done!')

