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


#we can skip the previous step and load the SVD feature matrix directly 
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/WikiVecTfidfSVDFea','rb')
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

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.30, random_state=30)
#just Wordvec
smote_enn = SMOTEENN(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state = 0)
rf = make_pipeline(smote_enn,StandardScaler(), RandomForestClassifier())

parameters = [{'randomforestclassifier__max_features':['auto','sqrt','log2'], 'randomforestclassifier__class_weight':['balanced'], 
           'randomforestclassifier__max_leaf_nodes':[10,50,100], 'randomforestclassifier__max_depth':[2,5,10,20], 'randomforestclassifier__n_estimators' : [50,100,200,300,400]}]
           
grid_search_item = GridSearchCV(rf,
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

#tfidf  word vec with scaler
# 0.6993865030674846
# {'randomforestclassifier__class_weight': 'balanced', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__max_features': 'auto', 'randomforestclassifier__max_leaf_nodes': 50, 'randomforestclassifier__n_estimators': 300}
#               precision    recall  f1-score   support

#            1       0.58      0.21      0.31        99
#            2       0.68      0.92      0.78       181

#    micro avg       0.67      0.67      0.67       280
#    macro avg       0.63      0.56      0.55       280
# weighted avg       0.65      0.67      0.61       280

#tfidf word vec without scaler
# Best scores and best parameters
# 0.6993865030674846
# {'randomforestclassifier__class_weight': 'balanced', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__max_features': 'auto', 'randomforestclassifier__max_leaf_nodes': 50, 'randomforestclassifier__n_estimators': 400}
#               precision    recall  f1-score   support

#            1       0.63      0.19      0.29        99
#            2       0.68      0.94      0.79       181

#    micro avg       0.68      0.68      0.68       280
#    macro avg       0.66      0.57      0.54       280
# weighted avg       0.66      0.68      0.61       280

#tfidf word vec with scaler and SMOTE

# Best scores and best parameters
# 0.4386503067484663
# {'randomforestclassifier__class_weight': 'balanced', 'randomforestclassifier__max_depth': 2, 'randomforestclassifier__max_features': 'auto', 'randomforestclassifier__max_leaf_nodes': 50, 'randomforestclassifier__n_estimators': 100}
#               precision    recall  f1-score   support

#            1       0.41      0.78      0.53        99
#            2       0.76      0.38      0.51       181

#    micro avg       0.52      0.52      0.52       280
#    macro avg       0.58      0.58      0.52       280
# weighted avg       0.63      0.52      0.52       280


####combine with liwc
text_liwc = pd.read_csv('./data/LIWC_self_label_valence.csv')
liwc = text_liwc.loc[:,'function':'OtherP'].values
X = np.concatenate((x_vec, liwc), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
#Normalize data, convert it to unit vectors
cv = StratifiedKFold(n_splits=3, random_state = 0)
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

y_true, y_pred = y_test, grid_search.predict(X_test)
print(classification_report(y_true, y_pred))

print('Done!')

#tfidf word embeddings + liwc + scaler

# computing RF (sentvec+tiidf+liwc) model...
# 0.7414163090128756
# {'randomforestclassifier__class_weight': 'balanced', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__max_features': 'auto', 'randomforestclassifier__max_leaf_nodes': 10, 'randomforestclassifier__n_estimators': 50}
#               precision    recall  f1-score   support

#            1       0.85      0.83      0.84        99
#            2       0.91      0.92      0.91       181

#    micro avg       0.89      0.89      0.89       280
#    macro avg       0.88      0.87      0.87       280
# weighted avg       0.89      0.89      0.89       280



#with smote
# computing RF (sentvec+tiidf+liwc) model...
# 0.5772532188841202
# {'randomforestclassifier__class_weight': 'balanced', 'randomforestclassifier__max_depth': 20, 'randomforestclassifier__max_features': 'sqrt', 'randomforestclassifier__max_leaf_nodes': 10, 'randomforestclassifier__n_estimators': 50}
#               precision    recall  f1-score   support

#            1       0.50      0.87      0.63        99
#            2       0.88      0.52      0.65       181

#    micro avg       0.64      0.64      0.64       280
#    macro avg       0.69      0.69      0.64       280
# weighted avg       0.74      0.64      0.65       280


print('RF(sentvec+tfidf+liwc+feature selection) model Done!')

