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
import gc
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

#direct to ~/cognitive_distortion/prediction_model then run the script
#This is a grid search RF model with tfidf sentiVec and LIWC as features, feature selection 
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


def getLabel(obj):
    labels = []
    for item in obj:
        labels.append(int(obj[item].label[0]))
    return labels

def RFClassifier(X_train,y_train, X_test, y_test):
  rf = make_pipeline(StandardScaler(), RandomForestClassifier())

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


def RFClassifierSmote(X_train,y_train, X_test, y_test):
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


if __name__ == '__main__':
  if len(argv) != 2:
    print("Usage: " + argv[0] + ' distortion_data sentence_vectors')
    exit(1)

#we can skip the previous step and load the SVD feature matrix directly 
  #infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/WikiVecTfidfSVDFea','rb')
  infile = open(argv[1],'rb')
  X_vec = pickle.load(infile)
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

  print('wordEmbedding features')
  X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.30, random_state=30)
  #just Wordvec
  smote_enn = SMOTEENN(random_state=42)
  cv = StratifiedKFold(n_splits=5, random_state = 0)

  print('wordEmbedding features')
  RFClassifier(X_train, y_train, X_test, y_test)
  RFClassifierSmote(X_train, y_train, X_test, y_test)


  ####combine with liwc
  text_liwc = pd.read_csv('../data/LIWC_self_label_valence.csv')
  liwc = text_liwc.loc[:,'function':'OtherP'].values
  X = np.concatenate((X_vec, liwc), axis=1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
  #Normalize data, convert it to unit vectors

  print('with liwc features')
  RFClassifier(X_train, y_train, X_test, y_test)
  RFClassifierSmote(X_train, y_train, X_test, y_test)

