import pandas as pd
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
import datetime
import csv


#merge with CESD
path1 = '/home/lucia/phd_work/mypersonality_data/cognitive_distortion/data/important_data/'

def merge_cesd_valence(path_to_files, valence_file):
    '''read and merge cesd and valence'''
    cesd = pd.read_csv(path_to_files + 'adjustedCESD.csv')
    cesd_sum = cesd[['userid', 'cesd_sum']]
    valence_vec = pd.read_csv(path_to_files + valence_file)
    val_cesd = pd.merge(valence_vec, cesd_sum, how='left', on='userid')
    val_cesd = val_cesd.drop_duplicates(subset='userid', keep="first")

    return val_cesd


def recode(array):
    '''recode y, cesd > 23 as high depressive symptoms'''
    new = []
    for num in array:
        if num <= 23:
            new.append(0)
        if num > 23:
            new.append(1)
    return new


def get_train_test(val_cesd, days_for_model):
    '''split train test'''
    X = val_cesd.iloc[:, 1:days_for_model]  #here you can customize the days
    y = val_cesd["cesd_sum"]

    y_recode = recode(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_recode, test_size=0.30, random_state = 300)
    return X_train, X_test, y_train, y_test

def SVM_classifier(X_train, y_train, y_test, X_test): 
    '''train svm'''
    cv_fold = StratifiedKFold(n_splits=3, random_state=0)
    svc = make_pipeline(svm.SVC())
    parameters = [{'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                   'svc__gamma': [0.5, 0.1, 0.01, 0.001, 0.0001],
                   'svc__C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 10],
                   'svc__class_weight':['balanced']}]

    grid_search_item = GridSearchCV(svc,
                                    param_grid=parameters,
                                    cv=cv_fold,
                                    scoring='accuracy',
                                    n_jobs=-1)
    grid_search = grid_search_item.fit(X_train, y_train)
    y_true, y_pred = y_test, grid_search.predict(X_test)
    return y_true, y_pred, grid_search


def store_results(path_to_save, y_true, y_pred, grid_search, days_for_model):
    report = precision_recall_fscore_support(y_true, y_pred)
    precision, recall, fscore, support=precision_recall_fscore_support(y_true, y_pred, average='macro')

    y_pred_series = pd.DataFrame(y_pred)
    y_true_series = pd.DataFrame(y_true)
    result = pd.concat([y_true_series, y_pred_series], axis=1)
    result.columns = ['y_true', 'y_pred']
    result.to_csv(path_to_save + 'best_result.csv')

    f = open(path_to_save +'result.csv', 'a')
    writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
    writer_top.writerow(['classifer'] + ['best_scores'] + ['best_parameters'] + ['classification_report'] + ['marco_precision']+['marco_recall']+['marco_fscore']+['support']+['runtime']+['affect_days'])
    result_row = [[grid_search.estimator.steps[0][0], grid_search.best_score_, grid_search.best_params_, report, precision, recall, fscore, support, str(datetime.datetime.now()), days_for_model]]
    writer_top.writerows(result_row)
    f.close

def run_model(path_to_files, valence_file, days_for_model): 
    all_cases = merge_cesd_valence(path_to_files, valence_file)
    X_train, X_test, y_train, y_test = get_train_test(all_cases, days_for_model)
    y_true, y_pred, grid_search = SVM_classifier(X_train, y_train, y_test, X_test)
    store_results(path_to_files + 'results/', y_true, y_pred, grid_search, days_for_model)

run_model(path1, 'ValenceVec_Norm_Mix.csv', 31)