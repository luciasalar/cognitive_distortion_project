# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import resample
import emot
from sklearn.svm import SVC

# Importing the dataset
dataset = pd.read_csv('status_dep_demog_big5_schwartz_swl_like.csv')


###prediction 

data = pd.read_csv('text_id_1000.csv')   ##this file contain text and userid AND SENTIMENT in the sample

####add emoticon
emo_fea = []
for i in data['text']:
	emo = emot.emoticons(i)
	if len(emo) > 0:
		emo_fea.append(emo[0]['value'])
	else:
		emo_fea.append(False)

emo_df = pd.DataFrame(emo_fea)
frames = [data, emo_df]
data1 = pd.concat(frames, axis = 1)
data1.columns.values[7] = 'emoticon'


####for some reason concat doesn't work well in here, so I join the dataframe mannually 
#data1 = pd.read_csv('data1.csv')

#remove na in id
data2 = data1[pd.notnull(data1['userid'])]
data2['senti_score'] = data2['Positive'] + data2['Negative']
data2 = data2[pd.notnull(data2['userid'])]

###process data

def preprocess(sent):

    words = str(sent).split()
    new_words = []
    # ps = PorterStemmer()
    
    for w in words:
        w = w.lower().replace("**bobsnewline**","")
        # remove non English word characters
        w = re.sub(r'[^\x00-\x7F]+',' ', w)
        # remove puncutation 
        w = re.sub(r'[^\w\s]','',w)
        # w = ps.stem(w)
        new_words.append(w)
        
    return ' '.join(new_words)

data2['text'] = data2['text'].apply(preprocess)

####length of post as feature
word_len =[]
for i in data2['text']:
	length = len(i)
	word_len.append(length)

length_df = pd.DataFrame(word_len)

#data3 = pd.concat([data2,length_df], axis = 1)
#data3.columns.values[9] = 'post_leng'

data2['post_len'] = word_len

###merge text, id with all FB features
all_data = pd.merge(dataset, data2, on = 'userid', how = 'inner')



#####convert categorical data 
#select useful variables in dataset
selected = ['userid', 'marital_status', 'ethnicity', 'gender','age','relationship_status', 'network_size','negative_yn','thoughtcat','text_y','emoticon','Positive','Negative','senti_score','post_len']
data_n = all_data[selected]
data_dep = data_n

data_dep['ethnicity'] = data_dep['ethnicity'].fillna('Other')
data_dep['marital_status'] = data_dep['marital_status'].fillna('Other')
data_dep['age'] = data_dep['age'].fillna(data_dep['age'].mean())
data_dep['relationship_status'] = data_dep['relationship_status'].fillna(0)
data_dep['network_size'] = data_dep['network_size'].fillna(data_dep['network_size'].mean())


####one hot encoding
features_oneHot = ['marital_status', 'ethnicity', 'gender','relationship_status','emoticon']

x = data_dep[features_oneHot].values
y = data_dep['negative_yn'].values
# Encoding categorical data

marital_status = pd.get_dummies(x[:, 0])
ethnicity = pd.get_dummies(x[:, 1])
gender = pd.get_dummies(x[:, 2])
relationship_status = pd.get_dummies(x[:, 3])
emoticon = pd.get_dummies(x[:, 4])

fea = pd.concat([marital_status,ethnicity,gender,relationship_status,emoticon], axis =1).values



#fea_weka = fea = pd.concat([marital_status,ethnicity,gender,relationship_status,emoticon], axis =1)
#selected2 = ['age', 'network_size','negative_yn','thoughtcat','Positive','Negative','senti_score','post_len']
#data_2 = data_dep[selected2]

#data_2.to_csv('data_2.csv')


features = ['age','network_size','Positive','Negative','senti_score'] 
x2 = data_dep[features].values

fb_fea = np.concatenate((fea, x2), axis=1)


###text as feature
cv = CountVectorizer()
text_vec = cv.fit_transform(data_dep['text_y']).toarray()


#convert label to number  ####one hot encoder randomly assign yes/no to 1 or 2
#labelencoder = LabelEncoder()
#y = labelencoder.fit_transform(y.astype(str))
y[y == 'yes'] = 1
y[y == 'no'] = 2
y[y == 'mixed'] = 1 #replace mix with yes
y=y.astype('int')

####convert to tfidf
tfidf_transformer = TfidfTransformer()
text_vec = tfidf_transformer.fit_transform(text_vec).toarray()

####combine with all features
x = np.concatenate((text_vec, fb_fea), axis=1)

####convert time 
#t = data['time'].values.reshape((81,1))
#x = np.concatenate((x, t), axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)
#0.64436619718309862

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean() # 0.61982082594022903


print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

#0.526418600219
#0.52653907496
#0.526315789474
#array([[156,  52],
#       [ 53,  23]])



####SVM no oversample

svc_clf = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svc_clf.fit(x_train, y_train) 

y_pred = svc_clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)
cm

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

#0.52014160811
#0.536963036963
#0.546811740891


#array([[118,  90],
#       [ 36,  40]])

print(accuracy_score(y_test, y_pred))
#0.556338028169




from sklearn.metrics import *




#####SVM   address imbalanced class
#y2 = y.reshape((944,1))
#xy = np.concatenate((x, y2), axis=1)
#xy_df = pd.DataFrame(xy)


####oversample before crossed validation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

#Separate majority and minority classes

x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)
x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

xy_df = pd.concat([x_train_df, y_train_df], axis =1)
xy_df2 = pd.concat([x_test_df, y_test_df], axis =1)


xy_df.iloc[:,-1]
df_majority = xy_df[xy_df.iloc[:,-1]==1]   ##461
df_minority = xy_df[xy_df.iloc[:,-1]==2]   #199


 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=461,    # to match majority class
                                 random_state=12) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.iloc[:,-1].value_counts()

#2.0    461
#1.0    461
#Name: 5609, dtype: int64
train_x2 = df_upsampled.iloc[:,:-1]
train_y2 = df_upsampled.iloc[:,-1]

######## test set resample

xy_df2.iloc[:,-1]
df_majority2 = xy_df2[xy_df2.iloc[:,-1]==1]   ##461
df_minority2 = xy_df2[xy_df2.iloc[:,-1]==2]   #199


 
# Upsample minority class
df_minority_upsampled2 = resample(df_minority2, 
                                 replace=True,     # sample with replacement
                                 n_samples=284,    # to match majority class
                                 random_state=10) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled2 = pd.concat([df_majority2, df_minority_upsampled2])
 
# Display new class counts
df_upsampled2.iloc[:,-1].value_counts()

#2.0    284
#1.0    208
#Name: 5609, dtype: int64
test_x2 = df_upsampled2.iloc[:,:-1]
test_y2 = df_upsampled2.iloc[:,-1]



#######SVC
from sklearn.svm import SVC
clf = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(train_x2, train_y2) 

y_pred = clf.predict(test_x2)

cm = confusion_matrix(test_y2, y_pred)
np.mean(y_pred == test_y2)
cm

print(f1_score(test_y2,y_pred, average = 'macro'))
print(precision_score(test_y2,y_pred, average = 'macro'))
print(recall_score(test_y2,y_pred, average = 'macro'))

##
#0.522261016879
#0.532608513139
#0.532875135428



#####grid search 
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


grid_search_item = GridSearchCV(estimator = clf,
                          param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search_item.fit(x_train, y_train)

grid_search.best_score_   
#0.69999999999999996

grid_search.best_params_
#{'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}



#######do grid search 10 times
grid_searches = []
best_accuracy = []
best_parameters = []

for i in range(10):
	grid_search_item = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
	grid_search = grid_search_item.fit(x_train, y_train)
	print(type(grid_search))
	best_accuracy_item = grid_search.best_score_   
	best_parameters_item = grid_search.best_params_
	grid_searches.append(grid_search_item)
#	np.vstack(best_accuracy, best_accuracy_item)
	best_parameters.append(best_parameters_item)
	best_accuracy.append(best_accuracy_item)


###10 fold cross validation                        
accuracies = cross_val_score(estimator = clf, X = x, y = y, cv = 10)
accuracies.mean()
#


# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(train_x2, train_y2)

# Predicting the Test set results
y_pred = classifier.predict(test_x2)


cm = confusion_matrix(test_y2, y_pred)
np.mean(y_pred == test_y2)
#0.53658536585365857


print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))



from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# x_train, y_train = make_classification(n_samples=1000, n_features=5651,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
weight= np.array([2 if i == 2 else 1 for i in y_train])
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(x_train, y_train, sample_weight= weight)

print(clf.feature_importances_)

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)

#array([[177,  31],
#       [ 69,   7]])

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

import pickle
with open ('randomforest_weighted.pickle'.'wb') as f:
	pickle.dump(clf,f)

pickle_in =open('randomforest_weighted.pickle')

###oversampling

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_x2, train_y2)


print(clf.feature_importances_)

y_pred = clf.predict(test_x2)

cm = confusion_matrix(test_y2, y_pred)
np.mean(y_pred == test_y2)

#0.54268292682926833

#array([[120,  88],
#       [137, 147]])

print(f1_score(test_y2,y_pred, average = 'macro'))
print(precision_score(test_y2,y_pred, average = 'macro'))
print(recall_score(test_y2,y_pred, average = 'macro'))

#0.541301510349
#0.546228992466
#0.547264355363


########### feature selection 

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train, y_train)


y_pred = neigh.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)
cm

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

#0.553805869074
#0.572600955276
#0.552884615385


#array([[178,  30],
#       [ 57,  19]])



parameters = [{'n_neighbors': [2, 3, 4, 5,6,7,8,9,10]}]


grid_search_item = GridSearchCV(estimator = neigh,
                          param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search_item.fit(x_train, y_train)

grid_search.best_score_   
#0.69999999999999996

grid_search.best_params_

############feature selection

# feature extraction
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
# feature extraction
neigh = KNeighborsClassifier(n_neighbors=5)

rfe = RFE(neigh, 10)
fit = rfe.fit(x_train, y_train)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_











##merge with lda
#lda_500 = pd.read_csv('~/phd_work/crowdflower/lda_500/lda_500.csv')

#status_sample = pd.read_csv('~/phd_work/crowdflower/status_sample.csv')
#status_sample = pd.DataFrame(status_sample['text'])

#lda_500 = pd.concat([status_sample,lda_500], axis =1)

#all_fea_lda = pd.merge(data_dep, lda_500, left_on = 'text_y', right_on = 'text', how = 'left')

####select lda features and convert it to array
#lda2 = all_fea_lda.loc[21:500]


###feature importance 
from sklearn.ensemble import ExtraTreesClassifier
# load data
# feature extraction
model = ExtraTreesClassifier()
model.fit(x_train, y_train)
print(model.feature_importances_)
fea_imp = model.feature_importances_

fea_list =[]
for i in range(len(fea_imp)):
	if fea_imp[i] != 0:
		fea_list.append((i, fea_imp[i]))

#transpost the orginal matrix
fea_df = pd.DataFrame(x_train)
fea_df_t = fea_df.transpose()

