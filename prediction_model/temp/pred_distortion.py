
# coding: utf-8

# In[91]:

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



# In[43]:

# Importing the dataset
dataset = pd.read_csv('status_dep_demog_big5_schwartz_swl_like.csv')


# In[44]:

dataset.columns


# In[45]:

data = pd.read_csv('self-labeled_id.csv')


# In[46]:

data.columns


# In[47]:

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
data1.columns.values[10] = 'emoticon'


# In[48]:

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


# In[49]:

data2


# In[50]:

####length of post as feature
word_len =[]
for i in data2['text']:
	length = len(str(i))
	word_len.append(length)

length_df = pd.DataFrame(word_len)

#data3 = pd.concat([data2,length_df], axis = 1)
#data3.columns.values[9] = 'post_leng'

data2['post_len'] = word_len

###merge text, id with all FB features
all_data = pd.merge(data2, dataset, on = 'userid', how = 'inner')

all_data.to_csv('view.csv')

# In[51]:

#####convert categorical data 
#select useful variables in dataset
selected = ['userid', 'marital_status', 'ethnicity', 'gender','age','relationship_status', 'network_size','negative_yn_self', 'negative_y', 'thoughtcat','text_y','emoticon','Positive','Negative','senti_score','post_len','distortion_yn', 'magnitude']
data_n = all_data.loc[:, (selected)]
data_dep = data_n

data_dep['ethnicity'] = data_dep['ethnicity'].fillna('Other')
data_dep['marital_status'] = data_dep['marital_status'].fillna('Other')
data_dep['age'] = data_dep['age'].fillna(data_dep['age'].mean())
data_dep['relationship_status'] = data_dep['relationship_status'].fillna(0)
data_dep['network_size'] = data_dep['network_size'].fillna(data_dep['network_size'].mean())


####one hot encoding
features_oneHot = ['marital_status', 'ethnicity', 'gender','relationship_status','emoticon','negative_yn_self']

x = data_dep[features_oneHot].values
y = data_dep['distortion_yn'].values
# Encoding categorical data

marital_status = pd.get_dummies(x[:, 0])
ethnicity = pd.get_dummies(x[:, 1])
gender = pd.get_dummies(x[:, 2])
relationship_status = pd.get_dummies(x[:, 3])
emoticon = pd.get_dummies(x[:, 4])
negative_yn_self = pd.get_dummies(x[:, 5])


fea = pd.concat([marital_status,ethnicity,gender,relationship_status,emoticon, negative_yn_self], axis =1).values



# In[53]:

features = ['age','network_size','Positive','Negative','senti_score'] 
x2 = data_dep[features].values

fb_fea = np.concatenate((fea, x2), axis=1)

fb_fea


# In[54]:

###text as feature
cv = CountVectorizer()
text_vec = cv.fit_transform(data_dep['text_y']).toarray()




####convert to tfidf
tfidf_transformer = TfidfTransformer()
text_vec = tfidf_transformer.fit_transform(text_vec).toarray()

####combine with all features
x = np.concatenate((text_vec, fb_fea), axis=1)


# In[57]:

####Bayesian classifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
cm


# In[58]:

np.mean(y_pred == y_test)


# In[59]:

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()


# In[60]:
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))


#0.573401721927
#0.560654685494
#0.640520446097

# In[64]:

####SVM 
svc_clf = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svc_clf.fit(x_train, y_train) 

y_pred = svc_clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)


# In[68]:

print(accuracy_score(y_test, y_pred))


# In[66]:

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))
print(cm)


# In[93]:

####random forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# x_train, y_train = make_classification(n_samples=1000, n_features=5651,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
weight= np.array([1 if i == 2 else 9 for i in y_train])
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(x_train, y_train, sample_weight= weight)

print(clf.feature_importances_)

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))
print(cm)


# In[94]:

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(x_train, y_train)


y_pred = neigh.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(cm)

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))


# In[70]:

####undersample after crossed validation so that no duplicated cases in train set and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

#Separate majority and minority classes

x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)
x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

xy_df = pd.concat([x_train_df, y_train_df], axis =1)
xy_df2 = pd.concat([x_test_df, y_test_df], axis =1)


xy_df.iloc[:,-1]
df_minority = xy_df[xy_df.iloc[:,-1]==1]   ##32
df_majority = xy_df[xy_df.iloc[:,-1]==2]   #629


 
# Upsample minority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=32,     # to match minority class
                                 random_state=123)
 
# Combine majority class with upsampled minority class
df_downsampled  = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
print(df_downsampled .iloc[:,-1].value_counts())

train_x2 = df_downsampled.iloc[:,:-1]
train_y2 = df_downsampled.iloc[:,-1]



# In[89]:

#######SVC oversample minority 
from sklearn.svm import SVC
clf = SVC(C=10, class_weight=None, coef0=0.0, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(train_x2, train_y2) 

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(cm)

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))


# In[85]:

#####grid search (the parameters predict everything to one class, we should use a separated 
#sample for tuning parameters, but not enough cases so far)
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


grid_search_item = GridSearchCV(estimator = clf,
                          param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search_item.fit(train_x2, train_y2)

print(grid_search.best_score_)####not sure how come this is so good,but the parameters don't work good in the model
print(grid_search.best_params_)


# In[83]:

# Naive Bayes oversample minority 
classifier = GaussianNB()
classifier.fit(train_x2, train_y2)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))
print(cm)


# In[95]:

####random forest oversample minority 
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_x2, train_y2)

#print(clf.feature_importances_)

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))
print(cm)


# In[97]:

###KNN oversample minority
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_x2, train_y2)


y_pred = neigh.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(cm)

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))



# In[103]:

#######feature selection
# Feature Extraction with PCA
from sklearn.decomposition import PCA
# load data
# feature extraction
pca = PCA(n_components=10)
fit = pca.fit(x_train)
# summarize components
print(fit.explained_variance_ratio_)
print(fit.components_)


# In[105]:

###feature importance 
from sklearn.ensemble import ExtraTreesClassifier
# load data
# feature extraction
model = ExtraTreesClassifier()
model.fit(x_train, y_train)
print(model.feature_importances_)


# In[135]:

#select non-zero features
fea_imp = model.feature_importances_
fea_list =[]
for i in range(len(fea_imp)):
	if fea_imp[i] != 0:
		fea_list.append((i, fea_imp[i]))


# In[138]:

fea_list


# In[166]:

###match the selected columns with training set 
train_fea = np.transpose(x_train)

fea_select =[]
for fea in range(len(train_fea)):
	for i in fea_list:
		if i[0] == fea:
			fea_select.append(train_fea[fea])

selected_fea = np.transpose(fea_select)
x_train_selected = selected_fea


# In[167]:

###match the selected columns with test set 
test_fea = np.transpose(x_test)

fea_select2 =[]
for fea in range(len(test_fea)):
	for i in fea_list:
		if i[0] == fea:
			fea_select2.append(test_fea[fea])

selected_fea2 = np.transpose(fea_select2)
x_test_selected = selected_fea2


# In[168]:

####Bayesian classifier
# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train_selected, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test_selected)


cm = confusion_matrix(y_test, y_pred)

print(cm)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))


# In[157]:

####SVM 
svc_clf = SVC(C=100, cache_size=20, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svc_clf.fit(x_train_selected, y_train) 

y_pred = svc_clf.predict(x_test_selected)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))



#[[ 14   1]
# [ 66 203]]
#0.764084507042
#0.57654389674
#0.585049019608
#0.843990086741



# In[169]:

####random forest
# x_train, y_train = make_classification(n_samples=1000, n_features=5651,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
weight= np.array([1 if i == 2 else 9 for i in y_train])
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(x_train_selected, y_train, sample_weight= weight)

print(clf.feature_importances_)

y_pred = clf.predict(x_test_selected)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))
print(cm)


# In[160]:

#####undersample 
###match  selected columns with training set 
trainx2 = train_x2.values
train_fea = np.transpose(trainx2)

fea_select =[]
for fea in range(len(train_fea)):
	for i in fea_list:
		if i[0] == fea:
			fea_select.append(train_fea[fea])

selected_fea = np.transpose(fea_select)
x_train_selected = selected_fea


# In[162]:

#######SVC oversample minority + selected features
svc_clf = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svc_clf.fit(x_train_selected, train_y2) 

y_pred = svc_clf.predict(x_test_selected)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

#[[ 54  42]
# [ 44 144]]
#0.697183098592
#0.663377253432
#0.662606978275
#0.664228723404



# In[164]:

####random forest oversample minority + selected features
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(x_train_selected, train_y2)

#print(clf.feature_importances_)

y_pred = clf.predict(x_test_selected)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))
print(cm)


# In[165]:

###KNN oversample minority + selected features
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train_selected, train_y2)


y_pred = neigh.predict(x_test_selected)

cm = confusion_matrix(y_test, y_pred)
print(np.mean(y_pred == y_test))
print(cm)

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))


# In[ ]:

#0.62323943662
#[[ 58  38]
# [ 69 119]]
#0.605017222331
#0.607327348413
#0.618572695035


