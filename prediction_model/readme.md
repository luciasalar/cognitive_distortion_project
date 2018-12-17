#Classifiers

This folder contains prediction models on valence 
/temp contains models that do not work well (prediction everything to 1 class)

Results are documented in the spreadsheet here
https://docs.google.com/spreadsheets/d/1QaSP3zI8PlSRT4w6lADviymHq9FyXwh1wtrWD0AHg78/edit#gid=0

The features used in the models contain word embeddings(fasttext, sentiVec) and LIWC
Models: SVM, SGD(best performance)


#Files
* wordEmbedding.py: wordembedding features from pretrained word vectors. Word embeddings are multipled with word count to serve as bag-of-words features in ordered to be used in the traditional ML models. These features are stored as condensed or sparse matrix. 
* unpackFea.py: Unpack feature matrix and conduct SVD truncate to reduce the dimensions of wordEmbeddings to 500.
* WvSVM.py: SVM model with word embeddings and LIWC
* WvSGD.py: SGD model with word embeddings and LIWC
* CountVec.ipynb: Bag-of-words models with or without LIWC



#Models
In the model, word embeddings are multipled with word count to serve as bag-of-words features. Then we reduce the high dimentional feature with SVDtruncate, later on LIWC were added to the feature matrix. Standard scaler is applied after LIWC is added. The data is splitted into train and test, we use the train set to do a grid search(5-fold). SMOTE is added to one of the trainining models in the pipeline. Model with SMOTE over sampling has increased slightly. 

Below is the best model so far

fasttext vectors + tfidf + liwc (SGD)
Best scores and best parameters
0.7254601226993865
{'sgdclassifier__alpha': 0.05, 'sgdclassifier__class_weight': 'balanced', 'sgdclassifier__loss': 'hinge', 'sgdclassifier__penalty': 'l1'}
              precision    recall  f1-score   support

           1       0.60      0.62      0.61        99
           2       0.79      0.78      0.78       181

   micro avg       0.72      0.72      0.72       280
   macro avg       0.70      0.70      0.70       280
weighted avg       0.72      0.72      0.72       280    



# BOW baseline model
Valence_prediction.ipython: bag-of-words model 

Best BOW model:
countvec + tfidf + LIWC  + demographic (feature selection with ...) (SVM)

[0.73862186 0.6629387  0.74119275 0.7429567  0.74914068]
f1: 0.73 (+/- 0.06)
0.5716852063884027
0.7004230565838181

#plans: need to try sentiVec and ensemble models


