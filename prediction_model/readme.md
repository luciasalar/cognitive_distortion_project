# Classifiers

This folder contains prediction models on valence 
/temp contains models that do not work well (prediction everything to 1 class)

Results are documented in the spreadsheet here
https://docs.google.com/spreadsheets/d/1QaSP3zI8PlSRT4w6lADviymHq9FyXwh1wtrWD0AHg78/edit#gid=0

The features used in the models contain word embeddings (fasttext, sentiVec) and LIWC
Models: SVM, SGD (best performance)


#Files
* wordEmbedding.py: wordembedding features from pretrained word vectors. Word embeddings are multipled with word count to serve as bag-of-words features in ordered to be used in the traditional ML models. These features are stored as condensed or sparse matrix. 
* unpackFea.py: Unpack feature matrix and conduct SVD truncate to reduce the dimensions of wordEmbeddings to 500.
* WvSVM.py: SVM model with word embeddings and LIWC
* WvSGD.py: SGD model with word embeddings and LIWC
* CountVec.ipynb: Bag-of-words models with or without LIWC


#Models
In the model, word embeddings are multipled with word count to serve as bag-of-words features. Then we reduce the high dimentional feature with SVDtruncate, later on LIWC were added to the feature matrix. Standard scaler is applied after LIWC is added. The data is splitted into train and test, we use the train set to do a grid search(5-fold). SMOTE is added to one of the trainining models in the pipeline. Model with SMOTE over sampling has increased slightly. 

The models are trained on 1000 cases annotated by myself, the cases only contains positve or negaive classes.
* class 1: negative
* class 2: positive 

Below is the best model so far

fasttext vectors + tfidf + liwc (RF)

Best scores and best parameters
computing RF (sentvec+tiidf+liwc) model...
0.7414163090128756
{'randomforestclassifier__class_weight': 'balanced', 'randomforestclassifier__max_depth': 10, 'randomforestclassifier__max_features': 'auto', 'randomforestclassifier__max_leaf_nodes': 10, 'randomforestclassifier__n_estimators': 50}
              precision    recall  f1-score   support

           1       0.85      0.83      0.84        99
           2       0.91      0.92      0.91       181

   micro avg       0.89      0.89      0.89       280
   macro avg       0.88      0.87      0.87       280
weighted avg       0.89      0.89      0.89       280


In the model, word embeddings are multipled with word count to serve as bag-of-words features. Then we reduce the high dimensional feature with SVDtruncate, later on LIWC were added to the feature matrix. Standard scaler is applied after LIWC is added. 

Note from Maria: LIWC is starting to be questioned. Good for a baseline model, but we can't just assume that it's the standard. 

* Answer: LIWC has been questioned all the time, it's not perfect, but they are very good features. The baseline models are usually simple BOW model. POS tags, name entities, emoji and event vow counts are often used in sentiment classification as well.

The data is splitted into train and test, we use the train set to do a grid search (5-fold). 

SMOTE is added to some of the training models in the pipeline. However, the performances of SMOTE varies, it mainly increase recall of class 1 but reduce the precision in both classes.

Q from Maria: what is the reason for using SMOTE? Best to spell it out. 

* Answer: The two classes are unevenly distributed, with the second class almost doubled of the first class.
* SMOTE: Synthetic Minority Over-sampling Technique 

Q from Maria: increased what? 

* Model with SMOTE over sampling has increased the f1 score for both classes slightly.

Below is the best model so far

Q from Maria: What is class 1, what is class 2? 
* class 1: negative
* class 2: positive 

*NOTE: need to further test it on Sentivec and using tfidf on word count, need a bag-of-words model as baseline*
*try majority baseline*


# BOW baseline model
Valence_prediction.ipython: bag-of-words model 

Best BOW model:
countvec + tfidf + LIWC  + demographic (feature selection with ...) (SVM)

[0.73862186 0.6629387  0.74119275 0.7429567  0.74914068]
f1: 0.73 (+/- 0.06)
0.5716852063884027
0.7004230565838181


#plans: need to try sentiVec and ensemble models
=======
*NOTE: need to try SGD*

# References

SMOTE: https://doi.org/10.1613/jair.953


