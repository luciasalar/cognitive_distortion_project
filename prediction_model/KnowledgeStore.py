import re
import pandas as pd
import csv
from gensim.models import KeyedVectors
import numpy as np
import pickle
import re
import nltk
from sys import argv
from scipy import sparse
from collections import Counter
import math


def get_vocab(inputfile, en_model):
    word_id = 0
    word_idx = {}
    with open(inputfile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sent = preprocess(row['text']).split()
            for word in sent:
                if word not in word_idx:
                    if word in en_model.vocab:
                        word_idx[word] = word_id
                        word_id = word_id + 1
    return word_idx

def __getDocList(file):
	doc_list= {}

	with open(file) as csvf:
	    reader = csv.DictReader(csvf)
	    for line in reader:
	        line = preprocess(line['text'].strip())
	        line = set(line.split())
	        for w in line:
	            if w not in doc_list:
	                doc_list[w] = 1
	                #doc_count += 1
	            else:
	                doc_list[w] = doc_list[w] + 1
	return doc_list

def preprocess(sent):
    #remove punctustion\n",
    sent = re.sub(r'[^\w\s]','',sent)
    words = sent.split()
    new_words = []
    for w in words:     
        new_words.append(w.lower())
    return ' '.join(new_words)

def __TF(doc):
    tfDict = {}
    length = len(doc.split())
    wordDict = Counter(preprocess(doc).split())
    for w, w_count in wordDict.items():
        #print(w_count)
        tfDict[w] = float(w_count)/length
    return tfDict
#number of docs contain the terms


def __iDF(file):
    idfDict = {}
    doc_list = __getDocList(file)
    length = len(pd.read_csv(file))
    for w, doc_count in doc_list.items():
        #print(doc_count)
        idfDict[w] = math.log10(length/doc_count)
    return idfDict

idfDict = __iDF('../data/self_label_distortion2.csv')
def tfidf(doc, idfDict):
    tfidf = {}
    tfDict = __TF(doc)
    for w, tf_count in tfDict.items():
        tfidf[w] = float(tf_count)*idfDict[w]
    return tfidf

class MyFea:
    def __init__(self, text):
        self.text = text
        self.userid = hash(self.text)
        self.label = []
    def __hash__(self):
        return self.quoteID

def preprocess(sent):
    #remove punctustion\n",
    sent = re.sub(r'[^\w\s]','',sent)
    words = sent.split()
    new_words = []
    for w in words:     
        new_words.append(w.lower())
    return ' '.join(new_words)  


def __getPositiveDict(PISobjects):
    positiveDict = {}
    for item in PISobjects:
        line = preprocess(PISobjects[item].text)
        if int(PISobjects[item].label[0]) == 1:
            for w in line.split():
                if w not in positiveDict:
                    positiveDict[w] = 1
                else:
                    positiveDict[w] += 1 
    return positiveDict

def __getNegativeDict(PISobjects):
    NegativeDict = {}
    for item in PISobjects:
        line = preprocess(PISobjects[item].text)
        if int(PISobjects[item].label[0]) == 1:
            for w in line.split():
                if w not in negativeDict:
                    negativeDict[w] = 1
                else:
                    negativeDict[w] += 1
    return NegativeDict

def getTfidfKnowledgeStorePos(doc, idfDict, word_idx, dim_x, en_model):
    positiveDict = __getPositiveDict(PISobjects)
    tfidfDict = tfidf(doc, idfDict)
    PositiveKnowledge = np.zeros((dim_x,1), dtype='float32')
    for word, val in tfidfDict.items():
        if word in positiveDict.keys() and en_model.vocab:
              PositiveKnowledge[positiveDict[word]] += val*tfidfDict[word]
    return PositiveKnowledge

b = getTfidfKnowledgeStorePos(doc, idfDict, word_idx, dim_x, en_model)
#this array is the same length as the feature matrix
#word -> tfidf score
def getTfidfKnowledgeStoreNeg(doc, idfDict, word_idx, dim_x, en_model):
    negativeDict = __getNegativeDict(PISobjects)
    tfidfDict = tfidf(doc, idfDict)
    NegativeKnowledge = np.zeros((dim_x,1), dtype='float32')
    for word, val in tfidfDict.items():
        if word in negativeDict.keys() and en_model.vocab:
            NegativeKnowledge[negativeDict[word]] += val*tfidfDict[word]
    return NegativeKnowledge

en_model = KeyedVectors.load_word2vec_format('/afs/inf.ed.ac.uk/user/s16/s1690903/share/fasttext/wikipedia.300d.txt')
#type language model
embeddingModel = argv[1]
#en_model = KeyedVectors.load_word2vec_format(embeddingModel)
print('Finish loading model')

word_idx = get_vocab('../data/self_label_distortion2.csv', en_model)
dim_x = len(word_idx.keys())
idfDict = __iDF('../data/self_label_distortion2.csv')
dim_y = en_model['i'].shape[0]

#this is the knowlwdge base object
PISobjects = {}
with open('../data/self_label_distortion2.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['text'])
        texthash = hash(row['text'])
        if texthash not in PISobjects:
            PISobjects[texthash] = MyFea(row['text'])
        PISobjects[texthash].label.append(row['negative_yn_self'])



k = KnowledgeStorePos(PISobjects, idfDict, word_idx, dim_x, en_model)














