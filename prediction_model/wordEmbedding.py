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

#this script generates wordembbeding features 
#you need to type the path of the pretrained model and the name of the object stored 
class MyFea:
    def __init__(self, text):
        self.text = text
        self.userid = hash(self.text)
        self.label = []
        self.vectors = []
        self.meanVec = []
        
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


# i -> 0
# me -> 1

def __TF(doc):
    tfDict = {}
    length = len(doc.split())
    wordDict = Counter(preprocess(doc).split())
    for w, w_count in wordDict.items():
        #print(w_count)
        tfDict[w] = float(w_count)/length
    return tfDict


#number of docs contain the terms
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

def __iDF(file):
    idfDict = {}
    doc_list = __getDocList(file)
    length = len(pd.read_csv(file))
    for w, doc_count in doc_list.items():
        #print(doc_count)
        idfDict[w] = math.log10(length/doc_count)
    return idfDict

#idf = __iDF('../data/self_label_distortion2.csv')
#idf = __iDF('./prediction_model/testfile.csv')
idfDict = __iDF('../data/self_label_distortion2.csv')
def tfidf(doc, idfDict):
    tfidf = {}
    tfDict = __TF(doc)
    for w, tf_count in tfDict.items():
        tfidf[w] = float(tf_count)*idfDict[w]
    return tfidf


def getFeatureArrayForSent(sentence_txt, word_idx, dim_x, dim_y, en_model):
    #dim_x = len(word_idx.keys())
    features = np.zeros((dim_x, dim_y))
    for word in sentence_txt.strip().split():
        if word in en_model.vocab:
            features[word_idx[word]] += en_model[word]
    #sparse_features = sparse.csr_matrix(features)
    sparse_features = features.astype('float32')
    return sparse_features


def convert_vec(objects, word_idx, dim_x, dim_y, en_model):
    maxlen = 0
    for item in objects: 
        text = preprocess(objects[item].text)
        #words = filter(lambda x: x in en_model.vocab, text.split())
        objects[item].vectors = getFeatureArrayForSent(text, word_idx, dim_x, dim_y, en_model)       
    print("Maximum sentence length: ", maxlen)


def getTfidfFeatureArray(doc, word_idx, dim_x, dim_y, en_model):
    tfidfDict = tfidf(doc, idfDict)
    score_matrix = np.zeros((dim_x,dim_y), dtype='float32')
    for word, val in tfidfDict.items():
        if word in en_model.vocab:
            score_matrix[word_idx[word]] += val*en_model[word]
            #score_matrix = score_matrix.reshape(dim_x*dim_y)
    #sparse_features = sparse.csr_matrix(score_matrix)
    convert_32 = score_matrix.astype('float32') #nolonger necessary
    return convert_32


def convert_vecTfidf(objects, word_idx, dim_x, dim_y, en_model):
    maxlen = 0
    for item in objects: 
        text = preprocess(objects[item].text)
        #words = filter(lambda x: x in en_model.vocab, text.split())
        objects[item].vectors = getTfidfFeatureArray(text, word_idx, dim_x, dim_y, en_model)  
            
    print("Maximum sentence length: ", maxlen)
      

if __name__ == '__main__':
if len(argv) != 3:
	print("Usage: " + argv[0] + 'language model')
	exit(1)

objects = {}
with open('../data/self_label_distortion2.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['text'])
        texthash = hash(row['text'])
        if texthash not in objects:
            objects[texthash] = MyFea(row['text'])
        objects[texthash].label.append(row['negative_yn_self'])


print('start to load language model..')
#sentivec 
#en_model = KeyedVectors.load_word2vec_format('/afs/inf.ed.ac.uk/user/s16/s1690903/share/fasttext/wikipedia.300d.txt')
#type language model
embeddingModel = argv[1]
en_model = KeyedVectors.load_word2vec_format(embeddingModel)
print('Finish loading model')


word_idx = get_vocab('../data/self_label_distortion2.csv', en_model)
dim_x = len(word_idx.keys())
dim_y = en_model['i'].shape[0]
convert_vec(objects, word_idx, dim_x, dim_y, en_model)
print('converted text to vectors')


#dump object to pickles
OutputFilename = (argv[2])
#OutputFilename = 'wikiVectors'
outfile = open('./wordEmbeddings/sentiVectorsBoW','wb')
pickle.dump(objects,outfile)
outfile.close()

###get tfidf version 
word_idx = get_vocab('../data/self_label_distortion2.csv', en_model)
idfDict = __iDF('../data/self_label_distortion2.csv')
dim_x = len(word_idx.keys())
dim_y = en_model['i'].shape[0]
convert_vecTfidf(objects, word_idx, dim_x, dim_y, en_model)
print('converted text to tfidf vectors')
OutputFilename = (argv[2])
#OutputFilename = 'wikiVectors'
outfile = open('./wordEmbeddings/sentiVectorsBoWTfidf','wb')
pickle.dump(objects,outfile)
outfile.close()




for item in objects:
    print(objects[item].vectors)
