import pandas as pd
import csv
from gensim.models import KeyedVectors
import numpy as np
import pickle
import re
import nltk
from sys import argv

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


def convert_vec(objects):
    maxlen = 0
    for item in objects: 
        text = preprocess(objects[item].text)
        words = filter(lambda x: x in en_model.vocab, text.split())
        objects[item].vectors = [en_model[x] for x in words]
        aver = np.mean(np.array([en_model[x] for x in words]))
        objects[item].meanVec.append(aver)
        length = len(objects[item].vectors)
        if length > maxlen:
            maxlen = length
            
    print("Maximum sentence length: ", maxlen)
    
#padding vectors 
    zero_vector = list(np.zeros(300))
    for item in objects:
        while len(objects[item].vectors) < maxlen:
            objects[item].vectors.append(zero_vector)

        

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
    #en_model = KeyedVectors.load_word2vec_format('/afs/inf.ed.ac.uk/user/s16/s1690903/share/fasttext/wiki.en.vec')
    #type language model
    embeddingModel = argv[1]
    en_model = KeyedVectors.load_word2vec_format(embeddingModel)
    print('Finish loading model')

    convert_vec(objects)
    print('converted text to vectors')


    #dump object to pickles
    OutputFilename = (argv[2])
    #OutputFilename = 'wikiVectors'
    outfile = open(OutputFilename,'wb')
    pickle.dump(objects,outfile)
    outfile.close()




