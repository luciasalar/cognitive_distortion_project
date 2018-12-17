from allennlp.modules.elmo import Elmo, batch_to_ids
import pandas as pd
import csv
from gensim.models import KeyedVectors
import numpy as np
import pickle
import re
import nltk
from sys import argv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

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
    #sent = re.sub(r'[^\w\s]','',sent)
    words = sent.split()
    new_words = []
    for w in words:     
        new_words.append(w.lower())
    return ' '.join(new_words)


def convert_Elmovec(objects):
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


#initate Elmo
options_file = "/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)
elmoEMB = ElmoEmbedder(options_file, weight_file)
tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
vectors = elmoEMB.embed_sentence(tokens)


objects = {}
with open('../data/self_label_distortion2.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['text'])
        texthash = hash(row['text'])
        if texthash not in objects:
            objects[texthash] = MyFea(row['text'])
        objects[texthash].label.append(row['negative_yn_self'])

#treat each doc as one sentence
count = 0
maxlen = 0
proto_matrix = []
for item in objects: 
	tokens = word_tokenize(objects[item].text)
	objects[item].vectors = elmoEMB.embed_sentence(tokens)
	for i in objects[item].vectors:
		print(len(i))
		length = len(i)
		if length > maxlen:
			maxlen = length
		#padding vectors 
	
	count += 1
	if count > 3:
		break


print("Maximum sentence length: ", maxlen)
	

	

	






fea = np.matrix(proto_matrix.reshape((3,28,1024)))

def append_features(ob):
    count = 0
    proto_matrix = []
    for item in ob:
        col2 = np.append(ob[item].vectors, ob[item].meanVec)
        #print(col2)
        proto_matrix.append(col2)
        count += 1                 
    return proto_matrix


#maxlen = 0
count = 0
for item in objects: 
    #text = preprocess(objects[item].text)
    #words = filter(lambda x: x in en_model.vocab, text.split())
	sent_tokens = sent_tokenize(objects[item].text)
	for sent in sent_tokens:
		tokens = word_tokenize(sent)
		objects[item].vectors.append([elmoEMB.embed_sentence(tokens)])
	print(objects[item].vectors[0])
	#meanVec = np.mean(objects[item].vectors)
	# #objects[item].meanVec.append()
	#print(meanVec)
	count += 1
	if count > 2:
		break

	#average sent token
	#print(length)
	# if length > maxlen:
	# 	maxlen = length 

# print("Maximum sentence length: ", maxlen)




# print("Maximum sentence length: ", maxlen)
		

		






	objects[item].vectors = [en_model[x] for x in words]
    objects[item].vectors = elmoEMB.embed_sentence(tokens)
    #aver = np.mean(np.array([en_model[x] for x in words]))
    #objects[item].meanVec.append(aver)
    length = len(objects[item].vectors)









