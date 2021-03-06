import pandas as pd
import csv
from gensim.models import KeyedVectors
import numpy as np
import pickle
import re

#this script unpack word vector features for the ML model

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
    


#append objects as festure matrix
def append_features(ob):
    count = 0
    proto_matrix = []
    for item in ob:
        col2 = np.append(ob[item].vectors, ob[item].meanVec)
        proto_matrix.append(col2)
        count += 1                 
    return proto_matrix


def getLabel(obj):
    labels = []
    for item in obj:
        labels.append(int(obj[item].label[0]))
    return labels


def unpackFeatureMatrix(object, dim_x,dim_y,dim_z):
    X = np.zeros((dim_x,dim_y*dim_z), dtype='float32')
    dim = 0
    for i in results:
      X[dim] = results[i].vectors.reshape(dim_y*dim_z) #because apparently 3D features is too much to ask for
      dim = dim + 1
    return X


def unpackFeatureMatrixSparse(object):
    X = np.zeros((932,5258*300), dtype='float32')
    dim = 0
    for i in results:
      X[dim] = results[i].vectors.todense().reshape(5258*300)#because apparently 3D features is too much to ask for
      dim = dim + 1
    return X


if __name__ == '__main__':
    #if len(argv) != 3:
    #   print("Usage: " + argv[0] + ' distortion_data sentence_vectors')
    #   exit(1)
    #load the word vector data





print('is reading data...')
infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/wikiVectorsBoWTfidfSparse','rb')
#infile = open('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cognitive_distortion/wordEmbeddings/wikiVectorsBoW2','rb')
#infile = open(argv[2],'rb')
results = pickle.load(infile)
infile.close()



print('is creating feature matrix...')
#proto_matrix = append_features(results)

dim_x = len(results.keys()) #932
dim_y = len(results[list(results.keys())[0]].vectors) #5258
dim_z = len(results[list(results.keys())[0]].vectors[0]) #300
print('X: {}, Y: {}, Z: {}'.format(dim_x, dim_y, dim_z))
X = unpackFeatureMatrix(results)


del(results)
gc.collect()


#if the data is sparse matrix, we need to use this to unpack it 
#reshape tfidf dim 
X = unpackFeatureMatrixSparse(results)

print('compute SVC with optimized parameters...')
svd_model = TruncatedSVD(n_components=500, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)

svd_matrix = svd_model.fit_transform(X)

del(X)
gc.collect()

#save SVD features
# outfile = open('./wordEmbeddings/WikiVecTfidfSVDFea','wb')
# pickle.dump(svd_matrix,outfile)
# outfile.close()
