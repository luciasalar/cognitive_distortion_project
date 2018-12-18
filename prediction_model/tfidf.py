

from collections import Counter
import math

def getTfidfFeatureArray(doc):
	tfidfDict = tfidfDict(doc)
	for i in range(len(tfidfDict)):
		#create empty matrix
		dim_x = len(tfidfDict.keys())
		dim_y = en_model['i'].shape[0]
		score_matrix = np.zeros((dim_x,dim_y))

		word = list(tfidfDict.values())[i]
		score = list(tfidfDict.values())[i]
		if word in en_model.vocab:
			score_matrix[en_model[word]] += score*en_model[word]
	return score_matrix



def __TF(doc):
	tfDict = {}
	length = len(doc.split())
	wordDict = Counter(preprocess(doc).split())
	for w, w_count in wordDict.items():
		#print(w_count)
		tfDict[w] = float(w_count)/length
	return tfDict

# example1 = 'this is a a sample'
# __TF(example1)
# example2 = 'this is another another example example example'
# __TF(example2)

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

def tfidfDict(doc,file):
	tfidf = {}
	idfDict = __iDF(file)
	tfDict = __TF(doc)
	for w, tf_count in tfDict.items():
		tfidf[w] = float(tf_count)*idfDict[w]
	return tfidf

example3 = 'i like'
d = tfidfDict(example3, '../data/self_label_distortion2.csv')


