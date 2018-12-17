str1
str2

word_id = 0
word_idx = {}
for w in str2.split():
	if w not in word_idx:
		if w in str1:
			word_idx[w] = word_id
			word_id += 1




str3 = 'python a does not support'
b= np.zeros((6, 300))


a = np.zeros((6, 300))
for w in str2.split():
	if w in en_model.vocab:
		a[word_idx[w]] += en_model[w]




#for each sentence
#1) tokenize
#2) computer tf-idf for each word
#3) arr1 = ['i', 'like', 'pie']
#   arr2 = ['0.23', '0.32423, '0.842']
# for i in range(len(arr1)):
#	word = arr1[i]
#	score = arr2[i]
# 	if word in embebdding_vocab:
#		score_matrix[embedding_vocab[word]] += score*en_model[word]

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


from collections import Counter
import math
def __TF(doc):
	tfDict = {}
	length = len(doc.split())
	wordDict = Counter(preprocess(doc).split())
	for w, w_count in wordDict.items():
		#print(w_count)
		tfDict[w] = float(w_count)/length
	return tfDict

example1 = 'this is a a sample'
__TF(example1)
example2 = 'this is another another example example example'
__TF(example2)

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


