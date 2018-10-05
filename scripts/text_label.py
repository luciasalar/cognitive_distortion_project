import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import nltk

text_label = pd.read_csv('text_label.csv')

def preprocess(sent):

    words = str(sent).split()
    new_words = []
    # ps = PorterStemmer()
    
    for w in words:
        w = w.lower().replace("**bobsnewline**","")
        # remove non English word characters
        w =re.sub(r'http://.*','',w)
        w = re.sub(r'[^\x00-\x7F]+',' ', w)
        # remove puncutation 
        w = re.sub(r'[^\w\s]','',w)
        w = re.sub("\d+", "", w)
        ps = PorterStemmer()
        review = [ps.stem(word) for word in w if not word in set(stopwords.words('english'))]
        new_words.append(w)
        
    return ' '.join(new_words)

high_cesd_t = text_label.loc[text_label['CESD_sum'] > 24]
high_cesd_te = high_cesd_t.text.apply(preprocess)

low_cesd_t = text_label.loc[text_label['CESD_sum'] < 16]
low_cesd_te = low_cesd_t.text.apply(preprocess)

#word frequency
count_words = high_cesd_te[high_cesd_te.str.contains("your status") == False].str.split(expand=True).stack().value_counts()
count_words[50:100]


count_words2 = low_cesd_te[low_cesd_te.str.contains("your status") == False].str.split(expand=True).stack().value_counts()
count_words2[50:100]

##bigram
bigrams = nltk.bigrams(high_cesd_te[high_cesd_te.str.contains("your status") == False].str.split(expand=True).stack())
freq_bi = nltk.FreqDist(bigrams)
freq_bi.most_common(50)


bigrams2 = nltk.bigrams(low_cesd_te[low_cesd_te.str.contains("your status") == False].str.split(expand=True).stack())
freq_bi2 = nltk.FreqDist(bigrams2)
freq_bi2.most_common(50)

##ngram
trigrams = nltk.ngrams(high_cesd_te[high_cesd_te.str.contains("your status") == False].str.split(expand=True).stack(),3)
freq_tri = nltk.FreqDist(trigrams)
freq_tri.most_common(50)


trigrams2 = nltk.ngrams(low_cesd_te[low_cesd_te.str.contains("your status") == False].str.split(expand=True).stack(),3)
freq_tri2 = nltk.FreqDist(trigrams2)
freq_tri2.most_common(50)

#top result is fff, which refers to the F word in the text

def preprocess2(sent):  #retain punctuation 

    words = str(sent).split()
    new_words = []
    # ps = PorterStemmer()
    
    for w in words:
        w = w.lower().replace("**bobsnewline**","")
        # remove non English word characters
        w =re.sub(r'http://.*','',w)
        # remove puncutation 
        w = re.sub("\d+", "", w)
        ps = PorterStemmer()
        review = [ps.stem(word) for word in w if not word in set(stopwords.words('english'))]
        new_words.append(w)
        
    return ' '.join(new_words)


high_cesd_t = text_label.loc[text_label['CESD_sum'] > 24]
high_cesd_te = high_cesd_t.text.apply(preprocess2)

low_cesd_t = text_label.loc[text_label['CESD_sum'] < 16]
low_cesd_te = low_cesd_t.text.apply(preprocess2)

#word frequency
count_words = high_cesd_te.str.split(expand=True).stack().value_counts()
count_words[0:50]


count_words2 = low_cesd_te.str.split(expand=True).stack().value_counts()
count_words2[0:50]

##############################################################high transdiagnostic symptoms
high_trans_t = text_label.loc[text_label['negative_emo_score2'] > np.mean(text_label['negative_emo_score2'].values)]
high_trans_t2 = high_trans_t.loc[high_trans_t['distortion_score'] > np.mean(text_label['distortion_score'].values)]
high_trans_te = high_trans_t2.text.apply(preprocess2)

low_trans_t = text_label.loc[text_label['negative_emo_score2'] < np.mean(text_label['negative_emo_score2'].values)]
low_trans_t2 = low_trans_t.loc[low_trans_t['distortion_score'] < np.mean(text_label['distortion_score'].values)]
low_trans_te = low_trans_t2.text.apply(preprocess2)


#word frequency

count_words_t = high_trans_te[high_trans_te.str.contains("your status") == False].str.split(expand=True).stack().value_counts()
count_words_t[50:100]


count_words2_t = low_trans_te[low_trans_te.str.contains("your status") == False].str.split(expand=True).stack().value_counts()
count_words2_t[50:100]

##bigram
high_trans_te = high_trans_t2.text.apply(preprocess)
low_trans_te = low_trans_t2.text.apply(preprocess)

bigrams = nltk.bigrams(high_trans_te[high_trans_te.str.contains("your status") == False].str.split(expand=True).stack())
freq_bi = nltk.FreqDist(bigrams)
freq_bi.most_common(50)


bigrams2 = nltk.bigrams(low_trans_te[low_trans_te.str.contains("your status") == False].str.split(expand=True).stack())
freq_bi2 = nltk.FreqDist(bigrams2)
freq_bi2.most_common(50)

##ngram more positive terms in the low score group
trigrams = nltk.ngrams(high_trans_te[high_trans_te.str.contains("your status") == False].str.split(expand=True).stack(),3)
freq_tri = nltk.FreqDist(trigrams)
freq_tri.most_common(50)


trigrams2 = nltk.ngrams(low_trans_te[low_trans_te.str.contains("your status") == False].str.split(expand=True).stack(),3)
freq_tri2 = nltk.FreqDist(trigrams2)
freq_tri2.most_common(50)

