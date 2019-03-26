import pandas as pd
import numpy as np
import statsmodels.api as sm
import csv
from datetime import datetime

def getTransitions(LyricObject):
    emptyTran = 0
    lyricsTran = 0
    lyricsEmpty = 0
    preValence = 0
    for valence in LyricObject:
    #these are self transition states
        if valence == 0 and preValence == 0:
            emptyTran = emptyTran + 1
        elif valence == 1 and preValence == 1:
            lyricsTran = lyricsTran + 1
        elif valence == 0 and preValence == 1:
            lyricsEmpty = lyricsEmpty + 1
        elif valence == 1 and preValence == 0:
            lyricsEmpty = lyricsEmpty + 1
            
        preValence = valence
    return [emptyTran, lyricsTran, lyricsEmpty]

def getLyricsVector(df):
    valenceVec = {}
    valences =[]
    preUser = None
    preDay = None
    preValence = []

    for valence, day, user in zip(df['label'], df['time_diff'], df['userid']):
        if preUser is None:
            if day < 60: #if first day is not 60 
                addDays = 60 - day
                for num in range(0, addDays):
                    valences.append(0)
                valenceVec[user] = valences
            else:
                valences = [valence]
                valenceVec[user] = valences

        elif user == preUser and preDay == day+1:
            valences.append(valence)
            
        elif user == preUser and preDay != day+1:
            valences.append(valence)
            addDays = preDay - (day+1)
            for num in range(0, addDays):
                valences.append(0)
                
        elif user != preUser:
            if day < 60: #if first day is not 60 
                addDays = 60 - day
               # print(user,addDays)
                valences = [valence]
                for num in range(0, addDays):
                    valences.insert(0, 0)
                valenceVec[user] = valences
                #print(valences)
            else:
                valences = [valence]
                valenceVec[user] = valences
                #print(valences)
                
            if preDay > 1:
                addDays = preDay
                for num in range(0, addDays):
                    preValence.append(0)
                    
        preUser = user
        preDay = day
        preValence = valences
#     valenceVec[preUser] = valences
    return valenceVec

def countLyrics(valenceVec,val):
    count = 0
    mylist = []
    for userid in valenceVec:
        mylist.append(valenceVec[userid].count(val))
    return mylist

def getUserTransitions(valencVec):
    result = {}
    for item in valencVec:
        result[item] = getTransitions(valencVec[item])
#         print(result)
    return result

def getStats(valenceVec, saveStatsPath):
	empty = countLyrics(valenceVec,0)
	lyrics = countLyrics(valenceVec,1)
	df = pd.DataFrame(np.array(empty).reshape(77,1), columns=['emptyDays'])
	df['lyrics'] = lyrics
	df['userid'] = valenceVec.keys()
	dfStats = df.describe()
	dfStats.to_csv(saveStatsPath)
	return df

def saveTransition(filePath, TransitionStates):
	with open(filePath,'w') as csv_file:
	    writer = csv.writer(csv_file)
	   # writer.writerow(i for i in header)
	    writer.writerow(TransitionStates.keys())
	    for row in zip(*TransitionStates.values()):
	        writer.writerow(list(row))

def TransformProb(inputFile):
	file = pd.read_csv(inputFile)
	file = file.transpose()
	file.columns = ['emptyTrans','lyricsTran', 'lyricsEmpty']
	file['allTrans'] = file.sum(axis=1) 
	Tranprob = file.apply(lambda x: x/file.iloc[:,-1])
	Tranprob['userid'] = Tranprob.index
	return Tranprob

def getCorMatrix(frequency, Tranprob, pathToMerge, pathToTranMatrix, pathToFreMatrix):
	allData = pd.read_csv( path + '/data/important_data/user_scale_post_time2.csv')
	Var = allData[['userid','ope','con','ext','agr','neu','swl','CESD_sum']]
	compare = pd.merge(Tranprob, Var, on ='userid', how = 'inner')
	#compare
	compare.to_csv(pathToMerge)
	corMatrix = compare.corr()
	corMatrix.to_csv(pathToTranMatrix)
	compareFreq = pd.merge(frequency, Var, on ='userid', how = 'left')
	corMatrix = compareFreq.corr()
	corMatrix.to_csv(pathToFreMatrix)
#merge files
path = '/home/lucia/phd_work/mypersonality_data/cognitive_distortion/'
time = pd.read_csv(path + '/data/important_data/twoM_newLabels2.csv')
ids = pd.read_csv(path + '/data/important_data/Id80PerRetained.csv')
lyrics = pd.read_csv(path + '/data/important_data/lyrics/QuotesDetected_all.csv')
lyrics = lyrics[['text','label']]
time2 = pd.merge(lyrics, time, on='text', how = 'left')
time = time2[['text', 'userid', 'time', 'time_diff', 'id', 'label']]
#recode
time['label'].replace(['NotQuote','quote','suspect'],[0,1,1],inplace=True)
#rank time
time['time'] = time['time'].apply(lambda x: x.split()[0])
time['time'] = time['time'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
time = time.sort_values(by=['userid','time_diff','label'],  ascending=False)

lyricsVec = getLyricsVector(time)
#get transition states
TransitionStates = getUserTransitions(lyricsVec)  
#get frequency table
saveStatsPath = path + 'newScripts/moodVector/moodVectorsData/lyricsPostEmptyStats.csv'
frequency = getStats(lyricsVec, saveStatsPath)
#save transition states
filePath = path + 'newScripts/moodVector/moodVectorsData/PostsLyricsTransitions(empty).csv'
saveTransition(filePath, TransitionStates)

#get transition states probability 
Tranprob = TransformProb(filePath)
#get correlation matrix
pathToMerge = path +'newScripts/moodVector/moodVectorsData/PostsLyricsTranEmptyAllVar.csv'
pathToTranMatrix = path + 'newScripts/moodVector/moodVectorsData/PostsLyricsTrancorrelationMatrix(Empty).csv'
pathToFreMatrix = path + 'newScripts/moodVector/moodVectorsData/PostsLyricsFreqMatrix(Empty).csv'
getCorMatrix(frequency, Tranprob, pathToMerge, pathToTranMatrix, pathToFreMatrix)