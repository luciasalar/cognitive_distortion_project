import pandas as pd
import numpy as np
import statsmodels.api as sm
import csv
from datetime import datetime
import pickle

def SortTime(file):
	#select date
	file['time'] = file['time'].apply(lambda x: x.split()[0])
	#convert to time series
	file['time'] = file['time'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
	#sort values according to userid, time and valence
	file = file.sort_values(by=['userid','time_diff','negative_ny'],  ascending=False)
	return file

#get transition states
def getTransitions(ValenceObject):
    emptyTran = 0
    negaTran = 0
    posiTran = 0
    mixTran = 0
    neuTran = 0
    PosAndNeg = 0
    MixAndPos = 0
    MixAndNeg = 0
    MixAndNeu = 0    
    NeuAndPos = 0
    NeuAndNeg = 0
    EmptyAndPos = 0
    EmptyAndNeg = 0
    EmptyAndMix = 0
    EmptyAndNeu = 0
    preValence = 0
    for valence in ValenceObject:
    #these are self transition states
        if valence == 0 and preValence == 0:
            emptyTran = emptyTran + 1
        elif valence == 1 and preValence == 1:
            negaTran = negaTran + 1
        elif valence == 2 and preValence == 2:
            posiTran = posiTran + 1
        elif valence == 3 and preValence == 3:
            mixTran = mixTran + 1
        elif valence == 4 and preValence == 4:
            neuTran = neuTran + 1
    #positive and negative transition:
        if (valence == 1 and preValence == 2) or (valence == 2 and preValence == 1) :
            PosAndNeg = PosAndNeg + 1
    #mix and positive transition
        if (valence == 3 and preValence == 2) or (valence == 2 and preValence == 3) :
            MixAndPos = MixAndPos + 1
    #mix and negative transition
        if (valence == 3 and preValence == 1) or (valence == 1 and preValence == 3) :
            MixAndNeg = MixAndNeg + 1
    #mix and neutral transition
        if (valence == 3 and preValence == 4) or (valence == 4 and preValence == 3) :
            MixAndNeu = MixAndNeu + 1
    #neutral and postive transition
        if (valence == 4 and preValence == 2) or (valence == 2 and preValence == 4) :
            NeuAndPos = NeuAndPos + 1
    #neutral and negative transition 
        if (valence == 4 and preValence == 1) or (valence == 1 and preValence == 4) :
            NeuAndNeg = NeuAndNeg + 1
    #Empty and positive transition
        if (valence == 0 and preValence == 2) or (valence == 2 and preValence == 0) :
            EmptyAndPos = EmptyAndPos + 1
    
    #Empty and negative transition
        if (valence == 0 and preValence == 1) or (valence == 1 and preValence == 0) :
            EmptyAndNeg = EmptyAndNeg + 1
    
    #Empty and mix transition
        if (valence == 0 and preValence == 3) or (valence == 3 and preValence == 0) :
            EmptyAndMix = EmptyAndMix + 1
    
    #Empty and neutral transition
        if (valence == 4 and preValence == 0) or (valence == -1 and preValence == 4) :
            EmptyAndNeu = EmptyAndNeu + 1
            
            
        preValence = valence
    return [emptyTran, negaTran, posiTran, mixTran, neuTran, PosAndNeg, MixAndPos, MixAndNeg, MixAndNeu, NeuAndPos, NeuAndNeg, EmptyAndPos, EmptyAndNeg, EmptyAndMix, EmptyAndNeu]

#all the users should start with 60 days, if not, that's because their posts contain foreign lanaguge and they were cleaned from the data set, in this case, we added empty to these days
def getValenceVector(df):
    valenceVec = {}
    valences =[]
    preUser = None
    preDay = None
    preValence = []

    for valence, day, user in zip(df['negative_ny'], df['time_diff'], df['userid']):
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


def getUserTransitions(valencVec):
    result = {}
    for item in valencVec:
        result[item] = getTransitions(valencVec[item])
#         print(result)
    return result


def saveCSV(userObj,file):
    data = pd.DataFrame.from_dict(userObj)
    data = data.T
    data.to_csv(file)

#compute transition states
def computeTrans(savePath2):
	file = pd.read_csv(savePath2)
	file.columns = ['userid','emptyTran', 'negaTran', 'posiTran', 'mixTran', 'neuTran', 'PosAndNeg', 'MixAndPos', 'MixAndNeg', 'MixAndNeu', 'NeuAndPos', 'NeuAndNeg', 'EmptyAndPos', 'EmptyAndNeg', 'EmptyAndMix', 'EmptyAndNeu']
	file['allPosts'] = file.sum(axis=1) 

	#we compute the pobability by dividing the transition with number of all posts
	file.index = file['userid']
	file = file.drop(['userid'], axis=1)
	Tranprob = file.apply(lambda x: x/file.iloc[:,-1])
	Tranprob.to_csv(savePath2)
	return Tranprob

#here we get correlation matrix of valenceVec and feature variables
def getCorMatrix(savePath, savePath2, alldata, transitionMatrix):
	Var = alldata[['userid','ope','con','ext','agr','neu','swl','CESD_sum']]
	transitionMatrix['userid'] = transitionMatrix.index
	compare = pd.merge(transitionMatrix, Var, on ='userid', how = 'left')
	corMatrix = compare.corr()
	corMatrix.to_csv(savePath)
	compare.to_csv(savePath2)

#here we get frequency matrix for the valence vector
def getFrequencyCor(ValenceVec,savePath1,savePath2,alldata):
	negativeD = []
	positiveD = []
	neutralD = []
	mixed = []
	empty = []
	useridL = []
	for userid in valenceVec:
	    negativeD.append(valenceVec[userid].count(1))
	    positiveD.append(valenceVec[userid].count(2))
	    neutralD.append(valenceVec[userid].count(4))
	    mixed.append(valenceVec[userid].count(3))
	    empty.append(valenceVec[userid].count(0))
	    useridL.append(userid)
	#merge all the lists to data frame
	df = pd.DataFrame(np.array(negativeD).reshape(74,1), columns=['NegativePosts'])
	df['PositivePosts'] = positiveD
	df['NeutralPosts'] = neutralD
	df['MixedPosts'] = mixed
	df['EmptyPosts'] = empty
	df['userid'] = useridL
	Var = alldata[['userid','ope','con','ext','agr','neu','swl','CESD_sum']]
	compareFreq = pd.merge(df, Var, on ='userid', how = 'left')
	compareFreq.to_csv(savePath1)
	corMatrix = compareFreq.corr()
	corMatrix.to_csv(savePath2)

	
        
path = '/Users/lucia/phd_work/cognitive_distortion'
#this file contain users with 80% of posts retained after cleaning foreign language
file = pd.read_csv(path + '/data/important_data/twoM_newLabels80P.csv')
# sort posts according to time
file = SortTime(file)



print('get valence vector')
valenceVec = getValenceVector(file)

print('get TransitionStates')
TransitionStates = getUserTransitions(valenceVec)  

print('save transition state objects')
savePath = path + '/newScripts/moodVector/moodVectorsData/ValenceEmpty.pickle'
with open(savePath, 'wb') as handle:
    pickle.dump(TransitionStates, handle, protocol=pickle.HIGHEST_PROTOCOL)

savePath2 = path + '/newScripts/moodVector/moodVectorsData/ValenceEmpty.csv'
saveCSV(TransitionStates, savePath2)

print('compute transition states probability')
#we compute the pobability by dividing the transition with number of all posts
Tranprob = computeTrans(savePath2)

print('get correlation corMatrix')
savePath3 = path + '/newScripts/moodVector/moodVectorsData/ValenceEmptyCor.csv'
savePath4 = path + '/newScripts/moodVector/moodVectorsData/ValenceEmptyAllVar.csv'
allData = pd.read_csv( path + '/data/important_data/user_scale_post_time2.csv')
getCorMatrix(savePath3, savePath4, allData, Tranprob)

savePath5 = path + '/newScripts/moodVector/moodVectorsData/ValenceEmptyFreqAllVar.csv'
savePath6 = path + '/newScripts/moodVector/moodVectorsData/ValenceEmptyFreqCor.csv'
getFrequencyCor(valenceVec,savePath5, savePath6,allData)













