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

#compute transition states
def getTransitions(ValenceObject):
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
    preValence = 0
    for valence in ValenceObject:
    #these are self transition states
        if valence == 1 and preValence == 1:
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
        preValence = valence
    return [negaTran, posiTran, mixTran, neuTran, PosAndNeg, MixAndPos, MixAndNeg, MixAndNeu, NeuAndPos, NeuAndNeg]

#get valence vecotr
def getValenceVector(df):
    valenceVec = {}
    valences =[]
    preUser = None
    for valence, day, user in zip(df['negative_ny'], df['time_diff'], df['userid']):
        #first case
        if preUser is None:
            valences = [valence]
            valenceVec[user] = valences
        elif user == preUser:
            valences.append(valence)
        else: 
            valences = [valence]
            valenceVec[user] = valences               
        preUser = user
    #valenceVec[preUser] = valences
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

def computeTrans(savePath2):
	file = pd.read_csv(savePath2)
	file.columns = ['userid','negaTran', 'posiTran', 'mixTran', 'neuTran', 'PosAndNeg', 'MixAndPos', 'MixAndNeg', 'MixAndNeu', 'NeuAndPos', 'NeuAndNeg']
	file['allPosts'] = file.sum(axis=1) 

	#we compute the pobability by dividing the transition with number of all posts
	file.index = file['userid']
	file = file.drop(['userid'], axis=1)
	Tranprob = file.apply(lambda x: x/file.iloc[:,-1])
	Tranprob.to_csv(savePath2)
	return Tranprob

def getCorMatrix(savePath, savePath2, alldata, transitionMatrix):
	Var = alldata[['userid','ope','con','ext','agr','neu','swl','CESD_sum']]
	transitionMatrix['userid'] = transitionMatrix.index
	compare = pd.merge(transitionMatrix, Var, on ='userid', how = 'left')
	corMatrix = compare.corr()
	corMatrix.to_csv(savePath)
	compare.to_csv(savePath2)

def getFrequencyCor(ValenceVec,savePath1,savePath2,alldata):
	negativeD = []
	positiveD = []
	neutralD = []
	mixed = []
	useridL = []
	for userid in valenceVec:
	    negativeD.append(valenceVec[userid].count(1))
	    positiveD.append(valenceVec[userid].count(2))
	    neutralD.append(valenceVec[userid].count(4))
	    mixed.append(valenceVec[userid].count(3))
	    useridL.append(userid)
	#merge all the lists to data frame
	df = pd.DataFrame(np.array(negativeD).reshape(74,1), columns=['NegativePosts'])
	df['PositivePosts'] = positiveD
	df['NeutralPosts'] = neutralD
	df['MixedPosts'] = mixed
	df['userid'] = useridL
	Var = alldata[['userid','ope','con','ext','agr','neu','swl','CESD_sum']]
	compareFreq = pd.merge(df, Var, on ='userid', how = 'left')
	compareFreq.to_csv(savePath1)
	corMatrix = compareFreq.corr()
	corMatrix.to_csv(savePath2)



path = '/Users/lucia/phd_work/cognitive_distortion'
#this file contain users with 80% of posts retained after cleaning foreign language
time = pd.read_csv(path + '/data/important_data/twoM_newLabels80P.csv')
# sort posts according to time
time = SortTime(time)


print('get valence vector')
valenceVec = getValenceVector(time)

print('get TransitionStates')
TransitionStates = getUserTransitions(valenceVec)  

print('save objects')
savePath = path + '/newScripts/moodVector/moodVectorsData/ValenceNoEmpty.pickle'
with open(savePath, 'wb') as handle:
    pickle.dump(TransitionStates, handle, protocol=pickle.HIGHEST_PROTOCOL)

savePath2 = path + '/newScripts/moodVector/moodVectorsData/ValenceNoEmpty.csv'
saveCSV(TransitionStates, savePath2)

print('compute transition states probability')
#we compute the pobability by dividing the transition with number of all posts
Tranprob = computeTrans(savePath2)

print('get correlation corMatrix')
savePath3 = path + '/newScripts/moodVector/moodVectorsData/ValenceNoEmptyCor.csv'
savePath4 = path + '/newScripts/moodVector/moodVectorsData/ValenceNoEmptyAllVar.csv'
allData = pd.read_csv( path + '/data/important_data/user_scale_post_time2.csv')
getCorMatrix(savePath3, savePath4, allData, Tranprob)

savePath5 = path + '/newScripts/moodVector/moodVectorsData/ValenceNoEmptyFreqAllVar.csv'
savePath6 = path + '/newScripts/moodVector/moodVectorsData/ValenceNoEmptyFreqCor.csv'
getFrequencyCor(valenceVec,savePath5, savePath6,allData)





