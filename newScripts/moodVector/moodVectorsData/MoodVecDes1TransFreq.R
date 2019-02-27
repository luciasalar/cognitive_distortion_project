
#set path
setwd('/Users/lucia/phd_work/cognitive_distortion/') 

file <- read.csv('newScripts/moodVector/moodVectorsData/MoodVecDes1.csv')
negTran <- sapply(levels,function(x)rowSums(file==1))
emptyTran <- sapply(levels,function(x)rowSums(file==-1))
posTran <- sapply(levels,function(x)rowSums(file==2))
mixTran <- sapply(levels,function(x)rowSums(file==3))
neuTran <- sapply(levels,function(x)rowSums(file==4))


FreqDf <- cbind(file['userid'], negTran[,1])
FreqDf <- cbind(FreqDf, emptyTran[,1])
FreqDf <- cbind(FreqDf, posTran[,1])
FreqDf <- cbind(FreqDf, mixTran[,1])
FreqDf <- cbind(FreqDf, neuTran[,1])
colnames(FreqDf) <- c('userid','negFreq', 'emptyFreq', 'posFreq', 'mixFreq', 'neuFreq')

#merge with scales info

allData <- read.csv('data/important_data/user_scale_post_time2.csv')
allData <-  allData[,c('userid','ope','con','ext','agr','neu','swl','CESD_sum')]
compare <- merge(FreqDf, allData, by = 'userid')

corDf <- cor(compare[2:13])
write.csv(corDf, file = 'newScripts/moodVector/moodVectorsData/MoodDes1FreqcorrelationMatrix.csv')

