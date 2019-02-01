######
labels <- read.csv("~/phd_work/cognitive_distortion/important_data/emotions_history.csv", header = F, fill=TRUE,row.names=NULL)
names(labels) <- c('id','userid','label','time')
mylab <- read.csv("~/phd_work/cognitive_distortion/important_data/twoM_newLabels2.csv", header = T, fill=TRUE,row.names=NULL)
#recode my label 1-> 2 , 2->1
mylab <- mylab[, c('id','negative_ny','text')]
mylab$negative_ny <- recode_factor(mylab$negative_ny, '1' = '2' , '2' = '1', '3' = '3' , '4'= '4', '5'='5')


n_occur <- data.frame(table(labels$id))
n_occur[n_occur$Freq > 1,]
cross_Anno <- labels[labels$id %in% n_occur$Var1[n_occur$Freq > 1],]
cross_Anno$time <- NULL
#merge with text
cross_Anno2 <- merge(mylab, cross_Anno, by = 'id', all.y = T)
write.csv(cross_Anno2, '~/phd_work/cognitive_distortion/data/crossAnnotation2.csv')

##a testing dataset
labels <- read.csv("~/phd_work/cognitive_distortion/data/emotions_history_old.csv", header = F, fill=TRUE,row.names=NULL)
names(labels) <- c('id','userid','label','time')
db <- read.csv("~/phd_work/cognitive_distortion/data/temp db data/database3.csv", header = T, fill=TRUE,row.names=NULL)
db <- db[, c('id','text')]
n_occur <- data.frame(table(labels$id))
n_occur[n_occur$Freq > 1,]
cross_Anno <- labels[labels$id %in% n_occur$Var1[n_occur$Freq > 1],]
#for some reason the database doesn't record the userid correctly when there are two annotation, it recorded the first 'id', 
#so we can see the userid is the same, but the annotation time is clearly different

#merge with text
cross_Anno2 <- merge(db, cross_Anno, by = 'id', all.y = T)
#there are 3 repeatitions, one user submitted the same item twice
write.csv(cross_Anno2, '~/phd_work/cognitive_distortion/data/crossAnnotation.csv')
#then combine the two sets mannually. 