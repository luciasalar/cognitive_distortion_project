labels <- read.csv("~/phd_work/cognitive_distortion/important_data/emotions_history.csv", header = F, fill=TRUE,row.names=NULL)
names(labels) <- c('id','userid','label','time')
mylab <- read.csv("~/phd_work/cognitive_distortion/important_data/twoM_newLabels2.csv", header = T, fill=TRUE,row.names=NULL)
#recode my label 1-> 2 , 2->1
mylab <- mylab[, c('id','negative_ny')]
mylab$negative_ny <- recode_factor(mylab$negative_ny, '1' = '2' , '2' = '1', '3' = '3' , '4'= '4', '5'='5')


n_occur <- data.frame(table(labels$id))
n_occur[n_occur$Freq > 1,]
cross_Anno <- labels[labels$id %in% n_occur$Var1[n_occur$Freq > 1],]
#