status_1000 <- read.csv("self_label_distortion3.csv", header = T, fill=TRUE,row.names=NULL)
status_5000 <- read.csv("sample_5000.csv", header = T, fill=TRUE,row.names=NULL)
status_5 <- status_5000[complete.cases(status_5000$negative_ny), ]


id <- status_300 %>% dplyr::select(userid)
status_labeled <- merge (id, status_1000, by = 'userid') 
status_labeled %>% dplyr::select(userid, time, text2, negative_yn_self, distortion_yn, quote, magnitude, distortion.category) -> status_labeled
colnames(status_labeled) <- c('userid', 'time','text','negative_ny', 'distortion','quote','magnitude','category')

status_labeled2 <- merge(id, status_5, by = 'userid')
new <- rbind(status_labeled2,status_labeled)


write.csv(new,'labeled.csv')
