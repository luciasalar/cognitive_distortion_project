require(dplyr)

s <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000)
sd_score <- aggregate(data.frame(sd_score = s$sd), list(userid = s$userid), sum)
post_count <- aggregate(data.frame(count = s$userid), list(value = s$userid), length)
colnames(post_count) <- c('userid','post_count')
score <- merge(post_count, sd_score, by = 'userid')

s2 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 1000000)
colnames(s2) <- c("userid","date","sd")
sd_score2 <- aggregate(data.frame(sd_score = s2$sd), list(userid = s2$userid), sum)
post_count <- aggregate(data.frame(count = s2$userid), list(value = s2$userid), length)
colnames(post_count) <- c('userid','post_count')
score2 <- merge(post_count, sd_score2, by = 'userid')

s3 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 2000000)
colnames(s3) <- c("userid","date","sd")
sd_score3 <- aggregate(data.frame(sd_score = s3$sd), list(userid = s3$userid), sum)
post_count <- aggregate(data.frame(count = s3$userid), list(value = s3$userid), length)
colnames(post_count) <- c('userid','post_count')
core3 <- merge(post_count, s3, by = 'userid')

s4 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 3000000)
colnames(s4) <- c("userid","date","sd")
sd_score4 <- aggregate(data.frame(sd_score = s4$sd), list(userid = s4$userid), sum)
post_count <- aggregate(data.frame(count = s4$userid), list(value = s4$userid), length)
colnames(post_count) <- c('userid','post_count')
score4 <- merge(post_count, sd_score4, by = 'userid')

s5 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 4000000)
colnames(s5) <- c("userid","date","sd")
sd_score5 <- aggregate(data.frame(sd_score = s5$sd), list(userid = s5$userid), sum)
post_count <- aggregate(data.frame(count = s5$userid), list(value = s5$userid), length)
colnames(post_count) <- c('userid','post_count')
score5 <- merge(post_count, sd_score5, by = 'userid')


s6 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 5000000)
colnames(s6) <- c("userid","date","sd")
sd_score6 <- aggregate(data.frame(sd_score = s6$sd), list(userid = s6$userid), sum)
post_count <- aggregate(data.frame(count = s6$userid), list(value = s6$userid), length)
colnames(post_count) <- c('userid','post_count')
score6 <- merge(post_count, sd_score6, by = 'userid')


s7 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 6000000)
colnames(s7) <- c("userid","date","sd")
sd_score7 <- aggregate(data.frame(sd_score = s7$sd), list(userid = s7$userid), sum)
post_count <- aggregate(data.frame(count = s7$userid), list(value = s7$userid), length)
colnames(post_count) <- c('userid','post_count')
score7 <- merge(post_count, sd_score7, by = 'userid')

s8 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 7000000)
colnames(s8) <- c("userid","date","sd")
sd_score8 <- aggregate(data.frame(sd_score = s8$sd), list(userid = s8$userid), sum)
post_count <- aggregate(data.frame(count = s8$userid), list(value = s8$userid), length)
colnames(post_count) <- c('userid','post_count')
score8 <- merge(post_count, sd_score8, by = 'userid')


s9 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 8000000)
colnames(s9) <- c("userid","date","sd")
sd_score9 <- aggregate(data.frame(sd_score = s9$sd), list(userid = s9$userid), sum)
post_count <- aggregate(data.frame(count = s9$userid), list(value = s9$userid), length)
colnames(post_count) <- c('userid','post_count')
score9 <- merge(post_count, sd_score9, by = 'userid')



s10 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 9000000)
colnames(s10) <- c("userid","date","sd")
sd_score10 <- aggregate(data.frame(sd_score = s10$sd), list(userid = s10$userid), sum)
post_count <- aggregate(data.frame(count = s10$userid), list(value = s10$userid), length)
colnames(post_count) <- c('userid','post_count')
score10 <- merge(post_count, sd_score10, by = 'userid')


s11 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 10000000)
colnames(s11) <- c("userid","date","sd")
sd_score11 <- aggregate(data.frame(sd_score = s11$sd), list(userid = s11$userid), sum)
post_count <- aggregate(data.frame(count = s11$userid), list(value = s11$userid), length)
colnames(post_count) <- c('userid','post_count')
score11 <- merge(post_count, sd_score11, by = 'userid')

s12 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 11000000)
colnames(s12) <- c("userid","date","sd")
sd_score12 <- aggregate(data.frame(sd_score = s12$sd), list(userid = s12$userid), sum)
post_count <- aggregate(data.frame(count = s12$userid), list(value = s12$userid), length)
colnames(post_count) <- c('userid','post_count')
score11 <- merge(post_count, sd_score11, by = 'userid')

s13 <- read.csv("~/phd_work/mypersonality_data/mypersonality-user_status_sd.csv", header = T, fill=TRUE,row.names=NULL, nrows = 1000000, skip = 12000000)
colnames(s13) <- c("userid","date","sd")
sd_score13 <- aggregate(data.frame(sd_score = s13$sd), list(userid = s13$userid), sum)
post_count <- aggregate(data.frame(count = s13$userid), list(value = s13$userid), length)
colnames(post_count) <- c('userid','post_count')
score13 <- merge(post_count, sd_score13, by = 'userid')


c <- rbind(sd_score, sd_score2)
c <- rbind(c, sd_score3)
c <- rbind(c, sd_score4)
c <- rbind(c, sd_score5)
c <- rbind(c, sd_score6)
c <- rbind(c, sd_score7)
c <- rbind(c, sd_score8)
c <- rbind(c, sd_score9)
c <- rbind(c, sd_score10)
c <- rbind(c, sd_score11)
c <- rbind(c, sd_score12)

write.csv(c, "self_dis.csv")





