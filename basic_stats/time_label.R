library(stringr)

time <- as.data.frame(str_split_fixed(labels$post_time, "\\s+", 2))
colnames(time) <- c('date', 'time')

#date2 <- str_split_fixed(time$date, "/", 2)
hour <- str_split_fixed(time$time, ":", 2)
colnames(hour) <- c('hour', 'min')
labels2 <- cbind(labels, hour)
labels2$hour <- as.numeric(as.character(labels2$hour))
labels2$late_night <- 0
labels2$late_night[labels2$hour<6] <- 1

late_night2 <- aggregate(data.frame(late_count = labels2$late_night), list(userid = labels2$userid), length)
late_night2$late_night_score <- late_night2$late_count/60

label_emo_time <- merge(late_night2, label_emo2, by = 'userid')
cor.test(label_emo_time$late_night_score, label_emo_time$CESD_sum)
cor.test(label_emo_time$late_night_score, label_emo_time$CESD_sum)