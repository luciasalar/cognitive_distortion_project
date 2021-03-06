---
title: "Data on CSCW Paper"
author: "Lushi Chen"
date: "12 3 2018"
output: html_document
---

 
require(data.table)
require(dplyr)
require(tidyverse)



#FB posts from users completed CES-D scale 
all_status <- read.csv("./data/user_status_match_dep.csv", header = T, fill=TRUE,row.names=NULL)
colnames(all_status) <- c('num1', 'userid','post_time','text', 'num2')

#User completed CES-D, demographic information, Big 5, schwartz  and SWL scale, here we refer them as 'the selected group'
status_300 <- read.csv("./data/status_dep_demog_big5_schwartz_swl_like.csv", header = T, fill=TRUE,row.names=NULL)

#select userid of the selected group
status_300 %>% dplyr::select(userid) -> status_3

#compute the total number of posts of each user in the selected group
selected <- merge(all_status, status_3, by = 'userid') %>% dplyr::select(userid, post_time, text)
uni <- unique(selected$userid)
post_count <- aggregate(data.frame(count = selected$userid), list(value = selected$userid), length)
colnames(post_count) <- c('userid','post_count')

#drop columns that do not need to be used
status_300$text <- NULL
status_300$like_id <- NULL

###sum scale values
#CES-D scale
#remove users with value <= 0
threshhold <- 0  
status_300 <- subset(status_300, status_300[ ,12] > threshhold & status_300[ ,13] > threshhold &  status_300[ ,14] > threshhold
                      & status_300[ ,15] > threshhold & status_300[ ,16] > threshhold & status_300[ ,17] > threshhold & status_300[ ,18] > threshhold
                      & status_300[ ,19] > threshhold & status_300[ ,20] > threshhold & status_300[ ,21] > threshhold & status_300[ ,22] > threshhold
                      & status_300[ ,23] > threshhold & status_300[ ,24] > threshhold  & status_300[ ,25] > threshhold & status_300[ ,26] > threshhold
                      & status_300[ ,27] > threshhold & status_300[ ,28] > threshhold  & status_300[ ,29] > threshhold & status_300[ ,30] > threshhold
                      & status_300[ ,31] > threshhold) 


#reverse item 4, 8, 12, 16  http://mypersonality.org/wiki/doku.php?id=ced-d_depression_scale
status_300$q4_x <- status_300$q4_x * -1
status_300$q8_x <- status_300$q8_x * -1
status_300$q12_x <- status_300$q12_x * -1
status_300$q16_x <- status_300$q16_x * -1
status_300$dep_sum <- rowSums(status_300[12:31])

#schwartz http://mypersonality.org/wiki/doku.php?id=list_of_variables_available#schwartz_s_values_survey
status_300$conformity <- rowSums(status_300[,c('q11_y','q20_y','q40','q47')])
status_300$tradition <- rowSums(status_300[,c('q18_y','q32','q36','q44','q51')])  
status_300$benevolence <- rowSums(status_300[,c('q33','q45','q49','q52','q54')])
status_300$universalism <- rowSums(status_300[,c('q1_y','q17_y','q24','q26','q29','q30','q35','q38')])
status_300$self_direction <- rowSums(status_300[,c('q5_y','q16_y','q31','q41','q53')])  
status_300$stimulation <- rowSums(status_300[,c('q9_y','q25','q37')])  
status_300$hedonism <- rowSums(status_300[,c('q4_y','q50','q57')])  
status_300$achievement <- rowSums(status_300[,c('q34','q39','q43','q55')])  
status_300$power <- rowSums(status_300[,c('q3_y','q12_y','q27','q46')])  
status_300$tradition <- rowSums(status_300[,c('q8_y','q13_y','q15_y','q22','q56')])

#select columns and change column names
selected_user <- status_300 %>% dplyr::select('userid','date_added_x', 'time_completed_x','ethnicity',
                                              'marital_status','parents_together','gender','birthday','age','relationship_status',
                                              'interested_in','mf_relationship','mf_dating','mf_random','mf_friendship','mf_whatever',
                                              'mf_networking','locale','network_size', 'timezone','ope','con','ext','agr','neu','blocks',
                                              'date','date_added_y','time_completed_y','completed','swl','dep_sum','conformity','tradition',
                                              'benevolence','universalism','self_direction', 'stimulation','hedonism', 'achievement','power')

colnames(selected_user) <- c('userid','date_added_CESD', 'time_completed_CESD','ethnicity',
                             'marital_status','parents_together','gender','birthday','age','relationship_status',
                             'interested_in','mf_relationship','mf_dating','mf_random','mf_friendship','mf_whatever',
                             'mf_networking','locale','network_size', 'timezone','ope','con','ext','agr','neu','blocks',
                             'date','date_added_schzwartz','time_completed_schzwartz','completed_schzwartz','swl','CESD_sum','conformity','tradition',
                             'benevolence','universalism','self_direction', 'stimulation','hedonism', 'achievement','power')

write_csv(selected_user,'selected_user2.csv')

selected_user_post <- merge(post_count,selected_user, by = 'userid')

#compute the earliest and last post time
time <- all_status %>% dplyr::select('userid','post_time')
time$post_time <- as.Date(time$post_time)

time %>% dplyr::group_by(userid)  %>% dplyr::summarise(earliest_date = min(post_time)) -> earliest_time
time %>% dplyr::group_by(userid)  %>% dplyr::summarise(last_date = max(post_time)) -> last_time
  
time2 <- merge(earliest_time, last_time, by = "userid")
time2$date_spread <- difftime(as.Date(time2$last_date), as.Date(time2$earliest_date) , units = c("days"))

all <- merge(time2, selected_user_post, by = 'userid')


write.csv(all, 'user_scale_post_time2.csv')





