require(tidyverse)
require(lubridate)
require(dplyr)

post_cutoff <- 0.3

userinfo <- read.csv("user_scale_post_time2.csv")

userinfo$postfreq <- userinfo$post_count / userinfo$date_spread

userinfo$regular <- ifelse(userinfo$postfreq > post_cutoff,1,0) 


userinfo$last_date_ld <-  ymd(userinfo$last_date)

userinfo$first_date_ld <-  ymd(userinfo$earliest_date)

userinfo$date_added_CESD_ld <- ymd_hms(userinfo$date_added_CESD)

userinfo$CESD_lastdate <- as.duration(interval(userinfo$last_date_ld,userinfo$date_added_CESD_ld))

userinfo$CESD_firstdate <- interval(userinfo$first_date_ld,userinfo$date_added_CESD_ld)

# is the last post after the person completed the CES-D?
userinfo$last_post_after_CESD <- userinfo$last_date_ld > date(userinfo$date_added_CESD_ld)

# is the last post in the week before the person completed the CES-D or later?
userinfo$last_post_week_CESD <- (userinfo$last_date_ld - date(userinfo$date_added_CESD_ld)) > -7


regusers <- subset(userinfo,userinfo$regular==1)

nreg <- nrow(regusers)

nuser <- nrow(userinfo)

all_status <- read.csv("user_status_match_dep.csv")
colnames(all_status) <- c('num1', 'userid','post_time','text', 'num2')
all <- merge(regusers, all_status, by = 'userid') %>% dplyr::select(userid,text,post_time,date_added_CESD)
all$time_diff <- difftime(as.Date(all$date_added_CESD), as.Date(all$post_time) , units = c("days"))
two_months <- subset(all, time_diff > 0 & time_diff <= 60)

#number of posts of each user
post_num <- aggregate(data.frame(count = two_months$userid), list(value = two_months$userid), length)
colnames(post_num) <- c('userid','post_count')


two_months <- merge(post_num,two_months, by = 'userid')   #4696
two_months_s <- subset(two_months, post_count > 20)  #4362

write_csv(two_months_s, 'two_months.csv')

two <- read.csv("two_months_clean.csv")
labeled <- read.csv("labeled.csv")
labeled2 <- merge(two, labeled, by = 'text')





