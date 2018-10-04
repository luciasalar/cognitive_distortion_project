library(stringr)

weibo_all22<-read.csv("weibo_all2.csv", header = T, fill=TRUE,row.names=NULL)
weibo_all3<-read.csv("weibo_all3.csv", header = T, fill=TRUE,row.names=NULL)
weibo_allx<- rbind(weibo_all3, weibo_all22)
weibo_all2x <- weibo_allx[!duplicated(weibo_allx$Source.Text),]

weibo_all_clean2 <-read.csv("weibo_all_clean2.csv", header = T, fill=TRUE,row.names=NULL)
senti_weibo2<- merge(weibo_all_clean2, weibo_all2x, by.x="weibo", by.y="Source.Text", all.x=TRUE)
senti_weibo2 <- senti_weibo2[complete.cases(senti_weibo2),]

write.csv(senti_weibo2, "senti_weibo2.csv")  


weibo_all <- read.csv("data2.csv", header = T, fill=TRUE,row.names=NULL)
weibo_all2 <- read.csv("data3.csv", header = T, fill=TRUE,row.names=NULL)
weibo_all<- rbind(weibo_all, weibo_all2)


#assign NA to blank cells and remove NAs 
weibo_all[weibo_all==''] <- NA
weibo_clean <- weibo_all[complete.cases(weibo_all),]
weibo_clean <- weibo_clean[!duplicated(weibo_clean$weibo),]


#remove non-original weibo
no_share <- weibo_clean[- grep("分享", weibo_clean$weibo),]
no_starSign <- no_share[- grep("摩羯", no_share$weibo), ]
no_apps1 <- no_starSign[- grep("发起的投票", no_starSign$weibo), ]
no_apps2 <- no_apps1[- grep("走过路过不要错过", no_apps1$weibo), ]
no_apps3 <- no_apps2[- grep("一不小心又中奖了", no_apps2$weibo), ]
no_apps4 <- no_apps3[- grep("听说打榜七次可以召唤", no_apps3$weibo), ]
no_apps5 <- no_apps4[- grep("好听哭", no_apps4$weibo), ]
no_apps6 <- no_apps5[- grep("帅哥美女们一起来", no_apps5$weibo), ]
no_apps7 <- no_apps6[- grep("萌王宝座", no_apps6$weibo), ]
no_apps8 <- no_apps7[- grep("有奖测试", no_apps7$weibo), ]
no_apps9 <- no_apps8[- grep("我成功模仿了一段", no_apps8$weibo), ]
no_apps10 <- no_apps9[- grep("个六级词汇单词", no_apps9$weibo), ]
no_apps11 <- no_apps10[- grep("投保扇贝考研保险", no_apps10$weibo), ]
no_apps12 <- no_apps11[- grep("推荐网易公开课", no_apps11$weibo), ]
no_apps13 <- no_apps12[- grep("发表了一篇转载博文", no_apps12$weibo), ]

write.csv(no_apps13, "Weibo_all_clean2.csv")  


