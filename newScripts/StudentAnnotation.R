label1 <- read.csv("~/phd_work/mypersonality_data/cognitive_distortion/data/important_data/studentAnnotations/emotions_historyMar.csv")
colnames(label1) <- c('id','annotator', 'label', 'time')
label2 <- read.csv("~/phd_work/mypersonality_data/cognitive_distortion/data/important_data/studentAnnotations/emotions_historyMar2.csv")
colnames(label2) <- c('id','annotator', 'label', 'time')

all <- rbind(label1,label2)
posts <-  read.csv("~/phd_work/mypersonality_data/cognitive_distortion/data/important_data/database/database/database4.csv")

LabelPosts <- merge(all, posts, by = 'id', all.y= T)

mylabels <- read.csv("~/phd_work/mypersonality_data/cognitive_distortion/data/important_data/twoM_newLabels2.csv")
mylabels <- mylabels[,c('text','negative_ny')]
allLabels <- merge(mylabels, LabelPosts, by = 'text', all.y = TRUE)

allLabelsClean <- allLabels[!duplicated(allLabels$id),]

write.csv(allLabelsClean, "~/phd_work/mypersonality_data/cognitive_distortion/data/important_data/allLabelsClean.csv")
