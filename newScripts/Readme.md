# This folder contains:

* scripts that generate database: DataBase.ipynb (database/database4.csv is data set)

* stats description of annotation: basicStatsCor.ipynb

* Valence vector for each user and the stats description: ValenceVector.ipynb

* CesdClassifier_Valence.ipynb: using valence vector to predcit depression score. 

* Interrater_reliability.Rmd: this script computes the interrater reliability between my annotation and the students annotation. Basic stats of student annotation.

* statsForMoodVec.Rmd: this script computes the stats for data that built the mood vector


Valence vector:

* Design 1: for each day, if negative_count > positive_count or negative_count > neutral_count or neative_count > mix_count, assign negative to that day; if positive_count > negative_count or positive_count > neutral_count or positive_count > mix_count, assign positive to that day; if mix_count >= positive_count or mix_count >= negative_count or negative_count == positive_count, assign mix to that day; if neutral_count >= positive_count or neutral_count >= negative_count or neutral_count > mix_count, assign neutral to that day.

* Design 2: after applied design 1, if assigned negative, new assignment = (1* N(neg))/N(all); if assigned positive, new assignment = (3* N(pos))/N(all); if assigned neutral, new assignment = 0, if assigned empty, new assigiment = 0. if assigned mixed, new assigiment = 0. Since 'mixed class' is very special that it doesn't indicate an increasement of valence, here we need to have a separate feature for the mixed class, in the new feature, if assigned mixed, (1* N(mix))/N(all), others classes are all assign 0.


