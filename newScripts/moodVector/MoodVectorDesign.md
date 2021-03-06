## This document describe the design of mood vector:

Studies suggest that mood states prompt valenced match emotion, for example, frequent positive emotion within a certain time window indicate a positive mood. The alternation of mood is important for identifying affective disorders. Valence, which is the polarity of mood, emotions, events or situations, is often studied in the social media context as an indicator for mental health. Scholars often inspect post valence in an aggregated form, for example, to identify the correlation between average valence and depressive symptoms. Despite the fact that the aggregated form of valence miss the information of frequency and time, it is still an important feature to infer mental health symptoms. 

Valence in social media text can be better represented other than an aggregated number. Valence including the frequency and time information is more accurate in reflecting mood than in its aggregated form. To include this information in valence, we created mood vector for social media users. The mood vector contains the valence information within a certain time window. In the mood vector, a number is assigned to each day to represent the valence of that day. In this study, the time windows for the mood vector range from 60 days to 14 days before participants completed a CES-D scale. The rationale for time windows is based on the diagnositic mannual DSM-5 suggests that the diagnostic criteria is only met when symptoms of depression last at most days for at least 2 weeks. If participants reported high depressive symptoms on the CES-D scale, we can trace back to the previous 2 weeks, 1 month and 2 months to inspect the valence change near the time they reported high symptoms. 

The mood vector based on social media data reflects mood pattern change, however, the mood pattern shown on social media data might be different from participants' real mood pattern because the number of posts each day varies and people do not post every day. Posting 1 or 2 negative posts on a day doesn't represent the participant has a negative mood on that day. Therefore, mood vector only represent mood changes based on social media data, but not necessarily real life mood. With this limitation in mine, we investigate how mood reflected by social media data related to depressive symptoms. In our sample, we selected participants who posted at least on more than half of the days in two months, in average, participants have 28 days without posting anything. 

# There are two designs for the mood vector so far (WE ADOPTED DESIGN ONE):

* Design 1: Dominant valence is assigned to each day. For each day, the type of valence that occurred most often in posts is assigned to that day in the mood vector. The limitation for this approach is that the vector does not reflect the proportion of the dominant valence posts. The advantage of this design is that we have one vector including information of four types of valence and empty days in categorical data.

The algorithm is implemented as below:

for each day, if negative_count > positive_count or negative_count > neutral_count or neative_count > mix_count, assign negative to that day; if positive_count > negative_count or positive_count > neutral_count or positive_count > mix_count, assign positive to that day; if mix_count >= positive_count or mix_count >= negative_count or negative_count == positive_count, assign mix to that day; if neutral_count >= positive_count or neutral_count >= negative_count or neutral_count > mix_count, assign neutral to that day.


* Design 2: Percentage of dominant valence is assigned to each day. In this design, we assigned a valence score (negative: -1, neutral: 0.01, positive: 1 1) to each post. Valence score of a day is determined by Valence Score(dominant valence)* Number of Posts(dominant valence)/ Number of valenced posts. For example, a user posted 1 negative post, 2 positive posts on Day 1, valence assignment for Day 1 = (1* 2)/(1+2).

The disadvantage of this approach is that negative, neutral and positive value represent an increasement of valence, we can assign -1, 0.01, 1 to them. However, mixed and empty is not an increasement of valence. We need to have these two classes as seperate features. The disadvantage of this approach is that mood vector in here only include information of 3 classes, mixed class and empty days need to be represented by a separate categorical feature. In this separate feature, empty day is assigned 0, mixed day is assigned 1, other assgin 2. We can combine the valence and categorical feature as tuple, the feature will look like [(0.33, 0), (-0.56, 2)...]. This is a 3d feature, when used in a machine learning model, we need to convert it to 2d

The algorithm is implemented as below:
* after applied design 1, if assigned negative, new assignment = (1* N(neg))/N(all); if assigned positive, new assignment = (3* N(pos))/N(all); if assigned neutral, new assignment = 0, if assigned empty, new assigiment = 0. if assigned mixed, new assigiment = 0. Since 'mixed class' is very special that it doesn't indicate an increasement of valence, here we need to have a separate feature for the mixed class, in the new feature, if assigned mixed, (1* N(mix))/N(all), others classes are all assign 0.


* Design 3:  In this design, we will plot positive, we will plot the time series of negative and negative valence separately. We assigned a dominant valence score to to each day that has been annotated as positive/negative according to design 1. Valence score of a day is determined by Valence Score(dominant valence)* Number of Posts(dominant valence)/ Number of valenced posts. For example, on a day which positive valence is the dominant valence, the user posted 1 negative post, 2 positive posts, positive valence assignment for day = (1* 2)/(1+2). We assign 0 to other days. 


* Design 4:  In this design, we will plot positive, we will plot the time series of negative and negative valence separately. We assigned a valence frequency score to to each day. Valence frequency score is determined by the amount of positive/negative valence posts on that day. For example, on a day with 2 positive posts,  valence frequency assignment for day = 2. We assign 0 to other days. The disadvantage of this approach is that the time vector does not include information of how dominant the type of valence is.

# Final designs

Representation 1: Daily Mood Representation 

Hypothesis 1: Social media daily mood transition states reflects mental health dimensions. (REF)
Assign dominant valence to each day, compute the transition states from day to day 

Vector = [  P P E N N  ] 

P - P = 1

P - E = 1

N - E = 1 (N -> E, E -> N)

N - N = 1 

Hypothesis 2: Social media mood frequency reflects mental health dimensions.
Here we compute the frequency of mood episodes

Vector = [  P P E N N  ] 

P = ⅖

N = ⅖

E = ⅕ 

Representation 2: Post Valence Representation 

Hypothesis 1: Social media valence transition states reflects mental health dimensions

Design 1: Empty day is count as a valence in this design

Vector = [ P P N P N P P N E N N P N] 

P - N = 6 (P -> N, N -> P)

N - E = 2 (N -> E, E -> N)

P - P = 2

N - N = 1 

Design 2: Empty day is removed from the valence in this design

Vector = [ P P N P N P P N N N P N] 

P - N = 6 (P -> N, N -> P)

N - N = 1 

P - P = 2

Hypothesis 2: Social media valence frequency reflects mental health dimensions

Design 1: Empty day is count as a valence in this design

Vector = [ P P N P N P P N E N N P N] 

P = 6/13

E = 1/13

N = 6/13

Design 2: Empty day is removed from the valence in this design

Vector = [ P P N P N P P N N N P N] 

P = 6/12

N = 6/12


-----Above text should answer the questions below --------------------
- what you want to achieve with the valence vector
- that the valence vector is actually a time series
- that you won’t have an entry for every day, and that the number of posts will vary from day to day - how do you handle that? And why?
- how you decide between different lengths of the vector, and what lengths of vector you will experiment with 

Given that it is a time series, I would also expect you to acknowledge that creating just a list might be misleading, and to discuss potential alternative representations, such as change to positive, change to negative, change to neutral. 