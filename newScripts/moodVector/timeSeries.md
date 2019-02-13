## Section 1: valence vector design

Studies suggest that mood states prompt valenced match emotion, for example, frequent positive emotion within a certain time window indicate a positive mood. The alternation of mood is important for identifying affective disorders. Valence, which is the polarity of mood, emotions, events or situations, is often studied in the social media context as an indicator for mental health. Scholars often inspect post valence in an aggregated form, for example, to identify the correlation between average valence and depressive symptoms. Despite the fact that the aggregated form of valence miss the information of frequency and time, it is still an important feature to infer mental health symptoms. 

Valence in social media text can be better represented other than an aggregated number. Valence including the frequency and time information is more accurate in reflecting mood than in its aggregated form. To include this information in valence, we created mood vector for social media users. The mood vector contains the valence information within a certain time window. In the mood vector, a number is assigned to each day to represent the valence of that day. In this study, the time windows for the mood vector range from 60 days to 14 days before participants completed a CES-D scale. The rationale for time windows is based on the diagnositic mannual DSM-5 suggests that the diagnostic criteria is only met when symptoms of depression last at most days for at least 2 weeks. If participants reported high depressive symptoms on the CES-D scale, we can trace back to the previous 2 weeks, 1 month and 2 months to inspect the valence change near the time they reported high symptoms. 

The mood vector based on social media data reflects mood pattern change, however, the mood pattern shown on social media data might be different from participants' real mood pattern because the number of posts each day varies and people do not post every day. Posting 1 or 2 negative posts on a day doesn't represent the participant has a negative mood on that day. Therefore, mood vector only represent mood changes based on social media data, but not necessarily real life mood. With this limitation in mine, we investigate how mood reflected by social media data related to depressive symptoms. In our sample, we selected participants who posted at least on more than half of the days in two months, in average, participants have 28 days without posting anything. 

There are two designs for the mood vector so far:

**Design 1**: Dominant valence is assigned to each day. For each day, the type of valence that occurred most often in posts is assigned to that day in the mood vector. The limitation for this approach is that the vector does not reflect the proportion of the dominant valence posts. The advantage of this design is that we have one vector including information of four types of valence and empty days in categorical data.

The algorithm is implemented as below:

for each day, if negative_count > positive_count or negative_count > neutral_count or neative_count > mix_count, assign negative to that day; if positive_count > negative_count or positive_count > neutral_count or positive_count > mix_count, assign positive to that day; if mix_count >= positive_count or mix_count >= negative_count or negative_count == positive_count, assign mix to that day; if neutral_count >= positive_count or neutral_count >= negative_count or neutral_count > mix_count, assign neutral to that day.


**Design 2**: Percentage of dominant valence is assigned to each day. In this design, we assigned a valence score (negative: -1, neutral: 0.01, positive: 1 1) to each post. Valence score of a day is determined by Valence Score(dominant valence)* Number of Posts(dominant valence)/ Number of valenced posts. For example, a user posted 1 negative post, 2 positive posts on Day 1, valence assignment for Day 1 = (1* 2)/(1+2).

The disadvantage of this approach is that negative, neutral and positive value represent an increasement of valence, we can assign -1, 0.01, 1 to them. However, mixed and empty is not an increasement of valence. We need to have these two classes as seperate features. The disadvantage of this approach is that mood vector in here only include information of 3 classes, mixed class and empty days need to be represented by a separate categorical feature. In this separate feature, empty day is assigned 0, mixed day is assigned 1, other assgin 2. We can combine the valence and categorical feature as tuple, the feature will look like [(0.33, 0), (-0.56, 2)...]. This is a 3d feature, when used in a machine learning model, we need to convert it to 2d

The algorithm is implemented as below:
* after applied design 1, if assigned negative, new assignment = (1* N(neg))/N(all); if assigned positive, new assignment = (3* N(pos))/N(all); if assigned neutral, new assignment = 0, if assigned empty, new assigiment = 0. if assigned mixed, new assigiment = 0. Since 'mixed class' is very special that it doesn't indicate an increasement of valence, here we need to have a separate feature for the mixed class, in the new feature, if assigned mixed, (1* N(mix))/N(all), others classes are all assign 0.


**Design 3**:  In this design, we will plot positive, we will plot the time series of negative and negative valence separately. We assigned a dominant valence score to to each day that has been annotated as positive/negative according to design 1. Valence score of a day is determined by Valence Score(dominant valence)* Number of Posts(dominant valence)/ Number of valenced posts. For example, on a day which positive valence is the dominant valence, the user posted 1 negative post, 2 positive posts, positive valence assignment for day = (1* 2)/(1+2). We assign 0 to other days. 


**Design 4**:  In this design, we will plot positive, we will plot the time series of negative and negative valence separately. We assigned a valence frequency score to to each day. Valence frequency score is determined by the amount of positive/negative valence posts on that day. For example, on a day with 2 positive posts,  valence frequency assignment for day = 2. We assign 0 to other days. The disadvantage of this approach is that the time vector does not include information of how dominant the type of valence is.




## Section 2: how we use the time series (mood vector)

1. Provide a compact valence description of the dataset (2 months before participants reported CESD) using HMM
2. Demonstrate trends of positive and negative valence 60 days before participants reported CESD. To demonstrate the valence, it's good to know the magnitude of the valence by computing the proportion of dominant valence posts among all the posts,design 3 is more appropriate to this task.
3. Explain trend difference between high CES-D and low CES-D group.

4. observe relationship between valence and other variables:
	1. satisfaction with life (I would suggest to remove this because it's not directly related to depression, it might be confusing to the theme)
		observe trend difference between people with high and low satisfaction with life. Satisfaction with life is a relatively stable variable. Participant's valence should not change much according to time, except that they have high CES-D score. Hence, the frequency of having positive/negative emotion might be more informative to satisfaction with life.  Design 4 might be a good option for this task. 
	2. personality 
	neuroticism:  
	Rumination and worry, cognitive reactivity have been found to be mediators in the relationship between neuroticism and symptoms of depressin and anxiety. A mediator explains why there is a relationship between the two variables. Since neuroticism is directly linked to the behavior of rumination, worry and cognitive reactivity, hence neuroticism might have a stronger correlation with the mood vector than with depression. We can do correlation tests, divide participants with high/low scores in neuriticism and see their valence trend. 

	Similar to satisfaction with life, neuroticism is a relatively stable varible. However, the high CESD group should show more frequent negative valence as they probably shows more rumination and worry in the posts (this is a speculation, need futher study, but we should state our speculation on the paper), which contains negative emotion. Design 4 seem to be more appropriate to this task.


---------
citations:

Roelofs, J., Huibers, M., Peeters, F., Arntz, A., & van Os, J. (2008). Rumination and worrying as possible mediators in the relation between neuroticism and symptoms of depression and anxiety in clinically depressed individuals. Behaviour Research and Therapy, 46(12), 1283-1289.

Barnhofer, T., & Chittka, T. (2010). Cognitive reactivity mediates the relationship between neuroticism and depression. Behaviour Research and Therapy, 48(4), 275-281.


##Section 3: The hidden markov model 
**The observable layer**
1. **design 1**
Use design 4 valence vector as the observable layer, we will have a positive valence vector and a negative valence vector. In the positive valence vector, we will have two states: positive, other. In the negative valence vector, we have two states: negative, other. 

Count the transition probability of each state:
positive -> other
other -> positive 
Positive -> positive
Other -> other 

Do the same for negative valence vector

The probability of positive and negative self-transition states represent the social media mood. 

Note: the disadvantage of this method is that the vector is relatively sparse. The other state is not specified, hence, neutral, mixed and empty states are lost. In fact, the self-transition of neutral state is important because we can see participant’s reactivity in daily life.

------------
2. **design 2**
Use design 1 vector vector as the observable layer,  we have five observable states in social media valence: 
positive, negative, mixed, neutral, empty

Count the transition probability of:
positive -> negative 
negative -> positive 
mix -> positive
Positive -> mix
mix -> negative
negative -> mix
Neutral ->positive 
Positive -> neutral 
Neutral ->negative 
negative -> neutral 
Neutral ->mix 
mix -> neutral 
Empty -> positive
positive -> empty
Empty -> negative
negative -> empty
Empty -> mix
mix -> empty
Empty ->neutral 
neutral  -> empty
And all the self-transition states

*The advantage of this approach is that all the states are included in this vector. We can also see the transition states of mix
Therefore, I suggest to use design one to display positive, negative valence pattern, use design 1 in the HMM model*

-----------
**The hidden layer (mood vector)**

Suppose we have positive (PM), negative (NEM) and neutral mood (NM) (hidden layer)
four states: positive (PV), negative (NEV) and neutral valence (NV) (observable layer)

If we define mood as continuous states of the valence, for positive mood, the observable layer tags must have at least two continuous PV:
PV, PV

 for negative mood, the observable layer tags must have at least two continuous NEV: 
NEV, NEV

However, there’s is no clear definition of how long a mood should last. For example, positive mood can be represented by  PV, PV or  PV, PV, PV, PV, or PV, PV, PV.. hence, the combination for positive mood can be many. We can compute the emission probability of PM like this:
PV, PV  0.2
PV, PV, PV 0.05
PV, PV, PV, PV, PV, PV  0.001
NEV, NEV  0
NEM, NEM 0

In case like this:
NEM, NEM, PV, NEV, NEV, NEM, NEM,
One ‘PV’ doesn’t represent PM. Shall we ignore PV or include PV in the previous or later string?

For the above example,  we will have hidden states:
NM, NM, NM 

Hence, for each participant, we will have a hidden state such as:
Participant 1:  NM, NM, NM 
Participant 2:  NM, NM, NM, PM, NM
Here we can compute the transition probability of the hidden state, which is the social media mood alteration of a participant. Then we can compare the mood alteration of a participant with depression score and personality score

*Note*: we need to emphasize that social media valence, social media mood is different from real life valence and mood. This technique can be applied to self-reported valence data. Valence is different from emotions, emotion is a person’s response to a stimuli, valence is the polarities of these responses or events, situations described in the text. However, events, situations described with valenced words also imply that the author has certain valenced emotions towards the events. Therefore, here we make the assumption that valence is very similar to people’s emotion in the social media context. 










