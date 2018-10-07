This project contains 4 folders:

1. scripts 
2. data
3. paper
4. prediction model


1. scripts
data selection process: 

1) cleaned_data.rmd : this script matched users who finished CESD, Big5, SWL and Schwartz value scale. It generates user_scale_post_time2.csv (333 users, remove users who didn’t complete all the items in CESD, we have 301 users )

2) sampling_strategy.rmd: sampling strategy that identify 90 users

3) 60d_beforeCESD.R: select posts that are posted 60 days before they (90 users) completed CESD (4362 posts)

basic_stats.rmd: analyse correlation between the negative disortion / negative emotion score and SWL, CESD, BIG5  (this is the script that generate the results we reported on the paper)

mergelabeleddata.R: a template to merge labeled data with scales

self-disclosure:machine label of self-disclosure level

text_label.py: count word frequency and bigram 


Regular users (122): post more than 0.3 post everyday.  30 day: 2331, 60 day: 4696, all posts: 48997, select post count > 20, 60 day (4362)
All users (301):30 day: 3007 , 60day: 6071, all posts: 64164
Regular users whose last post is less than a week before their CES-D posts (90):
30day : 2280,  60days: 4502,  select post count > 20, 60 day (4265)


The sample we used for labelling contains 4154 users, duplicated or empty posts were removed



