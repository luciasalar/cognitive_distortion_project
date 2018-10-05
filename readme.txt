This project contains 4 folders:

1. scripts 
2. data
3. paper
4. prediction model


1. scripts
data selection process: 

1) cleaned_data.rmd : this script matched users who finished CESD, Big5, SWL and Schwartz value scale. It generates user_scale_post_time2.csv (300 users)

2) sampling_strategy.rmd: sampling strategy that identify 90 users

3) 60d_beforeCESD.R: select posts that are posted 60 days before they (90 users) completed CESD (4362 posts)

basic_stats.rmd: analyse correlation between the negative disortion / negative emotion score and SWL, CESD, BIG5  (this is the script that generate the results we reported on the paper)

mergelabeleddata.R: a template to merge labeled data with scales

self-disclosure:machine label of self-disclosure level

text_label.py: count word frquency and bigram 