data selection process: 

1) cleaned_data.rmd : this script matched users who finished CESD, Big5, SWL and Schwartz value scale. 
MARIA: In the README, it's best to write out the full names and content of the scales. 
It generates user_scale_post_time2.csv (333 users, remove users who didn’t complete all the items in CESD, we have 301 users )
MARIA: We may be able to add some more users back in when looking at the amount of missing data. Also, we may 

2) sampling_strategy.rmd: sampling strategy that identify 90 users, generates users_to_be_analysed.csv

3) 60d_beforeCESD.R: select posts that are posted 60 days before they (90 users) completed CESD (4362 posts), generates two_months.csv

data analysis process:

basic_stats.rmd: analyse correlation between the negative disortion / negative emotion score and SWL, CESD, BIG5  (this is the script that generate the results we reported on the paper)

mergelabeleddata.R: a template to merge labeled data with scales

self-disclosure: machine label of self-disclosure level
MARIA: A bit more detail about algorithm?

text_label.py: count word frequency and bigram 
MARIA: in what?

MARIA: There are a few files in this folder that are not documented.
