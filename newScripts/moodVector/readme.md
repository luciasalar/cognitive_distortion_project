## This folder contains everything about the mood vector:

scripts to generate vectors:

* MoodVector.py: mood vector

* postVecEmpty.py: post valece vector contain empty days

* postVectNoEmpty.py: post valence vector does not contain empty days

These scripts compute the mood vector and valence vector, each script generates	
	* a pickle object for the transition states, filename.pickle, 
	* a csv file of the vector
	* a csv file to show the correlation between the vectors and all other variables
	* a csv file with vector information on user level and all other variables
(all the generated files are in moodVectorsData)



* predictCESD.ipynb: using mood vector to predict CESD

* MoodVectorDesign.md: this file explains how we design the mood vector

* timeSeries.md: this file explains how we use the time series.

Intermediate files directory:
jupyter notebook files used for designing the vectors

