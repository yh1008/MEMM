#### MEMM
Maximum-entropy Markov model
##### author: Emily Hua
##### Goal: implement MEMM to predict BOI tags

Information about the program:
some non-ASCII chars(mainly punctuations)  may break the stemmer: replace them with ASCII punc
It has two programs: MEMM_1.py and MEMM_2.py; 1 python pickle file: my_classifier.pickle; one evaluation program: conlleval.pl;
one training file: train.np; one development file: dev.np

###### Run MEMM_1.py takes LONG time as it starts with training the model
###### Run MEMM_2.py takes SHORTER time as it opens the pickle file and starts with testing. 

The program will produce a text file called "boi_output.txt" as the program output

##### Running the program:
---------------------
Option 1: 

	$ python MEMM_1.py  
	
Option 2: 

	$ python MEMM_2.py

##### Testing your program:
---------------------
	$ chmod +x conlleval.pl
	
	$ perl conlleval.pl < boi_output.txt

References: dataset credits to Professor Adam Meyers from NYU
