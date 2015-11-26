import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
reload(sys)  
sys.setdefaultencoding('utf8')

f = open('my_classifier.pickle', 'rb')
training_file = open("train.np", "rb")
testing_file = open("dev.np", "rb")
output_file = open("boi_output.txt", "wb")

change_of_sentence_flag = 0 #a marker for the end of sentence
BOI_list = ['B-NP', 'I-NP', 'O']
boi_full_list = [] #store all the boi tags that occur in the training set
boi_end_list = [] #store boi tags that are at the end of the sentence
wordStartList = [] #store words that are begining of the sentence
labeled_features = []#store features from the training set
maxent_classifier = pickle.load(f)
change_of_sentence_flag = 0 #a marker for the end of sentence
wordStartList = [] #store words that are begining of the sentence
f.close()

#****************************************************************building input features part.1
previous_BOI = "start"
input_file = training_file
for line in input_file:
	s = re.match(r'^\s*$', line)  #find empty line
	if s:
		change_of_sentence_flag = 1
		previous_BOI = "start"
	else: 
		sentenceList = line.split()
		word = sentenceList[0]
		tag = sentenceList[1]
		boi = sentenceList[2] 
		
		#store words that are begining of the sentence
		if change_of_sentence_flag == 1:
			wordStartList.append(word)
			boi_end_list.append(boi_full_list[-1])
			change_of_sentence_flag = 0
		boi_full_list.append(boi)
       		item = word, tag, boi, previous_BOI
        	labeled_features.append(item)
		previous_BOI = boi
input_file.close()
#****************************************************************calculate End Transition
#calculate the End prior
dicE = {} #temporarry dic 
countTag = 0
countEnd = 0
#calculate the prior (End|state) = C(state, End)/C(state) 
for i in BOI_list:
	for j  in range(len(boi_end_list)):
		for f in boi_full_list:
			if j == 0:
				if i == f:
					countTag = countTag + 1 
		if i == boi_end_list[j]:
			countEnd = countEnd + 1 

	ProbE = format(countEnd/(countTag*1.0), '.5f')

	dicE.update({i: {"END":ProbE}})

	countEnd = 0
	countTag = 0

#****************************************************************building input features part.2
def MEMM_features(word, tag, previous_BOI):
	stemmer = PorterStemmer() 
	features = {} 
	features['current_word'] = word
	features['current_tag'] = tag
	puc = '-'.decode("utf-8")
	features['capitalization'] = word[0].isupper()
	features['start_of_sentence'] = word in wordStartList
	features['cap_start'] = word not in wordStartList and word[0].isupper()
	features['previous_NC'] = previous_BOI
	return features

#********************************************************************Viterbi
def MEMM(wordList,tagList):
	BOI_list = ['B-NP', 'I-NP', 'O']
	w1 = wordList[0] #the first word of the sentence
	t1 = tagList[0]
	tRange = len(BOI_list)
	wRange = len(wordList)
	viterbi = [[0 for x in range(300)] for x in range(300)] 
	backpointer = [['' for x in range(300)] for x in range(300)] 
	#intialization
	for t in range(tRange):#t = 0,1,2
		probability = maxent_classifier.prob_classify(MEMM_features(w1,t1, "start" )) 
		posterior = float(probability.prob(BOI_list[t]))
		#print ("boi: " + BOI_list[t] + ' posterior (start)' + str(posterior))
		#score transition 0(start) -> q given w1
		viterbi[t][1] = posterior
		backpointer[t][1] = 0 #stand for q0 (start point)
	#for word w from 2 to T
	maxViterbi = 0
	maxPreviousState = 0 
	maxPreTerminalProb = 0
	for w in range (1, wRange):	
		for t in range (tRange):
			word = wordList[w]
			tag = tagList[w]
			probability = maxent_classifier.prob_classify(MEMM_features(word,tag,BOI_list[0] )) 
			posterior = float(probability.prob(BOI_list[t]))
			maxViterbi = float(viterbi[0][w]) * posterior
			maxPreviousState = 0
			for i in range (1, tRange):
				word = wordList[w]
				tag = tagList[w]
				probability = maxent_classifier.prob_classify(MEMM_features(word,tag,BOI_list[i] )) 
				posterior = float(probability.prob(BOI_list[t]))
				if float(viterbi[i][w]) * posterior > maxViterbi:
					 maxViterbi = float(viterbi[i][w]) * posterior
					 maxPreviousState = i #content BOI_List[i]		
			viterbi[t][w+1] = maxViterbi	
			backpointer[t][w+1] = BOI_list[maxPreviousState] #points to the matrix x axis (max previous)
			maxViterbi = 0
			maxPreviousState = 0 
			maxPreTerminalProb = 0
	#termination step
	#viterbi[qF, T] = max (viterbi[s,T] *as,qF)
	maxPreTerminalProb = float(viterbi[0][wRange] )* float(dicE[BOI_list[0]]["END"])
	maxPreviousState = 0
	for i in range (1, tRange):
		if float(viterbi[i][wRange]) * float(dicE[BOI_list[i]]["END"]) > maxPreTerminalProb:
			maxPreTerminalProb = float(viterbi[i][wRange]) * float(dicE[BOI_list[i]]["END"]) 
			maxPreviousState = i

	viterbi[tRange][wRange+1] = maxPreTerminalProb 
	#store the state that returns the maxPreTerminalProbability
	backpointer[tRange][wRange+1] = BOI_list[maxPreviousState]
	#return POS tag path 
	pathReverse = [BOI_list[maxPreviousState]]
	maxPreviousTag = BOI_list[maxPreviousState]
	i = 0
	while i < (wRange -1):
		pathReverse.append(backpointer[BOI_list.index(maxPreviousTag)][wRange - i])
		maxPreviousTag = backpointer[BOI_list.index(maxPreviousTag)][wRange - i]
		i = i + 1 
	#reverse the path to make it correct
	index = len(pathReverse)
	path = []
	while index >= 1 :
		path.append(pathReverse[index - 1])
		index = index -1 
	return path

#*******************************************************************MaxEnt+Viterbi = MEMM
#main()
wordList = [] #store words in a sentence
tagList = [] #store part-of-speech tag in a sentence 
boiList = [] #store boi tags in a sentence 
#prob_table = {} #stpre the posterior
previous_BOI = "start"
BOI_list = ['B-NP', 'I-NP', 'O']

input_file = testing_file
for line in input_file:
	
	if line.strip() != '': #if not empty do following 
		sentenceList = line.split()
		word = sentenceList[0]
		print (word)
		tag = sentenceList[1]
		boi = sentenceList[2]
		wordList.append(word)
		tagList.append(tag)
		boiList.append(boi)
		#store words that are begining of the sentence
		#store tags that are the begining of the sentence
		#store the end of sentence tag in tagEndList
		if change_of_sentence_flag == 1:
			wordStartList.append(word)
			change_of_sentence_flag = 0

		#{'B-NP':{'December':0.45}}

	s = re.match(r'^\s*$', line)  #find empty line
	if s:
		#print (wordList)
		change_of_sentence_flag = 1
		previous_BOI = "start"
		path = MEMM(wordList, tagList) #list of BOI_tags returned by HMM function call
		
		for i in range(len(wordList)): #part_of_speech_tag(tagList) and token_list(wordList) has the same length
			output_file.write(wordList[i]+"	"+ tagList[i]+ " " + boiList[i] + " " + path[i] + "\n")

		output_file.write("\n")
		wordList = [] # refresh word list
		tagList = []
		boiList = []
		#prob_table = {}#refresh prob_table

input_file.close()
output_file.close()

