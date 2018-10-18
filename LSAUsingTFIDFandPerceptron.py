import numpy as np
import sys
import os
import platform
import random
import math
import glob
import string

docCountTrain = 0
docCountTest = 0
wordDictTrain = {}
wordDictTest = {}
stopWords = []
classMappingTrain = {}	#Contains Class Label Mapping for Document Numbers of Training Data
classMappingTest = {}	#Contains Class Label Mapping for Document Numbers of Testing Data
DM = ""		#Forward Slash or Training 

# Checking if data directories are provided or not
if(len(sys.argv) < 3):
	print "Training/Testing Data Directories Not Provied."
	sys.exit(1)

train_directory = sys.argv[1]
test_directory = sys.argv[2]

# Getting the forward slash or backward slash depending upon OS
os = platform.system()
if os.lower().startswith('win'):
	DM = '\\'
else:
	DM = '/'

# Appending a slash added at the end if not added
if (train_directory[-1] != DM):
	train_directory += DM
if (test_directory[-1] != DM):
	test_directory += DM

# Helper Methods

# Dictionary Builder to be used later to create BOW of Training Data
def dictBuilderTraining(content):
	global wordDictTrain,docCountTrain
	words = content.split()
	for word in words:
		word = word.lower()
		word = word.translate(None,string.punctuation)	#Removes Punctuation Marks
		if word in stopWords:
			pass
		elif word in wordDictTrain:
			wordDictTrain[word].append(docCountTrain)
		else:
			wordDictTrain[word] = [docCountTrain]
	docCountTrain = docCountTrain + 1

# Dictionary Builder to be used later to create BOW of Testing Data
def dictBuilderTesting(content):
	global wordDictTest,docCountTest
	words = content.split()
	for word in words:
		word = word.lower()
		word = word.translate(None,string.punctuation)	#Remove Punctuation Marks
		if word in stopWords:
			pass
		elif word in wordDictTest:
			temp = wordDictTest[word]
			temp.append(docCountTest)
			wordDictTest[word] = temp
	docCountTest = docCountTest + 1

# Create the Bag of Words using the Dictionary Structure
def createBOW(wordDict,docCount):
	keys = wordDict.keys()
	keys.sort()
	mat = np.zeros([len(keys),docCount])
	count = 0
	for key in keys:
		for dno in wordDict[key]:
			mat[count,dno] = mat[count,dno] + 1
		count = count + 1
	return mat

# Build the TFIDF using the Bag of Words
def buildTFIDF(BOW):
	wordsPerDoc = np.sum(BOW,axis=0)
	docsPerWord = np.sum(np.asarray(BOW > 0,'i'),axis=1)
	rows, cols = BOW.shape
	for i in xrange(rows):
		for j in xrange(cols):
			BOW[i,j] = (BOW[i,j]/wordsPerDoc[j]) * math.log(float(cols)/(1+docsPerWord[i]))
	TFIDF = BOW.T 	# Transpose it to get into proper structure (i.e. docs * words)
	return TFIDF

# Find the training label associated with the document Number
def findTrainingLabel(docNumber):
	for key in classMappingTrain.keys():
		low,high = classMappingTrain[key]
		if low <= docNumber and docNumber <= high:
			return int(key)


def findTestingLabel(docNumber):
	for key in classMappingTest.keys():
		low,high = classMappingTest[key]
		if low <= docNumber and docNumber <= high:
			return int(key)

# Main Program

# Get the list of stopwords
f = open('.'+DM+'stopwords.txt','r')
for line in f:
	stopWords.append(line.strip())

# Generate TF-IDF for Training Data
train_dirs = glob.glob(train_directory+'*'+DM)
for i in xrange(len(train_dirs)):
	docs = glob.glob(train_dirs[i]+'*.txt')
	dname = train_dirs[i].split(DM)[-2]
	if (i==0):
		classMappingTrain[dname] = (docCountTrain,docCountTrain+len(docs))
	else:
		classMappingTrain[dname] = (docCountTrain+1,docCountTrain+len(docs))
	for doc in docs:
		desc = open(doc)
		content = desc.read()
		content = content.replace('\n','')
		dictBuilderTraining(content)

BOWTrain = createBOW(wordDictTrain,docCountTrain)
# BOW is in transpose form i.e. Rows are Words and Columns are Document Numbers (or words * docs)
TFIDFTrain = buildTFIDF(BOWTrain)

"""
# Can Store the TF-IDF Matrix in a file for the Training Data if data is fixed to avoid recalculation.
f = open('TF-IDF.txt','w')
for row in TFIDFTrain:
	f.write(str(list(row)))
	f.write('\n')
f.close()
"""

wordVocab = list(wordDictTrain.keys())		# To keep same vocabulary for Training and Testing Data
# Create Empty Lists in Testing Dictionary Structure
for word in wordVocab:
	wordDictTest[word] = []

# Generate TF-IDF for Testing Data!
test_dirs = glob.glob(test_directory+'*'+DM)
for i in xrange(len(test_dirs)):
	docs = glob.glob(test_dirs[i]+'*.txt')
	dname = test_dirs[i].split(DM)[-2]
	if (i==0):
		classMappingTest[dname] = (docCountTest,docCountTest+len(docs))
	else:
		classMappingTest[dname] = (docCountTest+1,docCountTest+len(docs))
	for doc in docs:
		desc = open(doc)
		content = desc.read()
		content = content.replace('\n','')
		dictBuilderTesting(content)
BOWTest = createBOW(wordDictTest,docCountTest)
TFIDFTest = buildTFIDF(BOWTest)

print "TFIDF Matrices Created."

"""
# SVD CODE
"""

# The Multi-Class Perceptron
classes = range(len(classMappingTrain.keys()))
noFeatures = len(TFIDFTrain[0])
noClasses = len(classes)
# Set the Epochs Here
epochs = 500	# Arbitary value got by trying different values of Epochs vs. Accuracies
w = np.zeros([noClasses,noFeatures+1])	#Weights for the Multiclass Perceptron

# Training the Multi-Class Perceptron
print "Training the Perceptron..."
for _ in xrange(epochs):
	for i in xrange(len(TFIDFTrain)):
		feature = np.hstack([TFIDFTrain[i],1])
		label = findTrainingLabel(i)
		max_act, predClass = 0,0
		for c in classes:
			curr_act = np.dot(feature,w[c])
			if curr_act > max_act:
				max_act = curr_act
				predClass = c
		if predClass != label:
			w[label] = w[label] + feature
			w[predClass] = w[predClass] - feature


# Testing Code
print "Testing the Perceptron..."
total = len(TFIDFTest)
correct = 0
for i in xrange(len(TFIDFTest)):
	test_vector = np.hstack([TFIDFTest[i],1])
	actual_class = findTestingLabel(i)
	max_act, predClass = 0,0
	for c in classes:
		curr_act = np.dot(test_vector,w[c])
		if curr_act > max_act:
			max_act = curr_act
			predClass = c
	if actual_class == predClass:
		correct = correct + 1
miss = total - correct
accuracy = (float(correct)/total)*100
print "Total Predicted: ",total
print "Correct Predicted: ",correct
print "Accuracy: %0.2f" %(accuracy)