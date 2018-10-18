# @Author - Mitesh Shah
# @Roll Number - 20172010
import numpy as np
import csv
import sys
from random import seed
from random import randrange
import matplotlib.pyplot as plt

def readIonoSphere():
	f = open("ionosphere.csv","r")
	dataraw = list(csv.reader(f))
	"""
	y = []
	temp = map(float,dataraw[0][:-1])
	x = np.array(temp)
	if dataraw[0][-1].strip() == 'b':
		y.append(1)
	else:
		y.append(-1)
	for i in xrange(1,len(dataraw)):
		temp = map(float,dataraw[i][:-1])
		x = np.vstack((x,np.array(temp,dtype=float)))
		if dataraw[i][-1].strip() == 'b':
			y.append(1)
		else:
			y.append(-1)
	y = np.array(y)
	return x,y
	"""
	return dataraw

def readBreastCancer():
	f = open("breast-cancer-wisconsin.csv","r")
	dataraw1 = list(csv.reader(f))
	dataraw = []
	# Remove the '?' Values
	for row in dataraw1:
		if "?" in row:
			continue
		else:
			dataraw.append(row)
	return dataraw

def VotedPerceptron(x,y,epochs):
	resvectors = []
	w = np.zeros(len(x[0]))
	b = 0.0
	c = 0
	for j in xrange(epochs):
		for i in xrange(len(x)):
			if (np.dot(w,x[i])+b)*y[i] <= 0:
				t = (w,b,c)
				resvectors.append(t)
				w = w + y[i]*x[i]
				b = b + y[i]
				c = 1
			else:
				c += 1
		t = (w,b,c)
		resvectors.append(t)
	return resvectors

def VanillaPerceptron(x,y,epochs):
	w = np.zeros(len(x[0]))
	b = 0.0
	for j in xrange(epochs):
		for i in xrange(len(x)):
			if (np.dot(w,x[i])+b)*y[i] <= 0:
				w = w + y[i]*x[i]
				b = b + y[i]
			else:
				continue
	return w,b

def k_fold_cross_validation(dataset,folds=10):		#Default Value of 10 Fold
	#seed(1)			# Can comment this line to get different data everytime
	dataset_split = []
	dataset_copy = list(dataset)
	fold_size = len(dataset)//folds 	#Using Integer Division
	for i in range(folds):
		fold = []
		while len(fold)<fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def sign(y):
	if y >= 0:
		return 1
	else:
		return -1

def testVoted(x,y,resTuples):
	total = len(x)
	miss = 0
	for i in xrange(len(x)):
		val = 0
		for ele in resTuples:
			val += ele[2]*sign(np.dot(ele[0],x[i])+ele[1])
		if sign(val) != y[i]:
			miss += 1
		else:
			continue
	correct = total - miss
	accuracy = float(correct)/total
	return accuracy

def testVanilla(x,y,w,b):
	total = len(x)
	miss = 0
	for i in xrange(len(x)):
		if sign(np.dot(w,x[i])+b) != y[i]:
			miss += 1
		else:
			continue
	correct = total - miss
	accuracy = float(correct)/total
	return accuracy

if __name__ == "__main__":
	Num_Epochs = range(5,65,5)
	Vanilla_Acc = []
	Voted_Acc = []
	x = readIonoSphere()	# Can change the dataset here.
	for epoch in Num_Epochs:
		print "Epochs : ",epoch
		validatedSet = k_fold_cross_validation(x)
		votedScore = []
		vanillaScore = []
		for fold in validatedSet:
			training = list(validatedSet)
			training.remove(fold)
			# Generate Training Data and Labels
			training_data = []
			training_labels = []
			for data in training:
				for row in data:
					training_data.append(map(float,row[:-1]))
					if row[-1].strip() == 'b':
						training_labels.append(1)
					else:
						training_labels.append(-1)
			training_data = np.array(training_data)
			training_labels = np.array(training_labels)
			# Generate Testing Data and Labels
			testing_data = []
			testing_labels = []
			for row in fold:
				testing_data.append(map(float,row[:-1]))
				if row[-1].strip() == 'b':
					testing_labels.append(1)
				else:
					testing_labels.append(-1)
			resTuples = VotedPerceptron(training_data,training_labels,epoch)
			votedScore.append(testVoted(testing_data,testing_labels,resTuples))
			w,b = VanillaPerceptron(training_data,training_labels,epoch)
			vanillaScore.append(testVanilla(testing_data,testing_labels,w,b))
		print "Voted Perceptron Validation Scores: "
		for ele in votedScore:
			print "%0.2f" %(ele),
		print
		print "Average Accuracy in Voted Perceptron: ",
		avgVot = (float(sum(votedScore))/len(votedScore)) * 100
		print "%0.2f" %(avgVot)
		print "Vanilla Perceptron Validation Scores: "
		for ele in vanillaScore:
			print "%0.2f" %(ele),
		print
		print "Average Accuracy in Vanilla Perceptron: ",
		avgVan = (float(sum(vanillaScore))/len(vanillaScore)) * 100
		print "%0.2f" %(avgVan)
		Voted_Acc.append(avgVot)
		Vanilla_Acc.append(avgVan)
		print "------------------------------------------------------------"
	# Plotting the Data
	v1 = plt.scatter(Num_Epochs,Voted_Acc,c = "red",label = "Voted")
	v2 = plt.scatter(Num_Epochs,Vanilla_Acc,c = "green",label = "Vanilla")
	v1 = plt.plot(Num_Epochs,Voted_Acc,c = "red")
	v2 = plt.plot(Num_Epochs,Vanilla_Acc,c = "green")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend(loc = "center left")
	plt.suptitle("Vanilla v/s Voted : Ionosphere Dataset", fontsize = 18)
	plt.show()