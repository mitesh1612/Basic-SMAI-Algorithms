# @Author - Mitesh Shah
# @Roll Number - 20172010
import numpy as np
import csv
import sys

def readIonoSphere():
	f = open("ionosphere.csv","r")
	dataraw = list(csv.reader(f))
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
	"""
	# To Store the "?" Removed Data
	f = open("newbcd.csv","w")
	for row in dataraw:
		f.write(','.join(row)+"\n")
	f.close()
	"""
	y = []
	temp = map(float,dataraw[0][1:-1])
	x = np.array(temp)
	if dataraw[0][-1] == '2':
		y.append(1)
	else:
		y.apppend(-1)
	for i in xrange(1,len(dataraw)):
		temp = map(float,dataraw[i][1:-1])
		x = np.vstack((x,np.array(temp,dtype=float)))
		if dataraw[i][-1] == '2':
			y.append(1)
		else:
			y.append(-1)
	y = np.array(y)
	return x,y

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

if __name__ == "__main__":
	print "Select your Dataset."
	print "Enter 1. for Ionosphere Dataset"
	print "Enter 2. for Breast Cancer Dataset"
	choice = input("Enter your choice: ")
	if choice == 1:
		traindata,trainlabels = readIonoSphere()
	elif choice == 2:
		traindata,trainlabels = readBreastCancer()
	else:
		print "Wrong Choice Entered!"
		sys.exit(1)
	print "Enter 1. for Vanilla Perceptron."
	print "Enter 2. for Voted Perceptron."
	choice1 = input("Enter your choice: ")
	epochs = input("Enter the Number of Epochs: ")
	if choice1 == 1:
		w,b = VanillaPerceptron(traindata,trainlabels,epochs)
		print "Perceptron Trained."
		print "Weight Vector:\n",w
		print "Bias Value:\n",b
	else:
		resTuples = VotedPerceptron(traindata,trainlabels,epochs)
		print "Perceptron Trained."
		print resTuples