import numpy as np
import matplotlib.pyplot as plt
from TranDataQ22 import td,labels #Import Table 2 for Least Square Classifier

def vanillaPerceptron(epochs,x,y):
	# x is the training data
	# y is labels
	w = np.zeros(len(x[0]))
	b = 0
	for i in xrange(epochs):
		for j in xrange(len(x)):
			if (np.dot(w,x[j])+b)*y[j] <= 0:
				w = w + y[j]*x[j]
				b = b + y[j]
			else:
				continue
	return w,b

class1 = [[3,3],[3,0],[2,1],[0,1.5]]
class2 = [[-1,1],[0,0],[-1,-1],[1,0]]
# Least Square Classifier
y = np.array(labels)
for i in range(len(td)):
	td[i].append(1)
x = np.array(td)
xt = np.matrix(x).getT()	#Calculates X^T
xm = np.matrix(x)			#Converts X to Matrix
a = xt*xm					 #Calculates X^T x X
a = a.getI()				#Calculates (X^TX)^-1
a = a*xt					#Calculates (X^TX)^-1 X
yt = np.matrix(y).getT()	#Calculates Y^T
wlsq = a*yt					#Calculates w = (X^TX)^-1XY^T
# Plotting the Data Points
x1 = []
y1 = []
x2 = []
y2 = []
for val in class1:
	x1.append(val[0])
	y1.append(val[1])
for val in class2:
	x2.append(val[0])
	y2.append(val[1])
plt.scatter(x1,y1,c="red",label="Class 1")
plt.scatter(x2,y2,c="green",label="Class 2")
# Plotting the Least Square Classfier
x = []
y = []
ymax = 3
xmax = 3
x.append(0)
y.append(float(-1*wlsq[2])/wlsq[1])
y.append(0)
x.append(float(-1*wlsq[2])/wlsq[0])
x.append(xmax)
y.append(float((-1*wlsq[0]*xmax)-wlsq[2])/wlsq[1])
y.append(ymax)
x.append(float((-1*wlsq[1]*ymax)-wlsq[2])/wlsq[0])
plt.plot(x,y,c="black",label="Least Square Classifier")
# Fisher's Linear Discriminant
m1 = np.mean(class1,axis=0)
m2 = np.mean(class2,axis=0)
s1 = np.dot((class1-m1).T,(class1-m1))
s2 = np.dot((class2-m2).T,(class2-m2))
s = s1 + s2
w = np.dot(np.linalg.inv(s),(m1-m2))
w = w / np.linalg.norm(w)
# Projecting the Points on the Line
c1d1 = np.dot(class1,w)
projClass1 = []
for value in c1d1:
	projClass1.append(np.dot(value,w))
c2d2 = np.dot(class2,w)
projClass2 = []
for value in c2d2:
	projClass2.append(np.dot(value,w))
# Plotting the Discriminant Line
slope = w[1]/w[0]
plt.plot([-2,4],[-2*slope,4*slope],"c--",label="Fisher Discriminant")
# Plotting the Projected Points and Projections
px1 = []
py1 = []
px2 = []
py2 = []
for pt in projClass1:
	px1.append(pt[0])
	py1.append(pt[1])
for pt in projClass2:
	px2.append(pt[0])
	py2.append(pt[1])
plt.plot(px1,py1,"rx",label="C1 Projection")
plt.plot(px2,py2,"gx",label="C2 Projection")
for i in range(len(px1)):
	plt.plot([x1[i],px1[i]],[y1[i],py1[i]],"r--")
	plt.plot([x2[i],px2[i]],[y2[i],py2[i]],"g--")
# Preparing Data for Perceptron
traindata = []
labels = []
for row in projClass1:
	traindata.append(row)
	labels.append(1)
for row in projClass2:
	traindata.append(row)
	labels.append(-1)
# Classify using Perceptron
wp,b = vanillaPerceptron(1000,traindata,labels)
# Plotting the Classifier Line
xc = range(-2,4)
yc = [(x*(-wp[0]/wp[1])-(b/wp[1])) for x in xc]
plt.plot(xc,yc,c="blue",label="Fisher Classifier")
plt.legend(loc="best")
plt.show()