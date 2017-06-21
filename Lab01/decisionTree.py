import numpy as np
import sklearn
from sklearn import datasets
from sklearn import tree
from sklearn import preprocessing
import math

data=sklearn.datasets.load_iris()
x=data.data
y=data.target
x=preprocessing.scale(x)


class CircularForest:	
	def __init__(self):
		self.model=[]
	def train(self,x,y):
		# Training with pre-pruning:
		# maximal depth: log2(instances)
		depth = 3
		treeCount= 5
		self.model=[]
		for remainder in range(treeCount):
			sX=[]
			sY=[]
			for i in range(len(x)):
				if(i%treeCount!=remainder):
					sX.append(x[i])
					sY.append(y[i])
			subModel = sklearn.tree.DecisionTreeClassifier(max_depth=depth).fit(sX,sY)
			self.model.append(subModel)
		return self
	def predict(self,x):
		predMatrix=[]
		for subModel in self.model:
			predMatrix.append(subModel.predict(x))
		predMatrix=np.array(predMatrix)
		result=[]
		for instanceCnt in range(len(x)):
			vote=predMatrix[:,instanceCnt]
			candidate = np.array([0 for i in range(3)])
			for i in vote:
				candidate[i]+=1
			result.append(np.argmax(candidate,axis=0))
		return np.array(result)

def validate(pred,ans):
	valid=0.0
	conf=np.array([[0 for i in range(3)] for j in range(3)])
	for i in range(len(pred)):
		conf[ans[i],pred[i]]+=1
		if(pred[i]==ans[i]):
			valid+=1
	return valid,conf


# Resubsitiution
totalModel=CircularForest().train(x,y)
totalPred=totalModel.predict(x)
totalValid, totalConf=validate(totalPred,y)
totalValid/=len(x)
print 'Resubstitution valid: ', totalValid
print 'Resubstitution Confusion Matrix:\n', totalConf,'\n'

# k fold validation
k=10
foldValid=0.0
foldConf = np.array([[0 for i in range(3)] for j in range(3)])
for remainder in range(k):
	sX=[]
	sY=[]
	tX=[]
	tY=[]
	for i in range(len(x)):
		if(i%k==remainder):
			tX.append(x[i])
			tY.append(y[i])
		else:
			sX.append(x[i])
			sY.append(y[i])
	
	
	sX,sY,tX,tY= np.array(sX), np.array(sY), np.array(tX), np.array(tY)

	foldModel = CircularForest().train(sX,sY)
	foldPred = foldModel.predict(tX)
	validIncre, confIncre = validate(foldPred,tY)
	foldValid+=validIncre
	foldConf+=confIncre

foldValid/=len(x)
print 'k fold valid: ',foldValid
print 'k fold confusion matrix:\n',foldConf

