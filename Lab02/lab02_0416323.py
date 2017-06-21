import sklearn
import numpy as np
import math
import time
from sklearn.neighbors import *


def evalResult(ansList, predList):
	conf = np.array([[0 for i in range(11)] for j in range(11)])
	valid = 0.0
	for ans, pred in zip(ansList, predList):
			if pred == ans:
				valid += 1.0
			conf[ans,pred] += 1
	valid /= len(ansList)
	return valid, conf

def neighborNumbers(trainData):
	dataSize = len(trainData)
	return int(math.pow(dataSize,0.4))

whtFile = 'winequality-white.csv'
whtFile = [line.split(';') for line in open(whtFile,'r').readlines()[1:]]
whtFile = np.array(whtFile).astype(float)
x,y = whtFile[:,:-1], np.sum(whtFile[:,-1:],axis=1)

algoList = ['brute', 'kd_tree', 'ball_tree']
metricList = ['manhattan', 'euclidean','euclidean', 'minkowski']
cosineMetricCount = 2

for cnt,metric in enumerate(metricList):
	if(cnt == cosineMetricCount):
		x=sklearn.preprocessing.normalize(x,axis=1, norm='l2')
	for algo in algoList:
		foldNum = 10
		# Resubstitution
		model = KNeighborsClassifier(
			n_neighbors = neighborNumbers(x),
			algorithm = algo,
			weights = 'distance',
			metric = metric
		).fit(x,y)
		queryTime = 0.0
		base = time.time()
		predList = model.predict(x)
		queryTime += time.time()-base
		valid, conf = evalResult(y, predList)
		print 'Algorithm: ', algo
		print 'Metric: ', metric if cnt!=cosineMetricCount else 'cosine_similarity'
		
		print 'Resubsitution: '
		print 'Query Time: ', queryTime
		print 'Validation: ', valid
		print 'Confusion Matrix:'
		print conf
		print 'K-Fold: k=', foldNum
		
		queryTime=0.0
		valid = 0.0
		for remainder in range(foldNum):
			trainX=[]
			trainY=[]
			testX=[]
			testY=[]
			for i, (insX,insY) in enumerate(zip(x,y)):
				if(i%foldNum == remainder):
					testX.append(insX)
					testY.append(insY)
				else:
					trainX.append(insX)
					trainY.append(insY)
			trainX = np.array(trainX)
			trainY = np.array(trainY)
			testX = np.array(testX)
			testY = np.array(testY)
			model = KNeighborsClassifier(
				n_neighbors = neighborNumbers(trainX),
				algorithm = algo,
				weights = 'distance',
				metric = metric,
			).fit(trainX, trainY)
			base = time.time()
			predList = model.predict(testX)
			queryTime += time.time() -base
			validIncre, confIncre = evalResult(testY, predList)
			valid += validIncre
			conf += confIncre
		valid /= foldNum
		print 'Query Time: ', queryTime
		print 'Validation: ', valid
		print 'Confusion Matrix:'
		print conf, '\n\n'
		

