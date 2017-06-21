import numpy as np
import sklearn, re, sys
import string

dataFile = open('ML_assignment 3_data.txt')
attrFile = open('ML_assignment 3_attr.txt')
[train, test] = [i for i in (re.split(r'(--- training ---)|(---- testing ----)',dataFile.read())) if i!=None and len(i)>20]
train = [[j for j in re.split(r',|\s',i) if len(j)>0] for i in re.split('\n', train) if '?' not in i and len(i)>2]
test = [[j for j in re.split(r',|\s',i) if len(j)>0] for i in re.split('\n', test) if '?' not in i and len(i)>2]
train = np.array(train)
test = np.array(test)
trainY, trainX = train[:,0], train[:,1:].astype(float)
testY, testX = test[:,0], test[:,1:].astype(float)
trainY = [[int(j) for j in re.split('\D*',i) if len(j)>0] for i in trainY]
testY = [[int(j) for j in re.split('\D*',i) if len(j)>0] for i in testY]
attr = re.split(r'-- Class\s\d:.*\r\n',attrFile.read())[1:]
attr = [[term for term in re.split(r'[,\s]',block) if len(term)>0] for block in attr]
for i,block in enumerate(attr):
	for j,term in enumerate(block):
		if(term == 'to'):
			block[j-1:j+2] = ['',string.join(block[j-1:j+2])]
	attr[i] = [[int(digit) for digit in re.split(r'\D*',term) if len(digit)>0] for term in block if len(term)>0]
def getDate(lis):
	return lis[0]+lis[1]*100+lis[2]*10000
attr = [[[getDate(dates)] if len(dates)==3 else
	range(getDate(dates[0:3]), getDate(dates[3:6])+1)
	for dates in block] for block in attr]
trainX = list(trainX)
dClass = {}
for i, block in enumerate(attr):
	for sub in block:
		for term in sub:
			dClass.update({term: i+1})
for i,date in enumerate(trainY):
	temp = date
	date = getDate(date)
	if (date not in dClass):
		trainY[i] = None
		trainX[i] = None
		continue
	trainY[i] = dClass[date]
trainX = [i for i in trainX if i is not None]
trainY = [i for i in trainY if i is not None]

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import preprocessing


trainX = np.array(trainX)
trainY = np.array(trainY)
foldNum = 15
kf = sklearn.model_selection.StratifiedKFold(n_splits = foldNum, shuffle = True)
avgPrecision = 0
# k Fold
for tr,te in kf.split(trainX,trainY):
	foldPrecision = 0
	foldTrainX, foldTrainY = trainX[tr], trainY[tr]
	foldTestX, foldTestY = trainX[te], trainY[te]
	model = GaussianNB().fit(foldTrainX, foldTrainY)
	result = model.predict(foldTestX)
	foldPrecision = sklearn.metrics.precision_score(foldTestY, result, average = 'macro')
	avgPrecision += foldPrecision
avgPrecision /= foldNum
print 'fold avg',avgPrecision
#resub
model = GaussianNB().fit(trainX,trainY)
print 'resub',sklearn.metrics.precision_score(trainY,model.predict(trainX),average = 'macro')
print 'test anwser',model.predict(testX)

