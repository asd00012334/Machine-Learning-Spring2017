import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
def statusCode(literal):
	if literal == 'ok':
		return 0
	elif literal == 'settler':
		return 1
	else: return 2
def codeStatus(code):
	if code==0:
		return 'ok'
	elif code==1:
		return 'settler'
	else: return 'solids'

data = [[168,3,1814,15,0.001,1879,'ok'],
[156,3,1358,14,0.01,1425,'ok'],
[176,3.5,2200,16,0.005,2140,'ok'],
[256,3,2070,27,0.2,2700,'ok'],
[230,5,1410,131,3.5,1575,'settler'],
[116,3,1238,104,0.06,1221,'settler'],
[242,7,1315,104,0.01,1434,'settler'],
[242,4.5,1183,78,0.02,1374,'settler'],
[174,2.5,1110,73,1.5,1256,'settler'],
[1004,35,1218,81,1172,33.3,'solids'],
[1228,46,1889,82.4,1932,43.1,'solids'],
[964,17,2120,20,1030,1996,'solids'],
[2008,32,1257,13,1038,1289,'solids']
]

window = 2
f = 6
for k in range(6):
	for i in range(len(data)-window):
		avg = 0.0
		for j in(i,i+window):
			avg+=data[j][k]
		avg/=window
		data[i][k]=avg

query = [222,4.5,1518,74,0.25,1642]
data = np.array(data)
x,y = data[:,:-1], data[:,-1]
y = list(y)
x = x.astype(float)
y = [statusCode(i) for i in y]
model = GaussianNB().fit(x,y)
print codeStatus(model.predict([query])[0])
