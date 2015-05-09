import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import BaggingClassifier
#from sklearn.decomposition import PCA
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import svm

# import the training and test data
df = pd.read_csv('data/train.csv', header=0)
cnum = len(df.columns)

# split data into training and test sets
dfTrain, dfTest = train_test_split(df, test_size=0.2)

# PCA dimensionality reduction
#pca_train = PCA(n_components=56)
#pca_train.fit(dfTrain[:,1:cnum-1])
#sum=0.0
#comp=0
#for _ in s:
#    sum += _
#    comp += 1
#    if(sum>=0.95):
#        break
#print comp
#dfTrain_red = pca_train.transform(dfTrain[:,1:cnum-1])
#dfTest_red = pca_train.transform(dfTest[:,1:cnum-1])

# Single model test
#model = GaussianNB()
#model = RandomForestClassifier(n_estimators=80)
#model = KNeighborsClassifier(n_neighbors=6)
#model = BaggingClassifier(KNeighborsClassifier(),
#                          max_samples=0.5, max_features=0.5)
#model = AdaBoostClassifier(n_estimators=80)
model = GradientBoostingClassifier(n_estimators=90, max_depth=8)
#model = svm.LinearSVC()
model.fit(dfTrain[:,1:cnum-1], dfTrain[:,-1])
#model.fit(dfTrain_red, dfTrain[:,-1])
# make and test predictions
expected = dfTest[:,-1]
predicted = model.predict(dfTest[:,1:cnum-1])
#predicted = model.predict(dfTest_red)
# misclassification rate
error_rate = (predicted != expected).mean()
print('%.2f' % error_rate)

# Test multiple models
# for i in range(2, 8, 1):
# 	#model = RandomForestClassifier(n_estimators=80, max_depth=i)
# 	#model = AdaBoostClassifier(n_estimators=50, learning_rate=i)
# 	model = GradientBoostingClassifier(n_estimators=90, max_depth=i)
# 	#model = svm.LinearSVC(C=i)
# 	model.fit(dfTrain[:,1:cnum-1], dfTrain[:,-1])
# 	# make and test predictions
# 	expected = dfTest[:,-1]
# 	predicted = model.predict(dfTest[:,1:cnum-1])
# 	# misclassification rate
# 	error_rate = (predicted != expected).mean()
# 	print('%d:, %.2f' % (i, error_rate))

