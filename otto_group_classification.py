import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# import the training and test data
dftest = pd.read_csv('data/test.csv', header=0)
dftrain = pd.read_csv('data/train.csv', header=0)

# model
#model = GaussianNB()
#model = RandomForestClassifier(n_estimators=80)
model = GradientBoostingClassifier(n_estimators=400, max_depth=7)
model.fit(dftrain.ix[:,'feat_1':'feat_93'], dftrain['target'])
# make predictions
predicted = model.predict_proba(dftest.ix[:,'feat_1':'feat_93'])

# prepare output
columnsout = [
              'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5',
              'Class_6', 'Class_7', 'Class_8', 'Class_9'
              ]
dfout = pd.DataFrame(predicted, columns=columnsout)
dfout['id'] = dftest['id']
cols = dfout.columns.tolist()
cols = cols[-1:] + cols[:-1]
dfout = dfout[cols]

# save prediction:
dfout.to_csv('gboost_400_prediciton.csv', index=False)

