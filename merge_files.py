import pandas as pd
import matplotlib.pyplot as plt

# import the training and test data
df1 = pd.read_csv('gboost_prediciton.csv', header=0)
df2 = pd.read_csv('ens_prediction.csv', header=0)

# merge the results of two models, taking the averages
# of the probability vectors
print 'Merging Results'
dfout = pd.concat((df1, df2))
dfout = dfout.groupby(dfout.index).mean()

#print dfout.head()

# save prediction:
print 'Writing Merged Results'
dfout.to_csv('ens_ave_prediction_gboost_and_rf.csv', index=False)

