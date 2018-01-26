#dependencies
import pandas as pd
import quandl as qd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df = qd.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100
df['PCT_CHG']  = df['Adj. Close'] - df['Adj. Open']/ df['Adj. Open']*100

df = df[['Adj. Close','HL_PCT','PCT_CHG','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.002*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X) #skip for high frequency trading

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf1 = LinearRegression(n_jobs = -1)
clf2 = svm.SVR()
clf3 = svm.SVR(kernel = 'poly')
clf1.fit(X_train,y_train)
accuracy_regression = clf1.score(X_test,y_test)

clf2.fit(X_train,y_train)
accuracy_svm = clf2.score(X_test,y_test)

clf3.fit(X_train,y_train)
accuracy_svm_poly = clf3.score(X_test,y_test)


print(accuracy_regression," regression")
print(accuracy_svm," svm")
print(accuracy_svm_poly," svm_poly")


#clf.predict("value or array")



