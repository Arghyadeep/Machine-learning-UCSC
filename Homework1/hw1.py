#dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

#reading data from file
data = pd.read_csv("train.csv")
X = data['X'].tolist()
y = data['y'].tolist()

#making X features in dataframe
data_x_features = []
columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9']

#creating data features i.e. powers of x
for i in range(len(columns)):
    temp = [X[j]**i for j in range(len(X))]
    data_x_features.append(temp)

#inserting all data in dataframe
df = pd.DataFrame(columns = columns)
for i in range(len(columns)):
    df[columns[i]] = data_x_features[i]
    
#converting y values to classes
mean = np.mean(y)
y = [1 if i >= mean else 0  for i in y]
df['y'] = y

X_train = np.array(df.drop(['y'],1))
y_train = np.array(df['y'])

#fitting data into model

model = LogisticRegressionCV(Cs = 10, fit_intercept = False, cv = 10,
                             penalty = 'l2', solver = 'liblinear', max_iter =100,
                             refit =True)
fit1 = model.fit(X_train,y_train)
#print(fit1.coef_)
#print(fit1.scores_)
#print(fit1.coefs_paths_[1][0])
#print(fit1.Cs_)
#print(fit1.C_)
score = (fit1.scores_)
c= fit1.Cs_
scores = []
for i in score[1]:
    scores.append(i[0])
print(scores)
plt.plot(c,scores)
plt.show()
