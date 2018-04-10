#dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

lambdas = list(1/np.logspace(-30,np.log(5),num=1000,endpoint = True,base =10))

#reading data from file
data = pd.read_csv("train.csv")
X = data['X'].tolist()
y = data['y'].tolist()
print(max(y),"maxy")
print(min(y),"miny")
print(np.mean(y),"meany")

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
df = df.sample(frac=1).reset_index(drop=True)
X_ = np.array(df.drop(['y'],1))
y_ = np.array(df['y'])

print(len(X_))
print(len(y_))
#cross validation
kfolds = 10
split = int(len(X_)/kfolds)


lambda_score_pair = []
for curr_lambda in lambdas:
    cross_val_scores=[]
    for i in range(kfolds):
        X_test = X_[split*i:split+split*i]
        Y_test = y_[split*i:split+split*i]
        X_train = np.delete(X_,[i for i in range(split*i,split+split*i)],0)
        Y_train = np.delete(y_,[i for i in range(split*i,split+split*i)],0)
        
        logreg = LogisticRegression(penalty = 'l1', C = curr_lambda,fit_intercept=True, max_iter=5000,
                                    solver = 'liblinear')
        
        cross_val_scores.append(logreg.fit(X_train,Y_train).score(X_test,Y_test))
    lambda_score_pair.append( np.mean(cross_val_scores))

print(max(lambda_score_pair),"maxl")
mod_lambda_score_pair = [(1-i) for i in lambda_score_pair]
print(min(mod_lambda_score_pair),"minerror")
print(max(mod_lambda_score_pair),"maxerror")
print(np.mean(mod_lambda_score_pair),"meanerror")
ind = (mod_lambda_score_pair.index(min(mod_lambda_score_pair)))
print(ind)
mod_lambdas = [1/i for i in lambdas]
#print(mod_lambdas)
plt.plot(mod_lambdas,mod_lambda_score_pair)
plt.show()

best_lambda = mod_lambdas[ind]
print(best_lambda,"best")
test_data = pd.read_csv('test.csv')
t_X = test_data['X'].tolist()
t_y = test_data['y'].tolist()
print(len(test_data),"test")
#making X features in dataframe
t_data_x_features = []
columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9']

#creating data features i.e. powers of x
for i in range(len(columns)):
    temp = [t_X[j]**i for j in range(len(t_X))]
    t_data_x_features.append(temp)

#inserting all data in dataframe
t_df = pd.DataFrame(columns = columns)
for i in range(len(columns)):
    t_df[columns[i]] = t_data_x_features[i]
    
#converting y values to classes
mean = np.mean(t_y)
t_y = [1 if i >= mean else 0  for i in t_y]
t_df['y'] = t_y
t_df = t_df.sample(frac=1).reset_index(drop=True)
t_X_ = np.array(t_df.drop(['y'],1))
t_y_ = np.array(t_df['y'])

logreg_t = LogisticRegression(penalty = 'l2', C = best_lambda,fit_intercept=True, max_iter=200,
                                    solver = 'liblinear')
fit_t = logreg_t.fit(X_train,Y_train)
print(fit_t.score(t_X_,t_y_))

res = list(fit_t.predict(t_X_))
TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(res)):
    if list(t_y_)[i] == 0 and res[i] == 0:
        TN += 1
    if list(t_y_)[i] == 1 and res[i] == 1:
        TP += 1
    if list(t_y_)[i] == 1 and res[i] == 0:
        FN += 1
    if list(t_y_)[i] == 0 and res[i] == 1:
        FP += 1

print(TP,TN,FP,FN)
print((FN+FP)/(FN+FP+TN+TP))
    

   


