#dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

#reading train or test data
def read_data(raw_data):
    train_data = open(raw_data,"r").readlines()
    y = []
    for i in range(len(train_data)):
        y.append(int(train_data[i].split(",")[0]))

    X = []
    for i in range(len(train_data)):
        temp = list(map(int,(train_data[i].split(",")[1:])))
        X.append(temp)
        
    X_columns = ['X'+str(i+1) for i in range(len(X[0]))]
    columns = X_columns.append('y')

    df = pd.DataFrame(columns = columns)
    df['y'] = y
    for i in range(len(X[0])):
        temp = []
        for j in X:
            temp.append(j[i])
        df['X'+str(i+1)] = temp
            
    print(len(df))
    return df

#perfform kfolds cross validation on train set
def K_folds_cross_validation(kfolds,lambda_num):
    df = read_data("train.txt")
    
    #shuffling data
    df = df.sample(frac=1).reset_index(drop=True)
    
    #getting X and y values for cross validation
    X_ = np.array(df.drop(['y'],1))
    y_ = np.array(df['y'])
    split = int(len(X_,)/kfolds)
    
    #getting lambda values in logspace
    lambdas = list(1/np.logspace(-2,1,num=lambda_num,endpoint = True,base =10)) 
    
    crossval_scores = []
    for curr_lambda in lambdas:
        temp = []
        for i in range(kfolds):
            X_test = X_[split*i:split+split*i]
            Y_test = y_[split*i:split+split*i]
            X_train = np.delete(X_,[i for i in range(split*i,split+split*i)],0)
            Y_train = np.delete(y_,[i for i in range(split*i,split+split*i)],0)

            #defining the classifier for current lambda value
            logreg = LogisticRegression(penalty = 'l1', C = curr_lambda,fit_intercept=True,
                                        max_iter=100, solver = 'liblinear')
            
            temp.append(logreg.fit(X_train,Y_train).score(X_test,Y_test))
        crossval_scores.append(np.mean(temp))

    return crossval_scores,lambdas

#plotting for best lambda value
def plot_lambda_crossval_scores(k,num):
    crossval_scores, lambdas = K_folds_cross_validation(k,num)
    errors = [(1-i) for i in crossval_scores]
    mod_lambdas = [1/i for i in lambdas]
    print(min(errors))
    print(mod_lambdas[errors.index(min(errors))])
    plt.plot(mod_lambdas,errors)
    plt.show()

#plotting for different holdouts
def plot_for_holdouts(kvalues):
    max_errors = []
    min_errors = []
    mean_errors = []
    
    for k in kvalues:
        crossval_scores, lambdas = K_folds_cross_validation(k,50)
        error = [(1-i) for i in crossval_scores]
        max_errors.append(max(error))
        min_errors.append(min(error))
        mean_errors.append(np.mean(error))

    return max_errors, min_errors, mean_errors

#finding test set accuracy
def test_accuracy(train_data,test_data):
    #getting test and train data
    train_data = read_data(train_data)
    test_data = read_data(test_data)
    X_train = np.array(train_data.drop(['y'],1))
    y_train = np.array(train_data['y']) 
    X_test = np.array(test_data.drop(['y'],1))
    y_test = np.array(test_data['y'])

    #getting the best regularization parameter from k fold cross validation
    crossval_scores, lambdas = K_folds_cross_validation(1999,10)
    best_crossval_score = max(crossval_scores)
    best_lambda = lambdas[crossval_scores.index(best_crossval_score)]

    #fitting test data to classifier for accuracy
    logreg = LogisticRegression(penalty = 'l1', C = best_lambda,fit_intercept=True,
                                max_iter=100, solver = 'liblinear',intercept_scaling=1000)
    model = logreg.fit(X_train,y_train)
    accuracy = model.score(X_test,y_test)
    print(accuracy,"accuracy")
    res = list(model.predict(X_test))
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(res)):
        if list(y_test)[i] == -1 and res[i] == -1:
            TN += 1
        if list(y_test)[i] == 1 and res[i] == 1:
            TP += 1
        if list(y_test)[i] == 1 and res[i] == -1:
            FN += 1
        if list(y_test)[i] == -1 and res[i] == 1:
            FP += 1

    print(TN,TP,FN,FP)

#crossval_scores,lambdas = K_folds_cross_validation(1999,10)        
#test_accuracy("train.txt","test.txt")
#plot_lambda_crossval_scores(10,1000)
#read_data("train.txt")
maxm,minm,mean = plot_for_holdouts([3,5,7,10])
k = [3,5,7,10]
plt.errorbar(k, mean, yerr = [maxm[i]-minm[i] for i in range(len(k))], fmt = 'o',
             ecolor = 'g', capthick = 2)
