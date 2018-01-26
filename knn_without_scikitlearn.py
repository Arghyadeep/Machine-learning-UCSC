import numpy as np
from math import sqrt
import random
import pandas as pd
import warnings
from collections import Counter

def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn("k is set to a lower value than total voting groups")
    distances = []
    #knn_algorithm
    for group in data:
        for features in data[group]:
            distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result

df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace("?",-99999,inplace= True)
df.drop(['id'],1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)


test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]


for i in train_data:
    #last value in the data is either 2 or 4
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    #last value in the data is either 2 or 4
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k =5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total)
            
