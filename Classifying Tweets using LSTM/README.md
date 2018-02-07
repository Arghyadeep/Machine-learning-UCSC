#classifying tweets by Hillary Clinton versus Donald Trump. 
The base task is not to work with a bag of words model. Instead we used an LSTM recurrent Neural Net that takes the
sequence of words in each text into account.
The labeled training set is tokenized and trained using an LSTM network. We also provide an
unlabeled test set. The probabilities for predicting each class for all tweets in this test set are returned.
The logistic loss wrt the true (hidden) labels are measured.
