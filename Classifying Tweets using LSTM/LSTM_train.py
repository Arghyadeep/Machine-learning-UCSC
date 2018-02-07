#dependencies
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from gensim.models import Word2Vec as w2v
import re
import nltk

#tokenize tweets
def tokenize(tweets):
    for j in range(len(tweets)):
        tweets[j]=nltk.word_tokenize(tweets[j])
    return(tweets)

#create the embedding matrix
def create_matrix(tweets,vector_size):
    words = []
    for tweet in tweets:
        for word in tweet:
            words.append(word)
    matrix = np.random.rand(len(set(words)),vector_size)
    return(matrix, list(set(words)))

def vectorize(tweets):    
    #convert the Hillary and trump labels to 0 and 1
    Y = []
    for i in tweets:
        if i == "Hillary cLinton":
            Y.append(1)
        else:
            Y.append(0)
    Y = np.array(Y)
    return(Y)

#create the dictionary
def create_dict(words):
    wordlist = {}
    j = 1
    for i in words:
        wordlist[i] = j
        j += 1
    return(wordlist)

#create sentence vectors with padding
def sentences2vector(dictionary,tweets):
    maximum_len = max([len(sents) for sents in tweets])
    padded_sents = []
    for sentences in tweets:
        padded_words = [0]*maximum_len
        for i in range(len(sentences)):
            padded_words[i] = (dictionary[sentences[i]])
        padded_sents.append(np.array(padded_words))
    return(padded_sents)

def run():
    train_data = np.array(pd.read_csv("train.csv"))
    labels = train_data[:,0]
    tweets = train_data[:,1]

    #removing urls from tweets
    for i in range(len(tweets)):
        tweets[i] = re.sub(r"http\S+", "", tweets[i])
        
    tokenized_tweets = tokenize(tweets)
    
    vector_size = 64 #change according to performance
    embedded_matrix, tokenized_words = create_matrix(tokenized_tweets,vector_size)
    
    length = len(tokenized_words) #number of tokenized words
    
    
    word_dict = create_dict(tokenized_words)
    padded_sents = sentences2vector(word_dict,tokenized_tweets)
    
def RNN():
    batchSize = 24
    lstmUnits = 64
    numClasses = 2
    iterations = 100000
    maxSeqlength = 45

    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)
    

if __name__ == '__main__':
    run()
    RNN()
