import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model
from gensim.models import Word2Vec, KeyedVectors


import theano
theano.config.optimizer="None"


model=load_model('LSTM100.h5')
mod = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary =True,limit =10000)
while(True):
    x= input("Enter the message:")
    sentend=np.ones((300,),dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.vocab]

    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend) 
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist=[mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    output=' '.join(outputlist)
    print (output)
