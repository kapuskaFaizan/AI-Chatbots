from gensim.models import Word2Vec, KeyedVectors
import pandas as pd 
import nltk

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary =True,limit =10000)

model.wv.most_similar('death')

vec = model.wv['marriage'] + model.wv['love']

model.wv.most_similar([vec])

print(model.vocab)
