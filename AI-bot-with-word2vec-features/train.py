from gensim.models import Word2Vec, KeyedVectors
import pandas as pd 
import nltk
import numpy as np 
import pickle
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from nltk import word_tokenize

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary =True,limit =10000)

df= pd.read_csv('testfrom.txt',names = ['q'])
df_reply= pd.read_csv('testto.txt',names = ['reply'])
df['reply'] = df_reply['reply']


x = np.array(df['q'])
y = np.array(df['reply'])

tok_x=[]
tok_y=[]
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))


sentend = np.ones((300,),dtype=np.float32) 


vec_x=[]
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
    
vec_y=[]

for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)           
    
for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)    
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend) 
vec_xx=np.array(vec_x,dtype=np.float64)
vec_yy=np.array(vec_y,dtype=np.float64)  
x_train,x_test, y_train,y_test = train_test_split(vec_xx, vec_yy, test_size=0.2, random_state=1)
    
model=Sequential()
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=1000,validation_data=(x_test, y_test))
model.save('LSTM100.h5');

