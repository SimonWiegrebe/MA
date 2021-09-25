#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr

import os


# In[2]:


data = pd.read_csv('../../data/sequence-level/data_seq.csv')


# In[7]:


data_final = pd.read_csv('../../data/sequence-level/data_seq_final.csv')


# In[20]:


# percentage of sequences of minimum length 20
data_final.groupby('sentenceID').filter(lambda x: len(x) >= 20).sentenceID.nunique()/data_final.sentenceID.nunique()


# In[25]:


min_item_support = 5
min_session_length = 2
min_user_sessions = 3
drop_first = True


# In[26]:


USER_KEY = 'userID'
TIME_KEY = 'timestamp'
ITEM_KEY = 'usID'
SESSION_KEY = 'sentenceID'


# In[27]:


N = 10
data = data[data['seq_freq']>=N].groupby(['sessionID']).first()


# In[21]:


start_day = pd.to_datetime(data.timestamp.min(), unit='s').date()
end_day = pd.to_datetime(data.timestamp.max(), unit='s').date()
day_range = pd.date_range(start_day, end_day, freq='D')

# helper list (same length as data) containing the day
user_day = data['userID'].astype(str).str.zfill(3) + '_' + pd.to_datetime(data['timestamp'], unit='s').apply(lambda x: x.date()).astype(str)

sentence_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(user_day)))])
sentence_indexes = [sentence_mapping[x] for x in user_day]

data['sentenceID'] = sentence_indexes
data['sentence_freq'] = data.groupby([SESSION_KEY])[SESSION_KEY].transform('size')

session_lengths = data.groupby('sentenceID').sentenceID.count()
q1_eps = np.quantile(session_lengths.values, 0.25)
median_eps = np.quantile(session_lengths.values, 0.50)
q3_eps = np.quantile(session_lengths.values, 0.75)


# In[22]:


print('For sequence-level analysis, user sequences are tokens (events) and daily concatenations thereof are sentences (sessions).')
print('For sequence-level analysis, only tokens with frequency of at least ' + str(N) + ' were included')
print('The sequence-level data contains a total of:')
print('     - ' + str(data[SESSION_KEY].nunique()) + ' sentences')
print('     - ' + str(sum(data.sentence_freq==1)) + ' sentences with only one token (excluded during data split due to minimum session length of 2)')
print('     - ' + str(data.shape[0]) + ' tokens')
print('     - ' + str(data.usID.nunique()) + ' unique tokens')
print('1st quartile of events per session: ' + str(round(q1_eps, 2)))
print('median number of events per session: ' + str(round(median_eps, 2)))
print('3rd quartile of events per session: ' + str(round(q3_eps, 2)))


# In[20]:


data.groupby('sentenceID').sentenceID.count()


# In[40]:


print('For sequence-level analysis, we have: ')
for i in [1,2,3,4,5,6,7,8,9,10,15,20,50,100,200,500]:
    print('     - ' + str(sum(data.sentence_freq>=i)) + ' sentences of length >= ' + str(i))
print('Clearly, most of the sentences are not short. Therefore, restricting ourselves to sentences of a certain minimum length, say, 20, would not have much of an impact.')
print('Based on this, language modeling techniques should work reasonably well.')
print('Furthermore, we might expect LM techniques to perform rather well on predicting tokens in higher positions.')

