#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import matplotlib
import numpy as np
import pandas as pd
import pyreadr

import os


# In[2]:


path = '../../data/sequence-level/'


# In[114]:


data = pyreadr.read_r('../../data/sequences_all_anon.Rds')[None]


# In[115]:


data.rename(columns={'datum':'date', 'value':'category_name', 'anon_apps.name':'app_name'}, inplace=True)
data['timestamp'] = data['date'].apply(lambda x: x.timestamp())
data['category_name'].replace(['OFF_LOCKED', 'OFF_UNLOCKED'], 'OFF', inplace=True)
data['category_name'].replace(['ON_LOCKED', 'ON_UNLOCKED'], 'ON', inplace=True)
data['sessionID'] = data['category_name'].shift(1).isin(['OFF']).cumsum() + 1 # sessionID is like sequence_number but does NOT start anew for each user


# In[116]:


# cat_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['category_name'])))])
# cat_indexes = [cat_mapping[x] for x in data['category_name']]

user_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['userId'])))])
user_indexes = [user_mapping[x] for x in data['userId']]

# data['category'] = cat_indexes
data.insert(0, 'userID', user_indexes)


# In[117]:


# a = data.groupby(['sessionID'])['category_name'].apply(','.join).reset_index()
# a.category_name.nunique()


# In[118]:


# combine consecutive events of same category into single event (by dropping all but the first event)
data = data[data.category_name != data.category_name.shift(+1)]


# In[119]:


# create columns containing event categories listed (string separated by commas) and as a set
data['category_list'] = data.groupby(['sessionID'])['category_name'].transform(','.join)


# In[120]:


seq_drop_onoff = True


# In[121]:


if seq_drop_onoff:
    
    # first drop all 'ON,OFF' sessions
    mask = data['category_list'] == 'ON,OFF'
    data = data[~mask]
    
    # then drop all 'ON' and 'OFF' items from remaining sessions
    def remove_from_string(string, to_remove):
        l = string.split(',')
        out = ",".join(list(filter(lambda x: (x not in to_remove), l)))
        return out
    
    data['category_list'] = data['category_list'].apply(lambda x: remove_from_string(x, ['ON', 'OFF']))
    
    filename = 'data_seq_drop_onoff'

else:
    
    filename = 'data_seq'


# In[122]:


# create columns containing event counts per sequence
data['seq_length'] = data['category_list'].apply(lambda x: len(x.split(','))) # all events in sequence
# add frequency of category lists and sets
data['seq_freq'] = data.groupby(['category_list'])['category_list'].transform(lambda x: x.count())/data['seq_length']
# add sequence duration
data['seq_duration'] = data.groupby(['sessionID'])['timestamp'].transform('last') - data.groupby(['sessionID'])['timestamp'].transform('first')

# data['category_set'] = data['category_list'].apply(lambda x: set(x.split(',')))
# data['category_set_count'] = data['category_set'].apply(lambda x: len(x)) # distinct events in sequence
# data['category_set_freq'] = data.groupby(['category_set'])['category_set'].transform(lambda x: str(x).count())/data['category_set_count']


# In[123]:


# create unique sequence ID (usID) in order to "tokenize" unique sequences
seq_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['category_list'])))])
seq_indexes = [seq_mapping[x] for x in data['category_list']]

data['usID'] = seq_indexes


# In[124]:


# seq_mapping['ON,OFF']


# In[125]:


data.to_csv(path + filename + '.csv', index=False)


# ### data filtering

# In[3]:


# filename = 'data_seq'
filename = 'data_seq'
data = pd.read_csv(path + filename + '.csv')


# ##### by sequence frequency

# In[191]:


for N in [1,2,5,10,20,50,100]:
    unique_items = data['category_list'][data['seq_freq']>=N].nunique()
    n = data['category_list'][data['seq_freq']>=N].shape[0]
    n_sessions = data[data['seq_freq']>=N].groupby(['sessionID']).count().shape[0]
    print('for "min seq freq" cutoff ' + str(N) + ': ' + str(unique_items) + ' unique sequences, a total of ' + str(n) + ' events, and a total of ' + str(n_sessions) + ' sessions')


# In[192]:


# for N in [10,20,50,100]:
#     unique_items = data['category_set'][data['category_set_freq']>=N].nunique()
#     n = data['category_set'][data['category_set_freq']>=N].shape[0]
#     n_sessions = data[data['category_set_freq']>=N].groupby('sessionID').count().shape[0]
#     print('for set cutoff ' + str(N) + ': ' + str(unique_items) + ' unique sets, a total of ' + str(n) + ' events, and a total of ' + str(n_sessions) + ' sessions')


# ##### by sequence length

# In[196]:


for N in [1,2,5,10,20,50,100]:
    unique_items = data['category_list'][data['seq_length']<=N].nunique()
    n = data['category_list'][data['seq_length']<=N].shape[0]
    n_sessions = data[data['seq_length']<=N].groupby(['sessionID']).count().shape[0]
    print('for "max seq length" cutoff ' + str(N) + ': ' + str(unique_items) + ' unique sequences, a total of ' + str(n) + ' events, and a total of ' + str(n_sessions) + ' sessions')


# ##### by sequence duration

# In[253]:


for N in [600,1200,1800,2700,3600]:
    unique_items = data['category_list'][data['seq_duration']<=N].nunique()
    n = data['category_list'][data['seq_duration']<=N].shape[0]
    n_sessions = data[data['seq_duration']<=N].groupby(['sessionID']).count().shape[0]
    print('for "max seq duration" cutoff ' + str(N) + ' seconds: ' + str(unique_items) + ' unique sequences, a total of ' + str(n) + ' events, and a total of ' + str(n_sessions) + ' sessions')


# ##### by a combination of the above

# In[6]:


N1 = 10
N2 = 10
N3 = 2700


# In[8]:


mask1 = data['seq_freq']>=N1
mask2 = data['seq_length']<=N2
mask3 = data['seq_duration']<=N3
mask = mask1 & mask2 & mask3


# In[17]:


unique_items = data['category_list'][mask].nunique()
n = data['category_list'][mask].shape[0]
n_sessions = data[mask].groupby(['sessionID']).count().shape[0]


# In[20]:


print('for "combined" cutoff: '  + str(unique_items) + ' unique sequences, a total of ' + str(n) + ' events, and a total of ' + str(n_sessions) + ' sessions')


# ##### filtering

# In[4]:


# filter out all sequences with global frequency < 10
N = 10
# keep only the first row of each sequence
data = data[data['seq_freq']>=N].groupby(['sessionID']).first()


# ### sentence creation

# In[5]:


start_day = pd.to_datetime(data.timestamp.min(), unit='s').date()
end_day = pd.to_datetime(data.timestamp.max(), unit='s').date()
day_range = pd.date_range(start_day, end_day, freq='D')


# In[6]:


# helper list (same length as data) containing the day
user_day = data['userID'].astype(str).str.zfill(3) + '_' + pd.to_datetime(data['timestamp'], unit='s').apply(lambda x: x.date()).astype(str)


# In[7]:


sentence_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(user_day)))])
sentence_indexes = [sentence_mapping[x] for x in user_day]

# data['category'] = cat_indexes
data['sentenceID'] = sentence_indexes


# ##### dropping irrelevant columns

# In[109]:


data.drop(['userId', 'date', 'activity', 'category_name', 'sequence_number', 'app_name', 'category_list', 'seq_length', 'seq_freq', 'seq_duration'], axis=1, inplace=True)


# In[111]:


data.to_csv(path + filename + '_final' + '.csv', index=False)

