#!/usr/bin/env python
# coding: utf-8

# In[6]:


import datetime
import matplotlib
import numpy as np
import pandas as pd
import pyreadr

import os


# In[2]:


data = pyreadr.read_r('../../data/sequences_all_anon.Rds')[None]


# In[3]:


data.rename(columns={'datum':'date', 'value':'category', 'anon_apps.name':'app_name'}, inplace=True)
data['timestamp'] = data['date'].apply(lambda x: x.timestamp())
data.loc[data['app_name'].isnull(),'app_name'] = data['category'] # replace NaNs in app_name by corresponding category value
data['sessionID'] = data['app_name'].shift(1).isin(['OFF_LOCKED','OFF_UNLOCKED']).cumsum() + 1 # sessionID is like sequence_number but does NOT start anew for each user


# In[4]:


app_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['app_name'])))])
app_indexes = [app_mapping[x] for x in data['app_name']]
# print(len(set(app_indexes)) == data['app_name'].nunique()) # check

user_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['userId'])))])
user_indexes = [user_mapping[x] for x in data['userId']]
# print(len(set(user_indexes)) == data['userId'].nunique()) # check

data['appID'] = app_indexes
data.insert(0, 'userID', user_indexes)


# In[ ]:


# app_mapping_reverse = dict((v,k) for k,v in app_mapping.items())
# user_mapping_reverse = dict((v,k) for k,v in user_mapping.items())
# print(list(app_mapping.keys())[list(app_mapping.values()).index(1194)])
# print(app_mapping_reverse[1194])


# In[2]:


path = '../../data/app-level/'


# In[6]:


data.to_csv(path + 'data_app_nodrop.csv', index=False)


# In[7]:


data.drop(['userId', 'date', 'activity', 'category', 'sequence_number', 'app_name'], axis=1, inplace=True)


# In[8]:


data.to_csv(path + 'data_app.csv', index=False)


# In[7]:


data = pd.read_csv(path + 'data_app.csv')


# In[8]:


data

