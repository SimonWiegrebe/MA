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


data_sa = pd.read_csv('../../data/app-level/data_app_nodrop.csv')


# In[6]:


min_item_support = 5
min_session_length = 2
min_user_sessions = 3
drop_first = True


# In[7]:


USER_KEY = 'userID'
ITEM_KEY = 'appID'
TIME_KEY = 'timestamp'
SESSION_KEY = 'sessionID'


# In[22]:


# def preprocess(df, min_item_support=5, min_session_length=2, min_user_sessions=3, drop_first = False):
#     '''
#     Preprocesses the dataframe by filtering out infrequent items, short sessions, and users with few sessions
#     -----
#         df: Pandas dataframe
#             Must contain the following columns: USER_KEY; ITEM_KEY; TIME_KEY; SESSION_KEY
#         drop_first: boolean
#             whether the first item of each session should be dropped
#         min_item_support: integer
#             minimum number of occurrences of an item (app) across all users and sessions for an item to be included
#         min_session_length: integer
#             minimum length (number of items) of a session for a session to be included
#         min_user_sessions: integer
#             minimum number of sessions per user for a user to be included
#     '''
#     if drop_first:
#         mask = df['appID'].shift(-1).isin([1389, 1390]) # 1389="OFF_LOCKED", 1390="OFF_UNLOCKED"
#         df = df[~mask] # filter out the first item of each session, i.e., items PRECEDED by 1389 or 1390
#     df = df.groupby(ITEM_KEY).filter(lambda x: len(x) >= min_item_support)
#     if df.groupby(USER_KEY)[SESSION_KEY].size().min() < min_session_length:
#         df = df.groupby([USER_KEY, SESSION_KEY]).filter(lambda x: len(x) >= min_session_length)
#     if df.groupby(USER_KEY)[SESSION_KEY].max().min() < min_user_sessions:
#         df = df.groupby(USER_KEY).filter(lambda x: x[SESSION_KEY].max() >= min_user_sessions)
#     return df

# data_sa = preprocess(data_sa, min_item_support=5, min_session_length=2, min_user_sessions=3, drop_first=False)


# In[8]:


# statistics about time
beginning_first_session = datetime.datetime.fromtimestamp(data_sa.timestamp.min()).isoformat()
end_last_session = datetime.datetime.fromtimestamp(data_sa.timestamp.max()).isoformat()
# print(beginning_first_session)
# print(end_last_session)


# In[9]:


data_sa.sessionID.nunique()


# In[10]:


num_users = data_sa.userID.nunique()
num_events = len(data_sa.appID)
num_sessions = data_sa.sessionID.nunique()
num_apps = data_sa.appID.nunique()
sessions_per_user = num_sessions/num_users
session_lengths = data_sa.groupby('sessionID').sessionID.count()
mean_eps = num_events/num_sessions
q1_eps = np.quantile(session_lengths.values, 0.25)
median_eps = np.quantile(session_lengths.values, 0.50)
q3_eps = np.quantile(session_lengths.values, 0.75)
# unique apps per session: TBD


# In[11]:


print('number of events: ' + str(num_events))
print('number of sessions: ' + str(num_sessions))
print('sessions per user: ' + str(sessions_per_user))
print('mean number of events per session: ' + str(round(mean_eps, 2)))
print('1st quartile of events per session: ' + str(round(q1_eps, 2)))
print('median number of events per session: ' + str(round(median_eps, 2)))
print('3rd quartile of events per session: ' + str(round(q3_eps, 2)))


# In[15]:


# which percentage of sessions (RHS) contain >= than i (LHS) apps?
[(i+1, sum(session_lengths>i+1)/num_sessions) for i in range(1,20)]


# In[17]:


(328554+92904)/844296


# In[60]:


# session length (LHS) with corresponding absolute frequency (RHS)
from collections import Counter
Counter(session_lengths).most_common()


# In[38]:


# data_sa = preprocess(data_sa, min_item_support=5, min_session_length=2, min_user_sessions=3, drop_first=True)


# In[62]:


num_users = data_sa.userID.nunique()
num_events = len(data_sa.appID)
num_sessions = data_sa.sessionID.nunique()
num_apps = data_sa.appID.nunique()
sessions_per_user = num_sessions/num_users
session_lengths = data_sa.groupby('sessionID').sessionID.count()
mean_eps = num_events/num_sessions
q1_eps = np.quantile(session_lengths.values, 0.25)
median_eps = np.quantile(session_lengths.values, 0.50)
q3_eps = np.quantile(session_lengths.values, 0.75)
# unique apps per session: TBD


# In[63]:


print('number of events: ' + str(num_events))
print('number of sessions: ' + str(num_sessions))
print('sessions per user: ' + str(sessions_per_user))
print('mean number of events per session: ' + str(round(mean_eps, 2)))
print('1st quartile of events per session: ' + str(round(q1_eps, 2)))
print('median number of events per session: ' + str(round(median_eps, 2)))
print('3rd quartile of events per session: ' + str(round(q3_eps, 2)))


# ### Issues

# ##### single window versus multiple windows

# * we have few users (310) but many sessions per user (~2724)
# * only 310 test sessions in total, all of them at the end of January 2018
# * if splitting the observation time span in, say, 5 equally long parts:
#     * 5 train sets per user
#     * 5 test sets per user (we can average performance across the 5 sets)

# ##### very short sessions on average

# * our data: average sequence length: 5.11
# * BERT4Rec: dataset with shortest average sequence length: 8.8 (Amazon Beauty)
# * HGRU4Rec: dataset with shortest average sequence length: 6.1 (Xing)
# * comparison paper: dataset with shortest average sequence length: 5.62 (Xing)
# * only 8% of all sessions contain at least 10 apps
# * this gives rise to the so-called cold start problem for sequential prediction
# * very questionable whether BERT4Rec will perform well because of the above (and the below)

# ##### first and last app in each session are not really informative

# * each session has a single OFF_ event at the end (either OFF_LOCKED or OFF_UNLOCKED) and no OFF_ event prior to that
# * each session has an ON_ event at the beginning (either ON_LOCKED or ON_UNLOCKED)

# * there are 162805 sessions starting with 2 (or more) consecutive ON_ events
#     * most of them starting with ON_LOCKED followed by ON_UNLOCKED
#     * 25 sessions starting with ON_UNLOCKED followed by ON_LOCKED (how does that make sense?)

# In[ ]:


on_2 = []
for i in range(1, len(data_sa)):
    if (data_sa.appID[i-1] in [1389, 1390]) and (data_sa.appID[i] in [1392, 1393]):
        if data_sa.appID[i+1] in [1392, 1393]:
            on_2.append(i)


# In[40]:


# example of a session which starts with ON_LOCKED followed by ON_UNLOCKED
i = on_2[0]
data_sa[i-1:i+5]


# In[37]:


on_unlocked_locked = []
for i in range(1, len(data_sa)):
    if (data_sa.appID[i-1] in [1389, 1390]) and (data_sa.appID[i]==1393):
        if data_sa.appID[i+1]==1392:
            on_unlocked_locked.append(i)


# In[41]:


# example of a session which starts with ON_UNLOCKED followed by ON_LOCKED
i = on_unlocked_locked[0]
data_sa[i-1:i+5]


# * there are sessions starting with ON_LOCKED, followed by some other apps, followed by ON_UNLOCKED

# In[31]:


on_other_on = []
for i in range(1, len(data_sa)):
    if (data_sa.appID[i-1] in [1389, 1390]) and (data_sa.appID[i] in [1392, 1393]):
        if (data_sa.appID[i+1] not in [1389, 1390, 1392, 1393]) and (data_sa.appID[i+2] in [1392, 1393]):
            on_other_on.append(i)


# In[42]:


# example of a session which starts with ON_LOCKED followed by some other app followed by ON_UNLOCKED
i = on_other_on[0]
data_sa[i-1:i+5]


# * if we decide to drop the first event of each session (ON_):
#     * not really consistent content-wise (b/c of the multiple ON_ events)
#     * we lose all sessions of length 2, i.e., 39% (328535/844296) of all sessions
#     * average sequence length increases to 6.09
#     * no major performance changes at first sight
# * if we decide to drop the last event of each session (OFF_):
#     * we lose all sessions of length 2 (as above)
#     * no longer able to predict the length of each session (yet this would be an entirely different task anyway)
#     * recall: we iteratively predict apps for the next spot in a session, yet we never actively predict whether there will be a next spot
#     * on the other hand, we would not learn how to predict an OFF_ event
#     * i.e., we would not learn which sequence is likely to indicate that a session has come to its end

# ### To do

# * Tuning:
#     * all data or only a single window (i.e., a subset)?
#     * 50 or 100 optimization iterations?
# * BERT4Rec: how much time to invest into it?
#     * run with our data
#     * no ad hoc performance comparison possible (only last item evaluated)
#     * extract "predictions"
#     * evaluate predictions using the comparison framework

# Alternative approaches to encode app sequences:
# * word2vec
# * glove
# 
# But these models only generate app embeddings and do not perform prediction.

# ### unique sessions

# ##### using non-categorized appID

# In[39]:


data_sa['app_string'] = data_sa.appID.astype(str)
a = data_sa.groupby(['userID', 'sessionID'])['app_string'].apply(','.join).reset_index()


# In[40]:


a


# In[41]:


a.app_string.nunique()


# ##### using categorized appID

# In[42]:


data_raw = pyreadr.read_r('../../data/sequences_all_anon.Rds')[None]
data_raw = data_raw.rename(columns={'datum':'date', 'value':'category', 'anon_apps.name':'app_name'})
data_raw['timestamp'] = data_raw['date'].apply(lambda x: x.timestamp())
data_raw.loc[data_raw['app_name'].isnull(),'app_name'] = data_raw['category'] # replace NaNs in app_name by corresponding category value
data_raw['sessionID'] = data_raw['app_name'].shift(1).isin(['OFF_LOCKED','OFF_UNLOCKED']).cumsum() + 1 # sessionID is like sequence_number but does NOT start anew for each


# In[43]:


data_sa = data_raw
app_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data_sa['app_name'])))])
app_indexes = [app_mapping[x] for x in data_sa['app_name']]
# print(len(set(app_indexes)) == data_sa['app_name'].nunique()) # check

user_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data_sa['userId'])))])
user_indexes = [user_mapping[x] for x in data_sa['userId']]
# print(len(set(user_indexes)) == data_sa['userId'].nunique()) # check

data_sa['appID'] = app_indexes
data_sa.insert(0, 'userID', user_indexes)

data_sa = data_sa.drop(['userId', 'date', 'activity', 'sequence_number', 'app_name'], axis=1)


# In[44]:


data_sa['category'].loc[data_sa['category'] == 'OFF_LOCKED'] = 'OFF'
data_sa['category'].loc[data_sa['category'] == 'OFF_UNLOCKED'] = 'OFF'
data_sa['category'].loc[data_sa['category'] == 'ON_LOCKED'] = 'ON'
data_sa['category'].loc[data_sa['category'] == 'ON_UNLOCKED'] = 'ON'


# In[45]:


b = data_sa.groupby(['userID', 'sessionID'])['category'].apply(','.join).reset_index()


# In[47]:


b


# In[46]:


b.category.nunique()


# ### most popular app

# In[ ]:


###### overall


###### excluding ON_ and OFF_

