#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import datetime
import matplotlib
import numpy as np
import pandas as pd
import pyreadr
import os


# ##### setup

# In[2]:


os.chdir('C:\\Users\\Simon\\Desktop\\MA\\session-rec')


# In[3]:


# load data

# inpath = '../../data/app-level/'
# filename = 'data_app'
# data = pd.read_csv(inpath + filename + '.csv')

inpath = '../data/app-level/'
filename = 'data_app'
data = pd.read_csv(inpath + filename + '.csv')


# In[4]:


# assign keys
USER_KEY = 'userID'
TIME_KEY = 'timestamp'
ITEM_KEY = 'appID'
SESSION_KEY = 'sessionID'
# ITEM_KEY = 'usID'
# SESSION_KEY = 'sentenceID'


# ##### workhorse functions

# In[5]:


def preprocess(df, min_item_support=5, min_session_length=2, min_user_sessions=3,
               drop_on=False, drop_off=False, drop_first=False, seq_drop_onoff=False):
    '''
    Preprocesses the dataframe by filtering out infrequent items, short sessions, and users with few sessions
    -----
        df: Pandas dataframe
            Must contain the following columns: USER_KEY; ITEM_KEY; TIME_KEY; SESSION_KEY
        drop_first: boolean
            whether the first item of each session should be dropped
        min_item_support: integer
            minimum number of occurrences of an item (app) across all users and sessions for an item to be included
        min_session_length: integer
            minimum length (number of items) of a session for a session to be included
        min_user_sessions: integer
            minimum number of sessions per user for a user to be included
    '''
    if drop_first:
        mask = df[ITEM_KEY].shift(-1).isin([1389, 1390]) # 1389="OFF_LOCKED", 1390="OFF_UNLOCKED"
        df = df[~mask] # filter out the first item of each session, i.e., items PRECEDED by 1389 or 1390
    if drop_on:
        mask = df[ITEM_KEY].isin([1392, 1393])
        df = df[~mask]
    if drop_off:
        mask = df[ITEM_KEY].isin([1389, 1390])
        df = df[~mask]
    if seq_drop_onoff:
        mask = df[ITEM_KEY]==76202
        df = df[~mask]
    # min_item_support
    df = df.groupby(ITEM_KEY).filter(lambda x: len(x) >= min_item_support)
    # min_session_length
    if df.groupby(SESSION_KEY)[SESSION_KEY].size().min() < min_session_length:
        df = df.groupby(SESSION_KEY).filter(lambda x: len(x) >= min_session_length)
    # min_user_sessions
    user_sessions = df.groupby([USER_KEY])[SESSION_KEY].nunique()
    mask = df[USER_KEY].apply(lambda x: user_sessions[x]) >= min_user_sessions
    df = df[mask]
    
    return df


# In[6]:


def split_last_session(df):
    '''
    Splits off the last session of a sequence of sessions for each user
    -----
        df: Pandas dataframe
            Must contain the following columns: USER_KEY; ITEM_KEY; TIME_KEY; SESSION_KEY
    '''
    last_sessions = df[SESSION_KEY].groupby(df[USER_KEY]).transform('last')
    train = df[df[SESSION_KEY]!=last_sessions]
    test = df[df[SESSION_KEY]==last_sessions]
    
    return (train, test)


# In[7]:


def filter_new_items(train, test):
    '''
    Filters out observations from a test set which do not appear in the corresponding training set
    -----
        train: Pandas dataframe
            Training set; must contain the following columns: USER_KEY; ITEM_KEY; TIME_KEY; SESSION_KEY
        test: Pandas dataframe
            Test set; must contain the following columns: USER_KEY; ITEM_KEY; TIME_KEY; SESSION_KEY
    '''
    test = test[test[ITEM_KEY].isin(train[ITEM_KEY].unique())]
    return test


# In[8]:


# combine all the above functions
def split_data(df,
               min_item_support, min_session_length, min_user_sessions,
               USER_KEY, ITEM_KEY, TIME_KEY, SESSION_KEY,
               drop_on=False, drop_off=False, drop_first=False, seq_drop_onoff=False):
    df_preprocessed = preprocess(df,
                                 min_item_support=min_item_support, min_session_length=min_session_length, min_user_sessions=min_user_sessions,
                                 drop_on=drop_on, drop_off=drop_off, drop_first=drop_first, seq_drop_onoff=seq_drop_onoff)
    train, test = split_last_session(df_preprocessed)
    valid_train, valid_test = split_last_session(train)
    test = filter_new_items(train, test)
    valid_test = filter_new_items(valid_train, valid_test)
    return (train, valid_train, valid_test, test)


# ##### helper function for multiple windows

# In[9]:


# assign a single item to a window (from 1,...,win) based on timestamp of first item of current session
def assign_window(timestamp, cutoff_list):
    num_windows = len(cutoff_list)
    for i in range(num_windows):
        if timestamp <= cutoff_list[i]:
            window = i+1
            break
    return window


# ##### apply preprocessing and splitting

# In[13]:


min_item_support = 5
min_session_length = 20
min_user_sessions = 3

drop_on = False
drop_off = False
drop_first = False # should always be set to False if drop_on=True
seq_drop_onoff = False # flag for sequence-level analysis to drop 'ON,OFF' sequences (tokens); set to False if filename is 'data_seq_drop_onoff_final'

multiple_windows = False # flag for multiple windows
win = 5 # only needed if multiple_windows=True

outpath = '../data/app-level/multiple/'


# In[15]:


if multiple_windows:
    
    ts_min = data.timestamp.min()
    ts_max = data.timestamp.max()
    win_timespan = (ts_max-ts_min)/win
    win_cutoffs = [ts_min+(i+1)*win_timespan for i in range(win)]
    
    # create new column containing timestamp from first item of each session for each item of the session
    data['window'] = data['timestamp'].groupby(data[SESSION_KEY]).transform('first')

    # based on timestamp from first item, assign the entire session to one of the win windows
    # to do so, apply assign_window to entire column "window"
    # this way, we never split up sessions
    data['window'] = data['window'].apply(lambda x: assign_window(x, win_cutoffs))
    
    for i in range(win):
        name = 'events' + '-' + str(i+1) # set up dataset name, e.g., data_1 corresponding to windows 1
        df = data[data.window==i+1].drop('window',axis=1) # choose one single window only
        train, valid_train, valid_test, test = split_data(df,
                                                         min_item_support, min_session_length, min_user_sessions,
                                                         USER_KEY, ITEM_KEY, TIME_KEY, SESSION_KEY,
                                                         drop_on=drop_on, drop_off=drop_off, drop_first=drop_first,
                                                         seq_drop_onoff=seq_drop_onoff)
        # save output to hdf files
        if min_session_length > 2:
            outname = outpath + str(name) + '-min' + str(min_session_length)
        else:
            outname = outpath + str(name)
        if drop_on:
            outname += '-drop_on'
        if drop_off:
            outname += '-drop_off'
        if drop_first:
            outname += '-drop_first'
        if seq_drop_onoff:
            outname += '-seq_drop_onoff'
        if filename == 'data_seq_drop_onoff_final':
            outname += '-seq_drop_onoff_all'
        outname += '.hdf'
        
        train.to_hdf(outname, key='train', mode='w') # create new file (to avoid adding to existing file)
        valid_test.to_hdf(outname, key='valid_test', mode='a')
        valid_train.to_hdf(outname, key='valid_train', mode='a')
        test.to_hdf(outname, key='test', mode='a')
        
else:
    train, valid_train, valid_test, test = split_data(data,
                                                      min_item_support, min_session_length, min_user_sessions,
                                                      USER_KEY, ITEM_KEY, TIME_KEY, SESSION_KEY,
                                                      drop_on=drop_on, drop_off=drop_off, drop_first=drop_first,
                                                      seq_drop_onoff=seq_drop_onoff)
    if min_session_length > 2:
        outname = outpath + 'events' + '-min' + str(min_session_length)
    else:
        outname = outpath + 'events'
    if drop_on:
        outname += '-drop_on'
    if drop_off:
        outname += '-drop_off'
    if drop_first:
        outname += '-drop_first'
    if seq_drop_onoff:
        outname += '-seq_drop_onoff'
    if filename == 'data_seq_drop_onoff_final':
        outname += '-seq_drop_onoff_all'
    outname += '.hdf'
    
    test.to_hdf(outname, key='test', mode='w') # create new file (to avoid adding to existing file)
    train.to_hdf(outname, key='train', mode='a')
    valid_test.to_hdf(outname, key='valid_test', mode='a')
    valid_train.to_hdf(outname, key='valid_train', mode='a')

