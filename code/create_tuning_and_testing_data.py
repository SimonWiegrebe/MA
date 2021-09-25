#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import datetime
import matplotlib
import numpy as np
import pandas as pd
import os


# In[2]:


os.chdir('C:\\Users\\Simon\\Desktop\\MA\\session-rec')


# In[7]:


exclude_onoff = False
min20 = False # only for app-level
min20_test = True # only for app-level
# datatypes = ['app-level', 'sequence-level']
# datatypes = ['sequence-level']
datatypes = ['app-level'] # for min20 analysis: only app-level
windows = [1,2,3,4,5]
# windows = 'single'

USER_KEY = 'userID'
TIME_KEY = 'timestamp'


# In[73]:


# helper function to only include test sessions with minimum length 20
def only_min20(df):
    df = df.groupby(SESSION_KEY).filter(lambda x: len(x) >= 20)
    return df


# In[74]:


for datatype in datatypes:
    if datatype == 'app-level':
        ITEM_KEY = 'appID'
        SESSION_KEY = 'sessionID'
        onoff_key = 'drop_on-drop_off'
        min20_key = 'min20'
        min20_test_key = 'min20_test'
    else:
        ITEM_KEY = 'usID'
        SESSION_KEY = 'sentenceID'
        onoff_key = 'seq_drop_onoff_all'
    
    if windows == 'single':
        path_in = '../data/' + str(datatype) + '/single/'
        file_in = 'events'
        if min20:
            file_in += '-' + str(min20_key)
        if exclude_onoff:
            file_in += '-' + str(onoff_key) 
        file_in += '.hdf'

        file_out = 'single'
        if min20:
            file_out += '-' + str(min20_key)
        if min20_test:
            file_out += '-' + str(min20_test_key)
        file_out += '.hdf'

        if min20_test:
            path_out = 'data/testing'
            if exclude_onoff:
                path_out += '_onoff'
            path_out += '/' + str(datatype) + '/single/'
            dataset_test = only_min20(pd.read_hdf(path_in + file_in, 'test'))
            dataset_train = pd.read_hdf(path_in + file_in, 'train') 
            dataset_test.to_hdf(path_out + file_out, key='test', mode='w')
            dataset_train.to_hdf(path_out + file_out, key='train', mode='a')
        else:            
            for split in ['test', 'train']:
                path_out = 'data/testing'
                if exclude_onoff:
                    path_out += '_onoff'
                path_out += '/' + str(datatype) + '/single/'
                dataset = pd.read_hdf(path_in + file_in, split)
                if split == 'test': mode = 'w' # create new file for 'test', which is the first split (to avoid adding to existing file)
                else: mode = 'a' # append all other splits
                dataset.to_hdf(path_out + file_out, key=split, mode=mode)        

    else:    
        for window in windows:    
            path_in = '../data/' + str(datatype) + '/multiple/'
            file_in = 'events-' + str(window)
            if min20:
                file_in += '-' + str(min20_key)
            if exclude_onoff:
                file_in += '-' + str(onoff_key) 
            file_in += '.hdf'

            file_out = 'window_' + str(window)
            if min20:
                file_out += '-' + str(min20_key)
            if min20_test:
                file_out += '-' + str(min20_test_key)
            file_out += '.hdf'

            if min20_test: # no tuning, only testing
                path_out = 'data/testing'
                if exclude_onoff:
                    path_out += '_onoff'
                path_out += '/' + str(datatype) + '/multiple/'
                dataset_test = only_min20(pd.read_hdf(path_in + file_in, 'test'))
                dataset_train = pd.read_hdf(path_in + file_in, 'train') 
                dataset_test.to_hdf(path_out + file_out, key='test', mode='w')
                dataset_train.to_hdf(path_out + file_out, key='train', mode='a')
            else:          
                for split in ['valid_test', 'valid_train']:
                    path_out = 'data/tuning'
                    if exclude_onoff:
                        path_out += '_onoff'
                    path_out += '/' + str(datatype) + '/multiple/'
                    dataset = pd.read_hdf(path_in + file_in, split)
                    if split == 'valid_test': mode = 'w' # create new file for 'valid_test', which is the first split (to avoid adding to existing file)
                    else: mode = 'a' # append all other splits
                    new_split = split.split('_')[1]
                    dataset.to_hdf(path_out + file_out, key=new_split, mode=mode)

                for split in ['test', 'train']:
                    path_out = 'data/testing'
                    if exclude_onoff:
                        path_out += '_onoff'
                    path_out += '/' + str(datatype) + '/multiple/'
                    dataset = pd.read_hdf(path_in + file_in, split)
                    if split == 'test': mode = 'w' # create new file for 'test', which is the first split (to avoid adding to existing file)
                    else: mode = 'a' # append all other splits
                    dataset.to_hdf(path_out + file_out, key=split, mode=mode)

