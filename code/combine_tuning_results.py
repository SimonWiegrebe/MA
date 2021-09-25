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


# In[3]:


def read_single_file(file):
    df = pd.read_csv(file, sep=';')
    df.drop(df.filter(regex='Unnamed'), axis=1, inplace=True)
    return df


# In[10]:


exclude_onoff = False
if exclude_onoff:
    tuning_folder = 'tuning_onoff'
else:
    tuning_folder = 'tuning'
algos = ['gru4rec']
# datatypes = ['app-level', 'sequence-level']
datatypes = ['app-level']
windows = [1]
# windows = 'single'
budget = 100


# In[13]:


for datatype in datatypes:
    folder_in = 'results/' + str(tuning_folder) + '/' + str(datatype) + '/multiple_raw/'
    folder_out = 'results/' + str(tuning_folder) + '/' + str(datatype) + '/multiple/'
    for algo in algos:
        for window in windows:
            files_in = [folder_in + f for f in os.listdir(folder_in) 
                     if f.startswith('test_single_' + str(algo) + '_config_') and f.endswith('_window_' + str(window) + '.csv')]

            res = pd.DataFrame(np.nan, index=list(range(budget)),columns=['Metrics', 'HitRate@1: '])
            for i in range(budget):
                file = str(folder_in) + 'test_single_' + str(algo) + '_config_' + str(i) + '_window_' + str(window) + '.csv'
                if file in files_in:
                    df = read_single_file(file)
                    hr = df['HitRate@1: '][0]
                    res.iloc[i] = [df['Metrics'][0], df['HitRate@1: '][0]]
                    
#             res.drop_duplicates(inplace=True) # drop duplicate rows
            file_out = folder_out + 'test_single_' + str(algo) + '_window_' + str(window) + '.csv'
            res.to_csv(file_out, index=False, sep=';')

