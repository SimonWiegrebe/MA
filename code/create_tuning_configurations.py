#!/usr/bin/env python
# coding: utf-8

# In[45]:


import datetime
import matplotlib
import numpy as np
import pandas as pd
import pyreadr
import pickle

import os


# In[46]:


os.chdir('C:\\Users\\Simon\\Desktop\\MA\\session-rec')


# In[47]:


algo_classes = {
    'gru4rec': 'gru4rec.gru4rec.GRU4Rec',
    'gru4rec_Reminder': 'gru4rec.ugru4rec.UGRU4Rec',
    'hgru4rec': 'hgru4rec.hgru4rec.HGRU4Rec',
    'shan': 'shan.shan.SHAN',
    'sr': 'baselines.sr.SequentialRules',
    'sr_BR': 'baselines.usr.USequentialRules',
    'stan': 'knn.stan.STAN',
    'vstan': 'knn.vstan.VSKNN_STAN',
    'vstan_EBR': 'knn.uvstan.U_VSKNN_STAN',
    'sknn': 'knn.sknn.ContextKNN',
    'vsknn': 'knn.vsknn.VMContextKNN',
    'vsknn_EBR': 'knn.uvsknn.UVMContextKNN'
}


# In[48]:


algo_searchspaces = {
    'gru4rec': {
        'loss': ['bpr-max', 'top1-max'],
        'final_act': ['elu-0.5','linear'],
        'dropout_p_hidden': np.linspace(0.0, 0.9, 10, dtype=float),
        'momentum': np.linspace(0.0, 0.9, 10, dtype='f'),
        'learning_rate': np.concatenate([np.linspace(0.1, 0.01, 10, dtype='f'), np.linspace(0.5, 0.2, 4, dtype='f')]),
        'constrained_embedding': [True,False]                    
    },
    'gru4rec_Reminder': {
        'loss': ['bpr-max', 'top1-max'],
        'final_act': ['elu-0.5','linear'],
        'dropout_p_hidden': np.linspace(0.0, 0.9, 10, dtype=float),
        'momentum': np.linspace(0.0, 0.9, 10, dtype='f'),
        'learning_rate': np.concatenate([np.linspace(0.1, 0.01, 10, dtype='f'), np.linspace(0.5, 0.2, 4, dtype='f')]),
        'constrained_embedding': [True,False],
        'reminders': [True],
        'remind_strategy': ['hybrid'],
        'remind_sessions_num': np.linspace(1, 10, 10, dtype=int),
        #'weight_Rel': np.linspace(1, 10, 10, dtype=int),
        'weight_IRec': np.linspace(0, 9, 10, dtype=int)
    },
    'hgru4rec': {
        'final_act': ['linear', 'relu', 'tanh'],
        'dropout_p_hidden_usr': np.linspace(0.0, 0.9, 10, dtype=float),
        'dropout_p_hidden_ses': np.linspace(0.0, 0.9, 10, dtype=float),
        'dropout_p_init': np.linspace(0.0, 0.9, 10, dtype=float),
        'momentum': np.linspace(0.0, 0.9, 10, dtype='f'),
        'learning_rate': np.concatenate([np.linspace(0.1, 0.01, 10, dtype='f'), np.linspace(0.5, 0.2, 4, dtype='f')]),
        'user_propagation_mode': ['init', 'all'],
        'batch_size': [50, 100]
    },
    'shan': {
        'global_dimension': [100],
        'iter': [100],
        'lambda_uv': [0.01, 0.001, 0.0001],
        'lambda_a': [1, 10, 50]
    },
    'sr': {
        'steps': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30],
        'weighting': ['linear','div','quadratic','log']
    },
    'sr_BR': {
        'steps': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30],
        'weighting': ['linear','div','quadratic','log'],
        'boost_own_sessions': np.linspace(0.1, 3.9, 20, dtype='f'),
        'reminders': [True],
        'remind_strategy': ['hybrid'],
        'remind_sessions_num': np.linspace(1, 10, 10, dtype=int),
        #'weight_Rel': np.linspace(1, 10, 10, dtype=int),
        'weight_IRec': np.linspace(0, 9, 10, dtype=int)
    },
    'stan': {
        'k': [100,200,500,1000,1500],
        'sample_size': [1000,2500,5000,10000],
        'lambda_spw': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24],
        'lambda_snh': [2.5, 5, 10, 20, 40, 80, 100],
        'lambda_inh': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24]        
    },
    'vstan': {
        'k': [100,200,500,1000,1500],
        'sample_size': [1000,2500,5000,10000],
        'similarity': ['cosine', 'vec'],
        'lambda_spw': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24],
        'lambda_snh': [2.5, 5, 10, 20, 40, 80, 100],
        'lambda_inh': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24],
        'lambda_ipw': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24],
        'lambda_idf': [False,1,2,5,10]
    },
    'vstan_EBR': {
        'k': [100,200,500,1000,1500],
        'sample_size': [1000,2500,5000,10000],
        'similarity': ['cosine', 'vec'],
        'lambda_spw': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24],
        'lambda_snh': [2.5, 5, 10, 20, 40, 80, 100],
        'lambda_inh': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24],
        'lambda_ipw': [0.00001, 0.4525, 0.905, 1.81, 3.62, 7.24],
        'lambda_idf': [False,1,2,5,10],
        'extend_session_length': np.linspace(1, 25, 25, dtype=int),
        'boost_own_sessions': np.linspace(0.1, 3.9, 20, dtype='f'),
        'reminders': [True],
        'remind_strategy': ['hybrid'],
        #'weight_Rel': np.linspace(1, 10, 10, dtype=int),
        'weight_IRec': np.linspace(0, 9, 10, dtype=int),
        'weight_SSim': np.linspace(0, 9, 10, dtype=int)
    },
    'sknn': {
        'k': [50,100,500,1000,1500],
        'sample_size': [500,1000,2500,5000,10000]
    },
    'vsknn': {
        'k': [50,100,500,1000,1500],
        'sample_size': [500,1000,2500,5000,10000],
        'weighting': ['same','div','linear','quadratic','log'],
        'weighting_score': ['same','div','linear','quadratic','log'],
        'idf_weighting': [False,1,2,5,10]
    },
    'vsknn_EBR': {
        'k': [50,100,500,1000,1500],
        'sample_size': [500,1000,2500,5000,10000],
        'weighting': ['same','div','linear','quadratic','log'],
        'weighting_score': ['same','div','linear','quadratic','log'],
        'idf_weighting': [False,1,2,5,10],
        'extend_session_length': np.linspace(1, 25, 25, dtype=int),
        'boost_own_sessions': np.linspace(0.1, 3.9, 20, dtype='f'),
        'reminders': [True],
        'remind_strategy': ['hybrid'],
        #'weight_Rel': np.linspace(1, 10, 10, dtype=int),
        'weight_IRec': np.linspace(0, 9, 10, dtype=int),
        'weight_SSim': np.linspace(0, 9, 10, dtype=int)
    }
}


# In[49]:


def sample_configs(budget, config_ranges, seed = 0):
  np.random.seed(seed)
  configs = []
  for _ in range(budget):
    configs.append({key: r[np.random.randint(len(r))]                     for key, r in config_ranges.items()})
  return configs


# In[52]:


budget = 100
seed = 0
datatypes = ['app-level', 'sequence-level']
windows = [1,2,3,4,5]

drop_onoff = False

if drop_onoff:
    tuning_folder = 'tuning_onoff'
else:
    tuning_folder = 'tuning'


# In[53]:


# create configuration files (for tuning)
for key in algo_searchspaces:
    configs = sample_configs(budget, algo_searchspaces[key])
    algo_class = algo_classes[key]
    for datatype in datatypes:
        folder_data = 'data/' + str(tuning_folder) + '/' + str(datatype) + '/multiple/'
        folder_res = 'results/' + str(tuning_folder) + '/' + str(datatype) + '/multiple/'
        folder_config = 'conf/' + str(tuning_folder) + '/' + str(datatype) + '/multiple/'
        for window in windows:
            if key not in ['gru4rec', 'gru4rec_Reminder', 'hgru4rec', 'shan']: # due to errors when running config files with multiple configs
                filename = folder_config + str(key) + '-window_' + str(window) + '.yml'
                with open(filename, 'w') as file:
                    file.write('type: single \n')
                    file.write('key: ' + str(key) + '\n')
                    file.write('evaluation: evaluation_user_based \n')
                    file.write('data: \n')
                    file.write('  name: window_' + str(window) + '\n')
                    file.write('  folder: ' + str(folder_data) + '\n')
                    file.write('  prefix: window_' + str(window) + '\n')
                    file.write('  type: hdf \n')
                    file.write('\n')
                    file.write('results: \n')
                    file.write('  folder: ' + str(folder_res) + '\n')
                    file.write('\n')
                    file.write('metrics: \n')
                    file.write('- class: accuracy.HitRate \n')
                    file.write('  length: [1]')
                    file.write('\n \n')
                    file.write('algorithms: \n')

                for config in configs:
                    if 1e-05 not in config.values():
                        with open(filename, 'a') as file:
                            file.write('- class: ' + str(algo_class) + '\n')
                            file.write('  params: ' + str(config) + '\n')
                            file.write('  key: ' + str(key) + '\n')
                    else: # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
                        with open(filename, 'a') as file:
                            file.write('- class: ' + str(algo_class) + '\n')
                            file.write('  params: {')
                            first_key = next(iter(config)) # first key of dict
                            for h in config:
                                if config[h] != 1e-05:
                                    if h == first_key: # no comma before pasting hyperparam & value pair if first key
                                        file.write(str(h) + ': ' + str(config[h]))
                                    else: # comma before (!) 2nd thru last k&v pair (to avoid having comma after final pair)
                                        file.write(', ' + str(h) + ': ' + str(config[h]))
                                else: # so if the value is precisely 1e-05, we want to change it to 0.00001
                                    if h == first_key:
                                        file.write(str(h) + ': ' + str(config[h]))
                                    else:
                                        file.write(', ' + str(h) + ': ' + str(0.0) + str(0) + str(0) + str(0) + str(1))
                            file.write('} \n')
                            file.write('  key: ' + str(key) + '\n')
            else:
                unique_configs = []
                for config in configs:
                    if config not in unique_configs:
                        unique_configs.append(config)
                for i, config in enumerate(unique_configs):
                    filename = folder_config + str(key) + '-config_' + str(i) + '-window_' + str(window) + '.yml'
                    with open(filename, 'w') as file:
                        file.write('type: single \n')
                        file.write('key: ' + str(key) + '\n')
                        file.write('evaluation: evaluation_user_based \n')
                        file.write('data: \n')
                        file.write('  name: ' + 'config_' + str(i) + '_window_' + str(window) + '\n')
                        file.write('  folder: ' + str(folder_data) + '\n')
                        file.write('  prefix: window_' + str(window) + '\n')
                        file.write('  type: hdf \n')
                        file.write('\n')
                        file.write('results: \n')
                        file.write('  folder: ' + str(folder_res) + '\n')
                        file.write('\n')
                        file.write('metrics: \n')
                        file.write('- class: accuracy.HitRate \n')
                        file.write('  length: [1]')
                        file.write('\n \n')
                        file.write('algorithms: \n')
                    if 1e-05 not in config.values():
                        with open(filename, 'a') as file:
                            file.write('- class: ' + str(algo_class) + '\n')
                            file.write('  params: ' + str(config) + '\n')
                            file.write('  key: ' + str(key) + '\n')
                    else: # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
                        with open(filename, 'a') as file:
                            file.write('- class: ' + str(algo_class) + '\n')
                            file.write('  params: {')
                            first_key = next(iter(config)) # first key of dict
                            for h in config:
                                if config[h] != 1e-05:
                                    if h == first_key: # no comma before pasting hyperparam & value pair if first key
                                        file.write(str(h) + ': ' + str(config[h]))
                                    else: # comma before (!) 2nd thru last k&v pair (to avoid having comma after final pair)
                                        file.write(', ' + str(h) + ': ' + str(config[h]))
                                else: # so if the value is precisely 1e-05, we want to change it to 0.00001
                                    if h == first_key:
                                        file.write(str(h) + ': ' + str(config[h]))
                                    else:
                                        file.write(', ' + str(h) + ': ' + str(0.0) + str(0) + str(0) + str(0) + str(1))
                            file.write('} \n')
                            file.write('  key: ' + str(key) + '\n')

