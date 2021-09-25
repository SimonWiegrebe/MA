#!/usr/bin/env python
# coding: utf-8

# In[123]:


import datetime
import matplotlib
import numpy as np
import pandas as pd
import pyreadr
import pickle

import os


# In[124]:


os.chdir('C:\\Users\\Simon\\Desktop\\MA\\session-rec')


# In[125]:


algo_classes = {
    'gru4rec': 'gru4rec.gru4rec.GRU4Rec',
    'gru4rec_Reminder': 'gru4rec.ugru4rec.UGRU4Rec',
    'hgru4rec': 'hgru4rec.hgru4rec.HGRU4Rec',
    'shan': 'shan.shan.SHAN',
    'ar': 'baselines.ar.AssociationRules',
    'ct-pre': 'ct.ct.ContextTree',
    'sr': 'baselines.sr.SequentialRules',
    'sr_BR': 'baselines.usr.USequentialRules',
    'stan': 'knn.stan.STAN',
    'vstan': 'knn.vstan.VSKNN_STAN',
    'vstan_EBR': 'knn.uvstan.U_VSKNN_STAN',
    'sknn': 'knn.sknn.ContextKNN',
    'vsknn': 'knn.vsknn.VMContextKNN',
    'vsknn_EBR': 'knn.uvsknn.UVMContextKNN'
}


# In[126]:


hyperparams_app = {
    'gru4rec': {},
    'gru4rec_Reminder': {},
    'hgru4rec': {},
    'shan': {},
    'ar': {},
    'ct-pre': {},
    'sr': {},
    'sr_BR': {},
    'stan': {},
    'vstan': {},
    'vstan_EBR': {},
    'sknn': {},
    'vsknn': {},
    'vsknn_EBR': {}
}

hyperparams_seq = {
    'gru4rec': {},
    'gru4rec_Reminder': {},
    'hgru4rec': {},
    'shan': {},
    'ar': {},
    'ct-pre': {},
    'sr': {},
    'sr_BR': {},
    'stan': {},
    'vstan': {},
    'vstan_EBR': {},
    'sknn': {},
    'vsknn': {},
    'vsknn_EBR': {}
}


# In[130]:


seed = 0
datatypes = ['app-level', 'sequence-level']
windows = [1,2,3,4,5]
drop_onoff = False

if drop_onoff:
    tuning_folder = 'tuning' # switch to 'tuning' for on-off analysis w/o separate tuning
    testing_folder = 'testing_onoff'
else:
    tuning_folder = 'tuning'
    testing_folder = 'testing'


# ### all data

# In[131]:


for datatype in datatypes:
    if datatype == 'app-level':
        d = hyperparams_app
    else:
        d = hyperparams_seq
    # extract optimal hyperparameter configuration per datatype and algorithm
    for key in d.keys():
        folder_res = 'results/' + str(tuning_folder) + '/' + str(datatype) + '/multiple/'
        key_results = [f for f in os.listdir(folder_res) if f.startswith('test_single_' + key + '_window_')]
        
        if len(key_results) > 0: # tuning results available -> extract optimal hyperparam config
            results = pd.DataFrame()
            for file in key_results:
                window = file.strip('.csv').split('_')[-1]
                df = pd.read_csv(folder_res + file, sep = ';')
                df.drop(df.filter(regex='Unnamed'), axis=1, inplace=True)
                df.rename(columns={'HitRate@1: ': 'window_' + str(window)}, inplace=True)
                if 'Metrics' not in results.columns:
                    results = df
                else:
                    results = pd.concat([results, df['window_' + str(window)]], axis=1)
            results.dropna(inplace=True)
            model = results['Metrics'].apply(lambda x: x.split('-')[0])
            results.insert(0, 'model', model)
            results['average'] = results.iloc[:, 2:6].mean(axis=1)

            maxrow = results['average'].argmax()
            hyperparams = results['Metrics'].iloc[maxrow]
            # convert hyphen to subscore b/c below we split based on hyphen
            if 'bpr-max' in hyperparams:
                hyperparams = hyperparams.replace('bpr-max', 'bpr_max')
            if 'top1-max' in hyperparams:
                hyperparams = hyperparams.replace('top1-max', 'top1_max')
            if 'elu-0.5' in hyperparams:
                hyperparams = hyperparams.replace('elu-0.5', 'elu_0.5')
            # change 1e-05 to 0.00001
            if '1e-05' in hyperparams:
                hyperparams = hyperparams.replace('1e-05', '0.00001')
            
            for h in hyperparams.split('-')[1:]:
                k = h.split('=')[0]
                v = h.split('=')[1]
                if v.replace('.','',1).isdigit():
                    if '.' in v: v = float(v)
                    else: v = int(v)
                if v in ['bpr_max', 'top1_max', 'elu_0.5']:
                    v = v.replace('_', '-')
                d[key][k] = v

        # create configuration files (for testing)
        if (len(key_results) != 0) or (key in ['ar', 'ct-pre']): # only parameter-free algos and algos with tuning results available 
            config = d[key]
            algo_class = algo_classes[key]
            # multiple windows
            folder_data = 'data/' + str(testing_folder) + '/' + str(datatype) + '/multiple/'
            folder_res = 'results/' + str(testing_folder) + '/' + str(datatype) + '/multiple/'
            folder_config = 'conf/' + str(testing_folder) + '/' + str(datatype) + '/multiple/'
            for window in windows:
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
                    file.write('  length: [1,5,10,20] \n' )
                    file.write('- class: accuracy.MRR \n')
                    file.write('  length: [5,10,20] \n')
                    file.write('- class: coverage.Coverage \n')
                    file.write('  length: [20] \n')
                    file.write('- class: popularity.Popularity \n')
                    file.write('  length: [20] \n')
                    file.write('- class: saver.Saver \n')
                    file.write('  length: [50] \n')
                    file.write('\n \n')
                    file.write('algorithms: \n')

                # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
                # (potentially something similar necessary for Boolean hyperparams)
                if 1e-05 not in config.values():
                    with open(filename, 'a') as file:
                        file.write('- class: ' + str(algo_class) + '\n')
                        file.write('  params: ' + str(config) + '\n')
                        file.write('  key: ' + str(key) + '\n')
                else:
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

            # single window
            folder_data = 'data/testing/' + str(datatype) + '/single/'
            folder_res = 'results/testing/' + str(datatype) + '/single/'
            folder_config = 'conf/testing/' + str(datatype) + '/single/'
            filename = folder_config + str(key) + '.yml'
            with open(filename, 'w') as file:
                file.write('type: single \n')
                file.write('key: ' + str(key) + '\n')
                file.write('evaluation: evaluation_user_based \n')
                file.write('data: \n')
                file.write('  name: single' + '\n')
                file.write('  folder: ' + str(folder_data) + '\n')
                file.write('  prefix: single' + '\n')
                file.write('  type: hdf \n')
                file.write('\n')
                file.write('results: \n')
                file.write('  folder: ' + str(folder_res) + '\n')
                file.write('\n')
                file.write('metrics: \n')
                file.write('- class: accuracy.HitRate \n')
                file.write('  length: [1,5,10,20] \n' )
                file.write('- class: accuracy.MRR \n')
                file.write('  length: [5,10,20] \n')
                file.write('- class: coverage.Coverage \n')
                file.write('  length: [20] \n')
                file.write('- class: popularity.Popularity \n')
                file.write('  length: [20] \n')
                file.write('- class: saver.Saver \n')
                file.write('  length: [50] \n')
                file.write('\n \n')
                file.write('algorithms: \n')

            # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
            # (potentially something similar necessary for Boolean hyperparams)
            if 1e-05 not in config.values():
                with open(filename, 'a') as file:
                    file.write('- class: ' + str(algo_class) + '\n')
                    file.write('  params: ' + str(config) + '\n')
                    file.write('  key: ' + str(key) + '\n')
            else:
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


# In[129]:


with open('../data/app-level/hyperparams_app.pickle', 'wb') as handle:
    pickle.dump(hyperparams_app, handle)
with open('../data/sequence-level/hyperparams_seq.pickle', 'wb') as handle:
    pickle.dump(hyperparams_seq, handle)


# ### min20

# In[92]:


datatype = 'app-level'
min20_key = 'min20'


# In[93]:


with open('../data/app-level/hyperparams_app.pickle', 'rb') as handle:
    d = pickle.load(handle)


# In[94]:


for key in d.keys():
    folder_res = 'results/tuning/' + str(datatype) + '/multiple/'
    key_results = [f for f in os.listdir(folder_res) if f.startswith('test_single_' + key + '_window_')]
    
    # create configuration files (for testing)
    if (len(key_results) != 0) or (key in ['ar', 'ct-pre']): # only parameter-free algos and algos with tuning results available 
        config = d[key]
        algo_class = algo_classes[key]
        folder_data = 'data/testing/' + str(datatype) + '/multiple/'
        folder_res = 'results/testing/' + str(datatype) + '/multiple/'
        folder_config = 'conf/testing/' + str(datatype) + '/multiple/'
        for window in windows:
            filename = folder_config + str(key) + '-window_' + str(window) + '-' + str(min20_key) + '.yml' # add "min20"
            with open(filename, 'w') as file:
                file.write('type: single \n')
                file.write('key: ' + str(key) + '\n')
                file.write('evaluation: evaluation_user_based \n')
                file.write('data: \n')
                file.write('  name: window_' + str(window) + '-' + str(min20_key) + '\n') # add "min20"
                file.write('  folder: ' + str(folder_data) + '\n')
                file.write('  prefix: window_' + str(window) + '-' + str(min20_key) + '\n') # add "min20"
                file.write('  type: hdf \n')
                file.write('\n')
                file.write('results: \n')
                file.write('  folder: ' + str(folder_res) + '\n')
                file.write('\n')
                file.write('metrics: \n')
                file.write('- class: accuracy.HitRate \n')
                file.write('  length: [1,5,10,20] \n' )
                file.write('- class: accuracy.MRR \n')
                file.write('  length: [5,10,20] \n')
                file.write('- class: coverage.Coverage \n')
                file.write('  length: [20] \n')
                file.write('- class: popularity.Popularity \n')
                file.write('  length: [20] \n')
                file.write('- class: saver.Saver \n')
                file.write('  length: [50] \n')
                file.write('\n \n')
                file.write('algorithms: \n')

            # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
            # (potentially something similar necessary for Boolean hyperparams)
            if 1e-05 not in config.values():
                with open(filename, 'a') as file:
                    file.write('- class: ' + str(algo_class) + '\n')
                    file.write('  params: ' + str(config) + '\n')
                    file.write('  key: ' + str(key) + '\n')
            else:
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


# ### min20_test

# In[95]:


datatype = 'app-level'
min20_test_key = 'min20_test'


# In[96]:


with open('../data/app-level/hyperparams_app.pickle', 'rb') as handle:
    d = pickle.load(handle)


# In[97]:


for key in d.keys():
    folder_res = 'results/tuning/' + str(datatype) + '/multiple/'
    key_results = [f for f in os.listdir(folder_res) if f.startswith('test_single_' + key + '_window_')]
    
    # create configuration files (for testing)
    if (len(key_results) != 0) or (key in ['ar', 'ct-pre']): # only parameter-free algos and algos with tuning results available 
        config = d[key]
        algo_class = algo_classes[key]
        folder_data = 'data/testing/' + str(datatype) + '/multiple/'
        folder_res = 'results/testing/' + str(datatype) + '/multiple/'
        folder_config = 'conf/testing/' + str(datatype) + '/multiple/'
        for window in windows:
            filename = folder_config + str(key) + '-window_' + str(window) + '-' + str(min20_test_key) + '.yml' # add "min20"
            with open(filename, 'w') as file:
                file.write('type: single \n')
                file.write('key: ' + str(key) + '\n')
                file.write('evaluation: evaluation_user_based \n')
                file.write('data: \n')
                file.write('  name: window_' + str(window) + '-' + str(min20_test_key) + '\n') # add "min20"
                file.write('  folder: ' + str(folder_data) + '\n')
                file.write('  prefix: window_' + str(window) + '-' + str(min20_test_key) + '\n') # add "min20"
                file.write('  type: hdf \n')
                file.write('\n')
                file.write('results: \n')
                file.write('  folder: ' + str(folder_res) + '\n')
                file.write('\n')
                file.write('metrics: \n')
                file.write('- class: accuracy.HitRate \n')
                file.write('  length: [1,5,10,20] \n' )
                file.write('- class: accuracy.MRR \n')
                file.write('  length: [5,10,20] \n')
                file.write('- class: coverage.Coverage \n')
                file.write('  length: [20] \n')
                file.write('- class: popularity.Popularity \n')
                file.write('  length: [20] \n')
                file.write('- class: saver.Saver \n')
                file.write('  length: [50] \n')
                file.write('\n \n')
                file.write('algorithms: \n')

            # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
            # (potentially something similar necessary for Boolean hyperparams)
            if 1e-05 not in config.values():
                with open(filename, 'a') as file:
                    file.write('- class: ' + str(algo_class) + '\n')
                    file.write('  params: ' + str(config) + '\n')
                    file.write('  key: ' + str(key) + '\n')
            else:
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


# ### embedding analysis

# In[98]:


datatype = 'app-level'
embedding_key = 'embedding'
applicable_models = ['gru4rec', 'gru4rec_Reminder']
embedding_sizes = [16, 32, 64, 128]


# In[99]:


with open('../data/app-level/hyperparams_app.pickle', 'rb') as handle:
    d = pickle.load(handle)


# In[100]:


for embedding_size in embedding_sizes:
    for key in applicable_models:
        folder_res = 'results/tuning/' + str(datatype) + '/multiple/'
        key_results = [f for f in os.listdir(folder_res) if f.startswith('test_single_' + key + '_window_')]

        # create configuration files (for testing)
        if (len(key_results) != 0) or (key in ['ar', 'ct-pre']): # only parameter-free algos and algos with tuning results available 
            config = d[key]
            algo_class = algo_classes[key]
            # multiple windows
            folder_data = 'data/testing/' + str(datatype) + '/multiple/'
            folder_res = 'results/testing/' + str(datatype) + '/multiple/'
            folder_config = 'conf/testing/' + str(datatype) + '/multiple/'
            for window in windows:
                filename = folder_config + str(key) + '-window_' + str(window) + '-' + str(embedding_key) + '_' + str(embedding_size) + '.yml'
                with open(filename, 'w') as file:
                    file.write('type: single \n')
                    file.write('key: ' + str(key) + '\n')
                    file.write('evaluation: evaluation_user_based \n')
                    file.write('data: \n')
                    file.write('  name: window_' + str(window) + '-' 
                               + str(embedding_key) + '_' + str(embedding_size) + '\n')
                    file.write('  folder: ' + str(folder_data) + '\n')
                    file.write('  prefix: window_' + str(window) + '\n') # do not add embedding_size here (this is the data name!)
                    file.write('  type: hdf \n')
                    file.write('\n')
                    file.write('results: \n')
                    file.write('  folder: ' + str(folder_res) + '\n')
                    file.write('  pickle_models:' + '\n')
                    file.write('\n')
                    file.write('metrics: \n')
                    file.write('- class: accuracy.HitRate \n')
                    file.write('  length: [1,5,10,20] \n' )
                    file.write('- class: accuracy.MRR \n')
                    file.write('  length: [5,10,20] \n')
                    file.write('- class: coverage.Coverage \n')
                    file.write('  length: [20] \n')
                    file.write('- class: popularity.Popularity \n')
                    file.write('  length: [20] \n')
                    file.write('- class: saver.Saver \n')
                    file.write('  length: [50] \n')
                    file.write('\n \n')
                    file.write('algorithms: \n')

                # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
                # (potentially something similar necessary for Boolean hyperparams)
                if 1e-05 not in config.values():
                    with open(filename, 'a') as file:
                        file.write('- class: ' + str(algo_class) + '\n')
                        file.write('  params: {')
                        first_key = next(iter(config)) # first key of dict
                        for h in config:
                            if h == first_key: # no comma before pasting hyperparam & value pair if first key
                                file.write(str(h) + ': ' + str(config[h]))
                            else: # comma before (!) 2nd thru last k&v pair (to avoid having comma after final pair)
                                file.write(', ' + str(h) + ': ' + str(config[h]))
                        file.write(', embedding: ' + str(embedding_size))
                        file.write('} \n')
                        file.write('  key: ' + str(key) + '\n')
                else:
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
                        file.write(', embedding: ' + str(embedding_size))
                        file.write('} \n')
                        file.write('  key: ' + str(key) + '\n')
                        
            # single window
            folder_data = 'data/testing/' + str(datatype) + '/single/'
            folder_res = 'results/testing/' + str(datatype) + '/single/'
            folder_config = 'conf/testing/' + str(datatype) + '/single/'
            filename = folder_config + str(key) + '-' + str(embedding_key) + '_' + str(embedding_size) + '.yml'
            with open(filename, 'w') as file:
                file.write('type: single \n')
                file.write('key: ' + str(key) + '\n')
                file.write('evaluation: evaluation_user_based \n')
                file.write('data: \n')
                file.write('  name: single' + '-' + str(embedding_key) + '_' + str(embedding_size) + '\n')
                file.write('  folder: ' + str(folder_data) + '\n')
                file.write('  prefix: single' + '\n') # do not add embedding_size here (this is the data name!)
                file.write('  type: hdf \n')
                file.write('\n')
                file.write('results: \n')
                file.write('  folder: ' + str(folder_res) + '\n')
                file.write('  pickle_models:' + '\n')
                file.write('\n')
                file.write('metrics: \n')
                file.write('- class: accuracy.HitRate \n')
                file.write('  length: [1,5,10,20] \n' )
                file.write('- class: accuracy.MRR \n')
                file.write('  length: [5,10,20] \n')
                file.write('- class: coverage.Coverage \n')
                file.write('  length: [20] \n')
                file.write('- class: popularity.Popularity \n')
                file.write('  length: [20] \n')
                file.write('- class: saver.Saver \n')
                file.write('  length: [50] \n')
                file.write('\n \n')
                file.write('algorithms: \n')

            # to avoid 1e-05 instead of 0.00001 (1e-05 cannot be processed by algos)
            # (potentially something similar necessary for Boolean hyperparams)
            if 1e-05 not in config.values():
                with open(filename, 'a') as file:
                    file.write('- class: ' + str(algo_class) + '\n')
                    file.write('  params: {')
                    first_key = next(iter(config)) # first key of dict
                    for h in config:
                        if h == first_key: # no comma before pasting hyperparam & value pair if first key
                            file.write(str(h) + ': ' + str(config[h]))
                        else: # comma before (!) 2nd thru last k&v pair (to avoid having comma after final pair)
                            file.write(', ' + str(h) + ': ' + str(config[h]))
                    file.write(', embedding: ' + str(embedding_size))
                    file.write('} \n')
                    file.write('  key: ' + str(key) + '\n')
            else:
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
                    file.write(', embedding: ' + str(embedding_size))
                    file.write('} \n')
                    file.write('  key: ' + str(key) + '\n')

