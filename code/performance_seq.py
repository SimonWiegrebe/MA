#!/usr/bin/env python
# coding: utf-8

# ### setup

# In[12]:


import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyreadr
import pickle
import re
import os


# In[13]:


os.chdir('C:\\Users\\Simon\\Desktop\\MA\\session-rec')


# In[14]:


# datatypes = ['app-level', 'seq-level']
datatype = 'sequence-level'
windows = [1,2,3,4,5]
model_index = [0, 3, 8, 9, 10, 4, 1, 2, 5, 6, 7]
model_name = 'Algorithm'


# In[15]:


USER_KEY = 'userID'
TIME_KEY = 'timestamp'
if datatype == 'app-level':    
    ITEM_KEY = 'appID'
    SESSION_KEY = 'sessionID'
else:
    ITEM_KEY = 'usID'
    SESSION_KEY = 'sentenceID'


# ### helper functions

# In[16]:


# for multiple windows (incl. min20)
# get average performance across all windows for a given algorithm
def get_av_perf(files, key):
    res = pd.DataFrame()
    for file in files:
        window = file.strip('.csv').split('_')[-1]
        df = pd.read_csv(folder_res + file, sep = ';')
        df.drop(['Metrics', 'Saver@50: '], axis=1, inplace=True)
        df.drop(df.filter(regex='Unnamed'), axis=1, inplace=True) # drop 'Unnamed: 24' column containing only NaNs
        df.rename(columns = lambda x : str(x)[:-2], inplace=True) # remove colon and whitespace from all column names
        df.insert(0, model_name, key)
        df.insert(1, 'window', window)
        res = res.append(df)
    res = res.groupby(model_name).mean().reset_index(level=0)
    return(res)


# In[17]:


# for single window
# get performance for a given algorithm
def get_perf(file, key):
    df = pd.read_csv(folder_res + file, sep = ';')
    df.drop(['Metrics', 'Saver@50: '], axis=1, inplace=True)
    df.drop(df.filter(regex='Unnamed'), axis=1, inplace=True) # drop 'Unnamed: 24' column containing only NaNs
    df.rename(columns = lambda x : str(x)[:-2], inplace=True) # remove colon and whitespace from all column names
    df.insert(0, model_name, key)
    return(df)


# In[18]:


# extract ground truth from test data (test_data) for a single item (position) in a single session (sessionId)
def extract_ground_truth(ID, position, test_data):
    relevant_df = test_data[test_data[SESSION_KEY]==ID]
    index = relevant_df.index[position+1]
    ground_truth = relevant_df[ITEM_KEY][index]
    return ground_truth


# In[19]:


# generate a df containing the ground truth as well as predictions for all available algorithms
def generate_predictions(predictions_files, test_data, mapping_id2name, multiple=True):
    predictions = pd.DataFrame()
    for file in predictions_files:
        if multiple:
            model = "_".join(file.split('_')[2:-2])
        else:
            model = "_".join(file.split('_')[2:-1])
        df = pd.read_csv(folder_res + file, sep = ';')
        if 'sessionID' not in predictions.columns:
            predictions['sessionID'] = df['SessionId']
        if 'position' not in predictions.columns:
            predictions['position'] = df['Position']
        if 'ground_truth' not in predictions.columns:
            predictions['ground_truth'] = predictions.apply(lambda x: extract_ground_truth(x['sessionID'], x['position'], test_data), axis=1)
            predictions['ground_truth_name'] = predictions['ground_truth'].apply(lambda x: mapping_reverse[x])
        predictions['recs-' + model] = df['Recommendations'].apply(lambda x: [int(i) for i in x.split(',')])
        predictions['recs_names-' + model] = predictions['recs-' + model].apply(lambda x: [mapping_reverse[i] for i in x])
        predictions['scores-' + model] = df['Scores'].apply(lambda x: x.split(','))
    return predictions


# In[20]:


# helper function outputting whether ground truth is in recommendation list of length k for a single algorithm and item
def calc_hr_k(ground_truth, rec_list, k):
    return ground_truth in rec_list[:k]


# In[21]:


# helper function for calculating the MRR
def calc_mrr_k(ground_truth, rec_list, k):
    if ground_truth not in rec_list[:k]:
        return 0
    else:
        score = rec_list.index(ground_truth) + 1
        return 1/score


# In[22]:


def capitalize_names(df):
    name_dict = {
        'ar': 'AR',
        'ct-pre': 'CT',
        'ctpre': 'CT',
        'gru4rec': 'GRU4Rec',
        'gru4rec_Reminder': 'GRU4Rec_R',
        'hgru4rec': 'HGRU4Rec',
        'shan': 'SHAN',
        'sknn': 'SKNN',
        'sr': 'SR',
        'sr_BR': 'SR_BR',
        'stan': 'STAN',
        'vsknn': 'VSKNN',
        'vsknn_EBR': 'VSKNN_EBR',
        'vstan': 'VSTAN',
        'vstan_EBR': 'VSTAN_EBR'
    }
    df[model_name] = df[model_name].apply(lambda x: name_dict[x])
    return df


# ### multiple windows

# ##### overall

# In[23]:


folder_res = 'results/testing/' + str(datatype) + '/multiple/'
algos = set([f.split('_window')[0].split('test_single_')[1] for f in os.listdir(folder_res)])
results_seq_multiple = pd.DataFrame()
for key in algos:
    files = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) 
             and (f.startswith('test_single_' + str(key) + '_window'))
             and ('min20' not in f)]
    res = get_av_perf(files, key)
    results_seq_multiple = results_seq_multiple.append(res)
results_seq_multiple = capitalize_names(results_seq_multiple)
results_seq_multiple = results_seq_multiple.round(4)
results_seq_multiple = results_seq_multiple.sort_values(model_name)
results_seq_multiple['model_index'] = model_index
results_seq_multiple = results_seq_multiple.set_index('model_index').sort_index()
results_seq_multiple.index.name = None


# In[24]:


results_seq_multiple


# In[25]:


with open('../MA/tables/results_seq_multiple.tex','w') as tf:
    tf.write(results_seq_multiple.to_latex(index=False))


# In[26]:


with open('../MA/results/sequence-level/results_seq_multiple.pickle', 'wb') as handle:
    pickle.dump(results_seq_multiple, handle)


# ### single window

# ##### overall

# In[27]:


folder_res = 'results/testing/' + str(datatype) + '/single/'
algos = set([f.split('test_single_')[1].split('_single')[0] for f in os.listdir(folder_res) if f.startswith('test_single')])
results_seq_single = pd.DataFrame()
for key in algos:
    file = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) 
             and (f.startswith('test_single_' + str(key) + '_single'))
             and ('min20' not in f)
             and ('embedding' not in f)][0] # list is of length 1 actually
    res = get_perf(file, key)
    results_seq_single = results_seq_single.append(res)
results_seq_single = capitalize_names(results_seq_single)
results_seq_single = results_seq_single.round(4)
results_seq_single = results_seq_single.sort_values(model_name)
results_seq_single['model_index'] = model_index
results_seq_single = results_seq_single.set_index('model_index').sort_index()
results_seq_single.index.name = None


# In[28]:


results_seq_single


# In[29]:


with open('../MA/tables/results_seq_single.tex','w') as tf:
    tf.write(results_seq_single.to_latex(index=False))


# In[30]:


with open('../MA/results/sequence-level/results_seq_single.pickle', 'wb') as handle:
    pickle.dump(results_seq_single, handle)


# ### removing on and off (unspecific tuning)

# ##### multiple windows

# In[31]:


folder_res = 'results/testing_onoff_unspecific_tuning/' + str(datatype) + '/multiple/'
algos = set([f.split('_window')[0].split('test_single_')[1] for f in os.listdir(folder_res)])
results_seq_multiple_droponoff = pd.DataFrame()
for key in algos:
    files = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) 
             and (f.startswith('test_single_' + str(key) + '_window'))
             and ('min20' not in f)]
    res = get_av_perf(files, key)
    results_seq_multiple_droponoff = results_seq_multiple_droponoff.append(res)
results_seq_multiple_droponoff = capitalize_names(results_seq_multiple_droponoff)
results_seq_multiple_droponoff = results_seq_multiple_droponoff.round(4)
results_seq_multiple_droponoff = results_seq_multiple_droponoff.sort_values(model_name)
results_seq_multiple_droponoff['model_index'] = model_index
results_seq_multiple_droponoff = results_seq_multiple_droponoff.set_index('model_index').sort_index()
results_seq_multiple_droponoff.index.name = None


# In[32]:


results_seq_multiple_droponoff


# In[33]:


with open('../MA/tables/results_seq_multiple_droponoff.tex','w') as tf:
    tf.write(results_seq_multiple_droponoff.to_latex(index=False))


# In[34]:


with open('../MA/results/sequence-level/results_seq_multiple_droponoff.pickle', 'wb') as handle:
    pickle.dump(results_seq_multiple_droponoff, handle)


# In[35]:


# # performance drop vis-à-vis results_seq_multiple
# 1 - (results_seq_multiple_droponoff['HitRate@1'] / results_seq_multiple['HitRate@1']).mean()


# ### performance by position

# ##### create mapping dicts

# In[36]:


folder_res = 'results/testing_onoff_unspecific_tuning/' + str(datatype) + '/multiple/'
folder_data = 'data/testing_onoff/' + str(datatype) + '/multiple/'
data = pd.read_csv('../data/sequence-level/data_seq.csv') # create app and user mappings
mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['category_list'])))])
mapping_reverse = dict((v,k) for k,v in mapping.items())


# ##### individual positions

# In[57]:


k = 20 # HR@k


# In[58]:


results_seq_multiple_pos = pd.DataFrame()

for window in windows:
    test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
    predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                         and f.endswith('window_' + str(window) + '-Saver@50.csv')]
    predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
    algorithms = [i for i in predictions.columns if i.startswith('recs-')]

    perf_by_pos = pd.DataFrame()
    positions = range(1,11)
    for pos in positions:
        pred_pos = predictions[predictions['position']==pos-1]
        df = pd.DataFrame()
        df['position'] = ['position = ' + str(pos)]
        df['window'] = [window]
        for algo in algorithms:
            algo_name = ''.join(algo.split('-')[1:])
            value = pred_pos.apply(lambda x: calc_hr_k(x['ground_truth'], x[algo], k), axis=1).sum()/len(pred_pos)
            df[algo_name] = [value]
        perf_by_pos = perf_by_pos.append(df).reset_index(drop=True)
    results_seq_multiple_pos = results_seq_multiple_pos.append(perf_by_pos)

results_seq_multiple_pos = results_seq_multiple_pos.groupby('position').mean() # average across positions
results_seq_multiple_pos.drop(['window'], axis=1, inplace=True)
results_seq_multiple_pos = results_seq_multiple_pos.transpose() # transpose to have algorithms as rows, positions as columns
columns_reordered = results_seq_multiple_pos.columns.tolist()
columns_reordered.sort(key=lambda x: int(re.search(r'\d+$',x).group()))
results_seq_multiple_pos = results_seq_multiple_pos[columns_reordered]
results_seq_multiple_pos.reset_index(inplace=True) # convert index to column named "index"
results_seq_multiple_pos.rename(columns={'index': model_name}, inplace=True) # rename column "index" to "model"
results_seq_multiple_pos.rename_axis(None, axis=1, inplace=True) # unname new index
results_seq_multiple_pos = capitalize_names(results_seq_multiple_pos) # adjust model names
results_seq_multiple_pos = results_seq_multiple_pos.round(4)
results_seq_multiple_pos = results_seq_multiple_pos.sort_values(model_name)
results_seq_multiple_pos['model_index'] = model_index
results_seq_multiple_pos = results_seq_multiple_pos.set_index('model_index').sort_index()
results_seq_multiple_pos.index.name = None


# In[59]:


results_seq_multiple_pos


# In[60]:


with open('../MA/tables/results_seq_multiple_pos_HR@' + str(k) + '.tex','w') as tf:
    tf.write(results_seq_multiple_pos.to_latex(index=False))


# In[61]:


with open('../MA/results/sequence-level/results_seq_multiple_pos_HR@' + str(k) + '.pickle', 'wb') as handle:
    pickle.dump(results_seq_multiple_pos, handle)


# ##### cutoffs

# In[62]:


cutoffs = [2, 5, 10]
k = 1 # HR@k


# In[63]:


results_seq_multiple_cutoff = pd.DataFrame()

for cutoff in cutoffs:
    for window in windows:
        test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
        predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                             and f.endswith('window_' + str(window) + '-Saver@50.csv')]
        predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
        algorithms = [i for i in predictions.columns if i.startswith('recs-')]

        # for  single cutoff and single window, create 'performance-by-position' df containing two rows and |algorithms| columns
        perf_by_pos = pd.DataFrame()
        positions = ['position <= ' + str(cutoff), 'position > ' + str(cutoff)]
        for pos in positions:
            if pos==('position <= ' + str(cutoff)):
                pred_pos = predictions[predictions['position']<=cutoff-1] # -1 b/c the first position has index 0
            else:
                pred_pos = predictions[predictions['position']>cutoff-1]
            df = pd.DataFrame()
            df['position'] = [pos]
            df['window'] = [window]
            for algo in algorithms:
                algo_name = ''.join(algo.split('-')[1:])
                value = pred_pos.apply(lambda x: calc_hr_k(x['ground_truth'], x[algo], k), axis=1).sum()/len(pred_pos)
                df[algo_name] = [value]
            perf_by_pos = perf_by_pos.append(df).reset_index(drop=True)
        results_seq_multiple_cutoff = results_seq_multiple_cutoff.append(perf_by_pos)

results_seq_multiple_cutoff = results_seq_multiple_cutoff.groupby('position').mean() # average across positions (e.g., "<= 2", "> 10")
results_seq_multiple_cutoff.drop(['window'], axis=1, inplace=True)
results_seq_multiple_cutoff = results_seq_multiple_cutoff.transpose() # transpose to have algorithms as rows, positions as columns
columns_reordered = results_seq_multiple_cutoff.columns.tolist()
columns_reordered.sort(key=lambda x: int(re.search(r'\d+$',x).group()))
results_seq_multiple_cutoff = results_seq_multiple_cutoff[columns_reordered]
results_seq_multiple_cutoff.reset_index(inplace=True) # convert index to column named "index"
results_seq_multiple_cutoff.rename(columns={'index': model_name}, inplace=True) # rename column "index" to "model"
results_seq_multiple_cutoff.rename_axis(None, axis=1, inplace=True) # unname new index
results_seq_multiple_cutoff = capitalize_names(results_seq_multiple_cutoff) # adjust model names
results_seq_multiple_cutoff = results_seq_multiple_cutoff.round(4)
results_seq_multiple_cutoff = results_seq_multiple_cutoff.sort_values(model_name)
results_seq_multiple_cutoff['model_index'] = model_index
results_seq_multiple_cutoff = results_seq_multiple_cutoff.set_index('model_index').sort_index()
results_seq_multiple_cutoff.index.name = None


# In[64]:


results_seq_multiple_cutoff


# In[65]:


with open('../MA/tables/results_seq_multiple_cutoff_HR@' + str(k) + '.tex','w') as tf:
    tf.write(results_seq_multiple_cutoff.to_latex(index=False))


# In[66]:


with open('../MA/results/sequence-level/results_seq_multiple_cutoff_HR@' + str(k) + '.pickle', 'wb') as handle:
    pickle.dump(results_seq_multiple_cutoff, handle)


# ### impact of ONOFF-removal

# ##### original data: which percentage of all top 1 predictions are ONOFF tokens (here: sum across all 5 windows)

# In[47]:


folder_res = 'results/testing/' + str(datatype) + '/multiple/'
folder_data = folder_res.replace('results', 'data')
data = pd.read_csv('../data/sequence-level/data_seq.csv') # create app and user mappings
mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['category_list'])))])
mapping_reverse = dict((v,k) for k,v in mapping.items())


# In[48]:


k = 1 # HR@k
ONOFF = mapping['ON,OFF']


# In[49]:


results_seq_multiple_onoff_perc = pd.DataFrame()

for window in windows:
    test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
    predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                         and f.endswith('window_' + str(window) + '-Saver@50.csv')]
    predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
    algorithms = [i for i in predictions.columns if i.startswith('recs-')]

    df = pd.DataFrame()
    df['window'] = [window]
    df['num_preds'] = [len(predictions)]
    for algo in algorithms:
        algo_name = ''.join(algo.split('-')[1:])
        value = predictions.apply(lambda x: calc_hr_k(ONOFF, x[algo], k), axis=1).sum()  
        df[algo_name] = [value]
    
    results_seq_multiple_onoff_perc = results_seq_multiple_onoff_perc.append(df)

results_seq_multiple_onoff_perc = pd.DataFrame(results_seq_multiple_onoff_perc.sum())
results_seq_multiple_onoff_perc = results_seq_multiple_onoff_perc.transpose()
num_preds = results_seq_multiple_onoff_perc['num_preds'][0]
results_seq_multiple_onoff_perc.drop(['window', 'num_preds'], axis=1, inplace=True)
results_seq_multiple_onoff_perc = results_seq_multiple_onoff_perc.div(num_preds)


# In[50]:


results_seq_multiple_onoff_perc


# In[51]:


with open('../MA/tables/results_seq_multiple_onoff_perc' + str(k) + '.tex','w') as tf:
    tf.write(results_seq_multiple_onoff_perc.to_latex(index=False))


# In[52]:


with open('../MA/results/sequence-level/results_seq_multiple_onoff_perc' + str(k) + '.pickle', 'wb') as handle:
    pickle.dump(results_seq_multiple_onoff_perc, handle)


# ##### original data: performance when excluding ONOFF from test sequences

# In[53]:


results_seq_multiple_non_onoff_perf = pd.DataFrame()

for window in windows:
    test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
    predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                         and f.endswith('window_' + str(window) + '-Saver@50.csv')]
    predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
    predictions = predictions[predictions['ground_truth'] != ONOFF]
    algorithms = [i for i in predictions.columns if i.startswith('recs-')]

    df = pd.DataFrame()
    df['window'] = [window]
    for algo in algorithms:
        algo_name = ''.join(algo.split('-')[1:])
        value = predictions.apply(lambda x: calc_hr_k(x['ground_truth'], x[algo], k), axis=1).sum()/len(predictions)
        df[algo_name] = [value]
    
    results_seq_multiple_non_onoff_perf = results_seq_multiple_non_onoff_perf.append(df)

results_seq_multiple_non_onoff_perf = pd.DataFrame(results_seq_multiple_non_onoff_perf.mean())
results_seq_multiple_non_onoff_perf = results_seq_multiple_non_onoff_perf.transpose()
results_seq_multiple_non_onoff_perf.drop(['window'], axis=1, inplace=True)
results_seq_multiple_non_onoff_perf = results_seq_multiple_non_onoff_perf.div(num_preds)


# In[54]:


results_seq_multiple_non_onoff_perf


# In[55]:


with open('../MA/tables/results_seq_multiple_non_onoff_perf' + str(k) + '.tex','w') as tf:
    tf.write(results_seq_multiple_non_onoff_perf.to_latex(index=False))


# In[56]:


with open('../MA/results/sequence-level/results_seq_multiple_non_onoff_perf' + str(k) + '.pickle', 'wb') as handle:
    pickle.dump(results_seq_multiple_non_onoff_perf, handle)

