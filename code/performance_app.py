#!/usr/bin/env python
# coding: utf-8

# ### setup

# In[43]:


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


# In[44]:


os.chdir('C:\\Users\\Simon\\Desktop\\MA\\session-rec')


# In[45]:


# datatypes = ['app-level', 'sequence-level']
datatype = 'app-level'
windows = [1,2,3,4,5]
model_index = [0, 3, 8, 9, 10, 4, 1, 2, 5, 6, 7]
model_name = 'Algorithm'


# In[46]:


USER_KEY = 'userID'
TIME_KEY = 'timestamp'
if datatype == 'app-level':    
    ITEM_KEY = 'appID'
    SESSION_KEY = 'sessionID'
else:
    ITEM_KEY = 'usID'
    SESSION_KEY = 'sentenceID'


# ### helper functions

# In[47]:


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


# In[48]:


# for single window
# get performance for a given algorithm
def get_perf(file, key):
    df = pd.read_csv(folder_res + file, sep = ';')
    df.drop(['Metrics', 'Saver@50: '], axis=1, inplace=True)
    df.drop(df.filter(regex='Unnamed'), axis=1, inplace=True) # drop 'Unnamed: 24' column containing only NaNs
    df.rename(columns = lambda x : str(x)[:-2], inplace=True) # remove colon and whitespace from all column names
    df.insert(0, model_name, key)
    return(df)


# In[49]:


# extract ground truth from test data (test_data) for a single item (position) in a single session (sessionId)
def extract_ground_truth(ID, position, test_data):
    relevant_df = test_data[test_data[SESSION_KEY]==ID]
    index = relevant_df.index[position+1]
    ground_truth = relevant_df[ITEM_KEY][index]
    return ground_truth


# In[50]:


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


# In[51]:


# helper function outputting whether ground truth is in recommendation list of length k for a single algorithm and item
def calc_hr_k(ground_truth, rec_list, k):
    if type(ground_truth) == int or len(ground_truth) == 1:
        return ground_truth in rec_list[:k]
    elif len(ground_truth) > 1:
        return any(x in ground_truth for x in rec_list[:k])
    else:
        return None


# In[52]:


# helper function for calculating the MRR
def calc_mrr_k(ground_truth, rec_list, k):
    if ground_truth not in rec_list[:k]:
        return 0
    else:
        score = rec_list.index(ground_truth) + 1
        return 1/score


# In[53]:


def print_predictions(predictions, sessionID, num_recs, positions, models):
    # predictions must contain columns named 'sessionID' and 'position', containing the respective values
    predictions_dict = {}
    for pos in positions:
        row = predictions[(predictions.sessionID == sessionID) & (predictions.position == pos)]
        ground_truth = row.ground_truth_name.to_string(index=False)
#         print('sessionID: ' + str(sessionID) + ', position: ' + str(pos))
#         print('ground truth: ' + str(row.ground_truth_name.to_string(index=False)))
        df = pd.DataFrame()
        for model in models:
            df[model] = [row['recs_names-' + model].tolist()[0][i] for i in range(num_recs)]
        name = str(sessionID) + '_' + str(pos)
        predictions_dict[name] = (sessionID, pos, ground_truth, df)
    return predictions_dict


# In[54]:


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


# In[55]:


# def highlight_max(x):
#     return ['font-weight: bold' if v==x.max() and x.name!=model_name and x.name!='Coverage@20' and x.name!='Popularity@20' 
#             else '' for v in x]


# ### multiple windows

# ##### overall

# In[78]:


folder_res = 'results/testing/' + str(datatype) + '/multiple/'
algos = set([f.split('_window')[0].split('test_single_')[1] for f in os.listdir(folder_res)])
algos -= {'vsknn', 'vsknn_EBR'}
results_app_multiple = pd.DataFrame()
for key in algos:
    files = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) 
             and (f.startswith('test_single_' + str(key) + '_window'))
             and ('min20' not in f)]
    res = get_av_perf(files, key)
    results_app_multiple = results_app_multiple.append(res)
results_app_multiple = capitalize_names(results_app_multiple)
results_app_multiple = results_app_multiple.round(4)
results_app_multiple = results_app_multiple.sort_values(model_name)
results_app_multiple['model_index'] = model_index
results_app_multiple = results_app_multiple.set_index('model_index').sort_index()
results_app_multiple.index.name = None


# In[79]:


results_app_multiple


# In[66]:


with open('../MA/tables/results_app_multiple.tex','w') as tf:
    tf.write(results_app_multiple.to_latex(index=False))


# In[16]:


with open('../MA/results/app-level/results_app_multiple.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple, handle)


# ##### min20

# Prediction on long sessions (20+) if trained on long sessions only (no extra tuning), implying a minimum sequence length of 20 for both training and test data:

# In[17]:


folder_res = 'results/testing/' + str(datatype) + '/multiple/'
algos = set([f.split('_window')[0].split('test_single_')[1] for f in os.listdir(folder_res)])
algos -= {'vsknn', 'vsknn_EBR'}
results_app_multiple_min20 = pd.DataFrame()
for key in algos:
    files = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) and (f.startswith('test_single_' + str(key) + '_window'))
             and ('min20' in f) and ('min20_test' not in f)]
    res = get_av_perf(files, key)
    results_app_multiple_min20 = results_app_multiple_min20.append(res)
results_app_multiple_min20 = capitalize_names(results_app_multiple_min20)
results_app_multiple_min20 = results_app_multiple_min20.round(4)
results_app_multiple_min20 = results_app_multiple_min20.sort_values(model_name)
results_app_multiple_min20['model_index'] = model_index
results_app_multiple_min20 = results_app_multiple_min20.set_index('model_index').sort_index()
results_app_multiple_min20.index.name = None


# In[18]:


results_app_multiple_min20


# In[19]:


with open('../MA/tables/results_app_multiple_min20.tex','w') as tf:
    tf.write(results_app_multiple_min20.to_latex(index=False))


# In[20]:


with open('../MA/results/app-level/results_app_multiple_min20.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_min20, handle)


# Prediction on long sessions (20+) if trained on all sessions, implying a minimum sequence length of 20 for test data only:

# In[21]:


algos = set([f.split('_window')[0].split('test_single_')[1] for f in os.listdir(folder_res)])
algos -= {'vsknn', 'vsknn_EBR'}
results_app_multiple_min20_test = pd.DataFrame()
for key in algos:
    files = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) and (f.startswith('test_single_' + str(key) + '_window')) and ('min20_test' in f)]
    res = get_av_perf(files, key)
    results_app_multiple_min20_test = results_app_multiple_min20_test.append(res)
results_app_multiple_min20_test = capitalize_names(results_app_multiple_min20_test)
results_app_multiple_min20_test = results_app_multiple_min20_test.round(4)
results_app_multiple_min20_test = results_app_multiple_min20_test.sort_values(model_name)
results_app_multiple_min20_test['model_index'] = model_index
results_app_multiple_min20_test = results_app_multiple_min20_test.set_index('model_index').sort_index()
results_app_multiple_min20_test.index.name = None


# In[22]:


results_app_multiple_min20_test


# In[23]:


with open('../MA/tables/results_app_multiple_min20_test.tex','w') as tf:
    tf.write(results_app_multiple_min20_test.to_latex(index=False))


# In[24]:


with open('../MA/results/app-level/results_app_multiple_min20_test.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_min20_test, handle)


# In[25]:


results_app_multiple_min20_diff = results_app_multiple_min20_test.copy(deep=True)
for col in results_app_multiple_min20_diff.columns:
    if col != model_name:
        results_app_multiple_min20_diff[col] = results_app_multiple_min20_diff[col] - results_app_multiple_min20[col]


# In[26]:


results_app_multiple_min20_diff


# In[27]:


with open('../MA/tables/results_app_multiple_min20_diff.tex','w') as tf:
    tf.write(results_app_multiple_min20_diff.to_latex(index=False))


# In[28]:


with open('../MA/results/app-level/results_app_multiple_min20_diff.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_min20_diff, handle)


# ### single window

# ##### overall

# In[29]:


folder_res = 'results/testing/' + str(datatype) + '/single/'
algos = set([f.split('test_single_')[1].split('_single')[0] for f in os.listdir(folder_res) if f.startswith('test_single')])
algos -= {'vsknn', 'vsknn_EBR'}
results_app_single = pd.DataFrame()
for key in algos:
    file = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) 
             and (f.startswith('test_single_' + str(key) + '_single'))
             and ('min20' not in f)
             and ('embedding' not in f)][0] # list is of length 1 actually
    res = get_perf(file, key)
    results_app_single = results_app_single.append(res)
results_app_single = capitalize_names(results_app_single)
results_app_single = results_app_single.round(4)
results_app_single = results_app_single.sort_values(model_name)
results_app_single['model_index'] = model_index
results_app_single = results_app_single.set_index('model_index').sort_index()
results_app_single.index.name = None


# In[30]:


results_app_single


# In[31]:


with open('../MA/tables/results_app_single.tex','w') as tf:
    tf.write(results_app_single.to_latex(index=False))


# In[32]:


with open('../MA/results/app-level/results_app_single.pickle', 'wb') as handle:
    pickle.dump(results_app_single, handle)


# In[33]:


results_app_single_diff = results_app_single.copy(deep=True)
for col in results_app_single_diff.columns:
    if col != model_name:
        results_app_single_diff[col] = results_app_single_diff[col] - results_app_multiple[col]


# In[34]:


results_app_single_diff


# In[35]:


with open('../MA/tables/results_app_single_diff.tex','w') as tf:
    tf.write(results_app_single_diff.to_latex(index=False))


# In[36]:


with open('../MA/results/app-level/results_app_single_diff.pickle', 'wb') as handle:
    pickle.dump(results_app_single_diff, handle)


# ### performance by position

# ##### create mapping dicts

# In[63]:


folder_res = 'results/testing/' + str(datatype) + '/multiple/'
folder_data = folder_res.replace('results', 'data')
data = pd.read_csv('../data/app-level/data_app_nodrop.csv') # create app and user mappings
mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['app_name'])))])
mapping_reverse = dict((v,k) for k,v in mapping.items())


# ##### individual positions

# In[64]:


k = 20 # HR@k


# In[65]:


results_app_multiple_pos = pd.DataFrame()

for window in windows:
    test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
    predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                         and f.endswith('window_' + str(window) + '-Saver@50.csv')]
    predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
    algorithms = [i for i in predictions.columns if i.startswith('recs-')]
    algorithms.remove('recs-vsknn')
    algorithms.remove('recs-vsknn_EBR')

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
    results_app_multiple_pos = results_app_multiple_pos.append(perf_by_pos)

results_app_multiple_pos = results_app_multiple_pos.groupby('position').mean() # average across positions
results_app_multiple_pos.drop(['window'], axis=1, inplace=True)
results_app_multiple_pos = results_app_multiple_pos.transpose() # transpose to have algorithms as rows, positions as columns
columns_reordered = results_app_multiple_pos.columns.tolist()
columns_reordered.sort(key=lambda x: int(re.search(r'\d+$',x).group()))
results_app_multiple_pos = results_app_multiple_pos[columns_reordered]
results_app_multiple_pos.reset_index(inplace=True) # convert index to column named "index"
results_app_multiple_pos.rename(columns={'index': model_name}, inplace=True) # rename column "index" to "model"
results_app_multiple_pos.rename_axis(None, axis=1, inplace=True) # unname new index
results_app_multiple_pos = capitalize_names(results_app_multiple_pos) # adjust model names
results_app_multiple_pos = results_app_multiple_pos.round(4)
results_app_multiple_pos = results_app_multiple_pos.sort_values(model_name)
results_app_multiple_pos['model_index'] = model_index
results_app_multiple_pos = results_app_multiple_pos.set_index('model_index').sort_index()
results_app_multiple_pos.index.name = None


# In[66]:


results_app_multiple_pos


# In[67]:


with open('../MA/tables/results_app_multiple_pos_HR@' + str(k) + '.tex','w') as tf:
    tf.write(results_app_multiple_pos.to_latex(index=False))


# In[68]:


with open('../MA/results/app-level/results_app_multiple_pos_HR@' + str(k) + '.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_pos, handle)


# ##### cutoffs

# In[69]:


cutoffs = [2, 5, 10]
k = 20 # HR@k


# In[70]:


results_app_multiple_cutoff = pd.DataFrame()

for cutoff in cutoffs:
    for window in windows:
        test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
        predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                             and f.endswith('window_' + str(window) + '-Saver@50.csv')]
        predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
        algorithms = [i for i in predictions.columns if i.startswith('recs-')]
        algorithms.remove('recs-vsknn')
        algorithms.remove('recs-vsknn_EBR')

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
        results_app_multiple_cutoff = results_app_multiple_cutoff.append(perf_by_pos)

results_app_multiple_cutoff = results_app_multiple_cutoff.groupby('position').mean() # average across positions (e.g., "<= 2", "> 10")
results_app_multiple_cutoff.drop(['window'], axis=1, inplace=True)
results_app_multiple_cutoff = results_app_multiple_cutoff.transpose() # transpose to have algorithms as rows, positions as columns
columns_reordered = results_app_multiple_cutoff.columns.tolist()
columns_reordered.sort(key=lambda x: int(re.search(r'\d+$',x).group()))
results_app_multiple_cutoff = results_app_multiple_cutoff[columns_reordered]
results_app_multiple_cutoff.reset_index(inplace=True) # convert index to column named "index"
results_app_multiple_cutoff.rename(columns={'index': model_name}, inplace=True) # rename column "index" to "model"
results_app_multiple_cutoff.rename_axis(None, axis=1, inplace=True) # unname new index
results_app_multiple_cutoff = capitalize_names(results_app_multiple_cutoff) # adjust model names
results_app_multiple_cutoff = results_app_multiple_cutoff.round(4)
results_app_multiple_cutoff = results_app_multiple_cutoff.sort_values(model_name)
results_app_multiple_cutoff['model_index'] = model_index
results_app_multiple_cutoff = results_app_multiple_cutoff.set_index('model_index').sort_index()
results_app_multiple_cutoff.index.name = None


# In[71]:


results_app_multiple_cutoff


# In[72]:


with open('../MA/tables/results_app_multiple_cutoff_HR@' + str(k) + '.tex','w') as tf:
    tf.write(results_app_multiple_cutoff.to_latex(index=False))


# In[73]:


with open('../MA/results/app-level/results_app_multiple_cutoff_HR@' + str(k) + '.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_cutoff, handle)


# ##### percentage of OFF predictions by position

# In[48]:


k = 1 # HR@k
OFF = [mapping['OFF_LOCKED'], mapping['OFF_UNLOCKED']]


# In[49]:


results_app_multiple_off = pd.DataFrame()

for window in windows:
    test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
    predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                         and f.endswith('window_' + str(window) + '-Saver@50.csv')]
    predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
    algorithms = [i for i in predictions.columns if i.startswith('recs-')]
    algorithms.remove('recs-vsknn')
    algorithms.remove('recs-vsknn_EBR')

    perf_by_pos = pd.DataFrame()
    positions = range(1,11)
    for pos in positions:
        pred_pos = predictions[predictions['position']==pos-1]
        df = pd.DataFrame()
        df['position'] = ['position = ' + str(pos)]
        df['window'] = [window]
        for algo in algorithms:
            algo_name = ''.join(algo.split('-')[1:])
            value = pred_pos.apply(lambda x: calc_hr_k(OFF, x[algo], k), axis=1).sum()/len(pred_pos)
            df[algo_name] = [value]
        perf_by_pos = perf_by_pos.append(df).reset_index(drop=True)
    results_app_multiple_off = results_app_multiple_off.append(perf_by_pos)

results_app_multiple_off = results_app_multiple_off.groupby('position').mean() # average across positions
results_app_multiple_off.drop(['window'], axis=1, inplace=True)
results_app_multiple_off = results_app_multiple_off.transpose() # transpose to have algorithms as rows, positions as columns
columns_reordered = results_app_multiple_off.columns.tolist()
columns_reordered.sort(key=lambda x: int(re.search(r'\d+$',x).group()))
results_app_multiple_off = results_app_multiple_off[columns_reordered]
results_app_multiple_off.reset_index(inplace=True) # convert index to column named "index"
results_app_multiple_off.rename(columns={'index': model_name}, inplace=True) # rename column "index" to "model"
results_app_multiple_off.rename_axis(None, axis=1, inplace=True) # unname new index
results_app_multiple_off = capitalize_names(results_app_multiple_off) # adjust model names
results_app_multiple_off = results_app_multiple_off.round(4)
results_app_multiple_off = results_app_multiple_off.sort_values(model_name)
results_app_multiple_off['model_index'] = model_index
results_app_multiple_off = results_app_multiple_off.set_index('model_index').sort_index()
results_app_multiple_off.index.name = None


# In[50]:


results_app_multiple_off


# In[51]:


with open('../MA/tables/results_app_multiple_pos_off' + str(k) + '.tex','w') as tf:
    tf.write(results_app_multiple_off.to_latex(index=False))


# In[52]:


with open('../MA/results/app-level/results_app_multiple_pos_off.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_off, handle)


# ### performance by category

# ##### create mapping dicts

# In[53]:


folder_res = 'results/testing/' + str(datatype) + '/multiple/'
folder_data = folder_res.replace('results', 'data')
data = pd.read_csv('../data/app-level/data_app_nodrop.csv') # create app and user mappings
mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(data['app_name'])))])
mapping_reverse = dict((v,k) for k,v in mapping.items())

# category_mapping = {}
# for app in data.app_name.value_counts().index:
#     if app not in category_mapping:
#         cat = data.category[data.app_name==app].iloc[0]
#         category_mapping[app] = cat
        
# with open('../data/app-level/category_mapping.pickle', 'wb') as handle:
#     pickle.dump(category_mapping, handle)

with open('../data/app-level/category_mapping.pickle', 'rb') as handle:
    category_mapping = pickle.load(handle)


# ##### category-level prediction

# Now, we also have to convert the recommendations to category-level. Furthermore, we now have to match based on names as we cannot use token IDs anymore.

# In[54]:


windows = [1,2,3,4,5]
ks = [1,5,10,20]
metrics = ['HitRate', 'MRR']


# In[55]:


results_app_multiple_cat_combined = pd.DataFrame()
for metric in metrics:
    results_app_multiple_cat = pd.DataFrame()
    for k in ks:

        perf_by_cat = pd.DataFrame()
        for window in windows:
            test_data = pd.read_hdf(str(folder_data) + 'window_' + str(window) + '.hdf', 'test') 
            predictions_files = [f for f in os.listdir(folder_res) if ('min20' not in f) 
                                 and f.endswith('window_' + str(window) + '-Saver@50.csv')]
            predictions = generate_predictions(predictions_files, test_data, mapping_reverse)
            predictions['ground_truth_category_name'] = predictions['ground_truth_name'].apply(lambda x: category_mapping[x])
            algorithms_names = [i for i in predictions.columns if i.startswith('recs_names-')]
            algorithms_names.remove('recs_names-vsknn')
            algorithms_names.remove('recs_names-vsknn_EBR')

            df = pd.DataFrame()
            for algo in algorithms_names:
                col_name = 'recs_names_cat-' + algo.split('recs_names-')[1]
                algo_name = ''.join(algo.split('-')[1:])
                predictions[col_name] = predictions[algo].apply(lambda x: [category_mapping[i] for i in x])
                if metric == metrics[0]: # HitRate
                    value = predictions.apply(lambda x: 
                                          calc_hr_k(x['ground_truth_category_name'], x[col_name], k), axis=1).sum()/len(predictions)
                else: # MRR
                     value = predictions.apply(lambda x: 
                                          calc_mrr_k(x['ground_truth_category_name'], x[col_name], k), axis=1).sum()/len(predictions)                   
                df[algo_name] = [value]
            perf_by_cat = perf_by_cat.append(df)
        perf_by_cat = pd.DataFrame(perf_by_cat.mean()) # average across windows
        perf_by_cat.rename(columns={0: str(metric) + '@' + str(k)}, inplace=True)

        if results_app_multiple_cat.shape == (0,0):
            results_app_multiple_cat = results_app_multiple_cat.append(perf_by_cat)
        else:
            results_app_multiple_cat = results_app_multiple_cat.merge(perf_by_cat, left_index=True, right_index=True)


    results_app_multiple_cat.reset_index(inplace=True) # convert index to column named "index"
    results_app_multiple_cat.rename(columns={'index': model_name}, inplace=True) # rename column "index" to "model"
    results_app_multiple_cat.rename_axis(None, axis=1, inplace=True) # unname new index
    results_app_multiple_cat = capitalize_names(results_app_multiple_cat) # adjust model names
    results_app_multiple_cat = results_app_multiple_cat.round(4)
    results_app_multiple_cat = results_app_multiple_cat.sort_values(model_name)
    results_app_multiple_cat['model_index'] = model_index
    results_app_multiple_cat = results_app_multiple_cat.set_index('model_index').sort_index()
    results_app_multiple_cat.index.name = None

    if results_app_multiple_cat_combined.shape == (0,0):
        results_app_multiple_cat_combined = results_app_multiple_cat_combined.append(results_app_multiple_cat)
    else:
        results_app_multiple_cat_combined = results_app_multiple_cat_combined.merge(results_app_multiple_cat, left_on=model_name, right_on=model_name)

if 1 in ks:
    results_app_multiple_cat_combined.drop(['MRR@1'], axis=1, inplace=True)


# In[56]:


results_app_multiple_cat_combined


# In[57]:


with open('../MA/tables/results_app_multiple_cat' + '.tex','w') as tf:
    tf.write(results_app_multiple_cat_combined.to_latex(index=False))


# In[58]:


with open('../MA/results/app-level/results_app_multiple_cat.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_cat_combined, handle)


# ### removing on and off

# ##### multiple windows

# In[59]:


folder_res = 'results/testing_onoff_unspecific_tuning/' + str(datatype) + '/multiple/'
algos = set([f.split('_window')[0].split('test_single_')[1] for f in os.listdir(folder_res)])
algos -= {'vsknn', 'vsknn_EBR'}
results_app_multiple_droponoff = pd.DataFrame()
for key in algos:
    files = [f for f in os.listdir(folder_res) 
             if ('Saver' not in f) 
             and (f.startswith('test_single_' + str(key) + '_window'))
             and ('min20' not in f)]
    res = get_av_perf(files, key)
    results_app_multiple_droponoff = results_app_multiple_droponoff.append(res)
results_app_multiple_droponoff = capitalize_names(results_app_multiple_droponoff)
results_app_multiple_droponoff = results_app_multiple_droponoff.round(4)
results_app_multiple_droponoff = results_app_multiple_droponoff.sort_values(model_name)
results_app_multiple_droponoff['model_index'] = model_index
results_app_multiple_droponoff = results_app_multiple_droponoff.set_index('model_index').sort_index()
results_app_multiple_droponoff.index.name = None


# In[60]:


results_app_multiple_droponoff


# In[61]:


with open('../MA/tables/results_app_multiple_droponoff.tex','w') as tf:
    tf.write(results_app_multiple_droponoff.to_latex(index=False))


# In[62]:


with open('../MA/results/app-level/results_app_multiple_droponoff.pickle', 'wb') as handle:
    pickle.dump(results_app_multiple_droponoff, handle)

