#!/usr/bin/env python
# coding: utf-8

# In[124]:


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


# In[125]:


plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
font = {'size': 22, 'weight': 'bold'}
get_ipython().run_line_magic('matplotlib', 'inline')


# In[126]:


os.chdir('C:\\Users\\Simon\\Desktop\\MA\\session-rec')


# In[127]:


folder_figures = '../MA/figures/'


# In[128]:


fig_width = 12
fig_height = 8


# In[129]:


colors = {
    0: 'grey',
    1: 'darkgrey',
    2: 'lightgrey',
    3: 'lime',
    4: 'blue',
    5: 'darkblue',
    6: 'deepskyblue',
    7: 'aqua',
    8: 'red',
    9: 'orangered',
    10: 'orange'}
model_cat = ['co-occurrence frequency-based',
             'tree-based',
             'neural network-based',
             'neural network-based',
             'neural network-based',
             'nearest neighbor-based',
             'co-occurrence frequency-based',
             'co-occurrence frequency-based',
             'nearest neighbor-based',
             'nearest neighbor-based',
             'nearest neighbor-based']


# ### app-level

# In[130]:


datatype = 'app-level'


# ##### multiple performance

# In[131]:


with open('../MA/results/' + str(datatype) + '/results_app_multiple.pickle', 'rb') as handle:
    res = pickle.load(handle)


# In[132]:


res['model_cat'] = model_cat


# In[133]:


fig_name = 'HitRate_' + str(datatype) + '_multiple'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)
#fig.clf()
x = np.array([1, 5, 10, 20])
cols = ['HitRate@' + str(k) for k in x]
for model in res['model']:
    y = res[res['model'] == model][cols].values.flatten()
    ax.plot(x, y, label = model)
[ax.lines[i].set_color(colors[i]) for i in colors]

ax.set_xlabel('k')
ax.set_xticks([1, 5, 10, 20])
ax.set_xticklabels([1, 5, 10, 20])
ax.set_xlim(xmin=1, xmax=20)

ax.set_ylabel('HitRate@k')
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.legend(loc = 'lower right', ncol = 3)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')


# ##### minimum sequence length

# In[134]:


with open('../MA/results/' + str(datatype) + '/results_app_multiple_min20.pickle', 'rb') as handle:
    res_min20 = pickle.load(handle)
# with open('../MA/results/' + str(datatype) + '/results_app_multiple_min20_test.pickle', 'rb') as handle:
#     res_min20_test = pickle.load(handle)


# In[135]:


fig_name = 'HitRate_' + str(datatype) + '_multiple_min20'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
legend = ['all', 'min 20', 'minimum sequence length 20 test']
df = res.copy(deep=True)
df.rename(columns={
    'HitRate@1': str(legend[0]) + '_HitRate@1',
    'HitRate@5': str(legend[0]) + '_HitRate@5',
    'HitRate@10': str(legend[0]) + '_HitRate@10',
    'HitRate@20': str(legend[0]) + '_HitRate@20',}, inplace=True)
df[str(legend[1]) + '_HitRate@1'] = res_min20['HitRate@1']
df[str(legend[1]) + '_HitRate@5'] = res_min20['HitRate@5']
df[str(legend[1]) + '_HitRate@10'] = res_min20['HitRate@10']
df[str(legend[1]) + '_HitRate@20'] = res_min20['HitRate@20']

# df[str(legend[2]) + '_HitRate@1'] = res_min20_test['HitRate@1']
# df[str(legend[2]) + '_HitRate@5'] = res_min20_test['HitRate@5']
# df[str(legend[2]) + '_HitRate@10'] = res_min20_test['HitRate@10']
# df[str(legend[2]) + '_HitRate@20'] = res_min20_test['HitRate@20']

fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)

all_seq = [plt.bar(df['model'], df[str(legend[0]) + '_HitRate@20'],
                     align='edge', width= -0.2, color='lightgrey',
                     label='k = 20 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@10'],
                     align='edge', width= -0.2, color='darkgrey',
                     label='k = 10 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@5'],
                     align='edge', width= -0.2, color='dimgray',
                     label='k = 5 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@1'],
                     align='edge', width= -0.2, color='black',
                     label='k = 1 (' + str(legend[0]) + ')')]

min20 =   [plt.bar(df['model'], df[str(legend[1]) + '_HitRate@20'],
                     align='edge', width= 0.2, color='lightsteelblue',
                     label='k = 20 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@10'],
                     align='edge', width= 0.2, color='deepskyblue',
                     label='k = 10 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@5'],
                     align='edge', width= 0.2, color='royalblue',
                     label='k = 5 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@1'],
                     align='edge', width= 0.2, color='mediumblue',
                     label='k = 1 (' + str(legend[1]) + ')')]

# min20_test =   [plt.bar(df['model'], df[str(legend[2]) + '_HitRate@20'],
#                      align='edge', width= -0.4, color='lightcoral',
#                      label='k = 20 (' + str(legend[2]) + ')'),
#              plt.bar(df['model'], df[str(legend[2]) + '_HitRate@10'],
#                      align='edge', width= -0.4, color='orangered',
#                      label='k = 10 (' + str(legend[2]) + ')'),
#              plt.bar(df['model'], df[str(legend[2]) + '_HitRate@5'],
#                      align='edge', width= -0.4, color='red',
#                      label='k = 5 (' + str(legend[2]) + ')'),
#              plt.bar(df['model'], df[str(legend[2]) + '_HitRate@1'],
#                      align='edge', width= -0.4, color='darkred',
#                      label='k = 1 (' + str(legend[2]) + ')')]

ax.tick_params(axis='x', rotation=30)
ax.set_ylabel('HitRate@k')
ax.legend(loc='upper center', ncol=4)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')


# ##### performance by position

# In[ ]:


with open('../MA/results/' + str(datatype) + '/results_app_multiple_pos_HR@1.pickle', 'rb') as handle:
    res_pos = pickle.load(handle)


# In[ ]:


fig_name = 'HitRate@1_' + str(datatype) + '_multiple_by_position'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)
#fig.clf()
x = np.linspace(1,10,10, dtype=int)
cols = ['position = ' + str(k) for k in x]
for model in res_pos['model']:
    y = res_pos[res_pos['model'] == model][cols].values.flatten()
    ax.plot(x, y, label = model)
[ax.lines[i].set_color(colors[i]) for i in colors]

ax.set_xlabel('Prediction Position in Test Sequence')
ax.set_xticks(np.linspace(1,10,10, dtype=int))
ax.set_xticklabels(np.linspace(1,10,10, dtype=int))
ax.set_xlim(xmin=1, xmax=10)

ax.set_ylabel('HitRate@1')
ax.set_yticks([0, 0.1, 0.2, 0.3])

ax.legend(loc = 'upper right', ncol = 3)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')


# ##### multiple category-level performance versus multiple performance

# In[ ]:


with open('../MA/results/' + str(datatype) + '/results_app_multiple_cat.pickle', 'rb') as handle:
    res_cat = pickle.load(handle)


# In[ ]:


fig_name = 'HitRate_' + str(datatype) + '_multiple_categories'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
legend = ['apps', 'categories']
df = res.copy(deep=True)
df.rename(columns={
    'HitRate@1': str(legend[0]) + '_HitRate@1',
    'HitRate@5': str(legend[0]) + '_HitRate@5',
    'HitRate@10': str(legend[0]) + '_HitRate@10',
    'HitRate@20': str(legend[0]) + '_HitRate@20',}, inplace=True)
df[str(legend[1]) + '_HitRate@1'] = res_cat['HitRate@1']
df[str(legend[1]) + '_HitRate@5'] = res_cat['HitRate@5']
df[str(legend[1]) + '_HitRate@10'] = res_cat['HitRate@10']
df[str(legend[1]) + '_HitRate@20'] = res_cat['HitRate@20']

fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)

undropped = [plt.bar(df['model'], df[str(legend[0]) + '_HitRate@20'],
                     align='edge', width= -0.2, color='lightgrey',
                     label='k = 20 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@10'],
                     align='edge', width= -0.2, color='darkgrey',
                     label='k = 10 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@5'],
                     align='edge', width= -0.2, color='dimgray',
                     label='k = 5 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@1'],
                     align='edge', width= -0.2, color='black',
                     label='k = 1 (' + str(legend[0]) + ')')]

dropped =   [plt.bar(df['model'], df[str(legend[1]) + '_HitRate@20'],
                     align='edge', width= +0.2, color='lightsteelblue',
                     label='k = 20 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@10'],
                     align='edge', width= +0.2, color='deepskyblue',
                     label='k = 10 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@5'],
                     align='edge', width= +0.2, color='royalblue',
                     label='k = 5 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@1'],
                     align='edge', width= +0.2, color='mediumblue',
                     label='k = 1 (' + str(legend[1]) + ')')]

ax.tick_params(axis='x', rotation=30)
ax.set_ylabel('HitRate@k')
ax.legend(loc='upper center', ncol=4)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')


# In[ ]:


# fig_name = 'HitRate@1_' + str(datatype) + '_multiple_categories'

# fig, ax = plt.subplots()
# fig.set_figwidth(fig_width)
# fig.set_figheight(fig_height)
# ax.bar(res['model'], res['HitRate@1'], width=0.35, label='individual apps', color = 'grey')
# ax.bar(res['model'], res_cat['HitRate@1']-res['HitRate@1'], width=0.35, bottom=res['HitRate@1'], label='app categories',
#       color = 'lightgrey')

# ax.tick_params(axis='x', rotation=45)
# ax.set_ylabel('HitRate@1')
# ax.legend()

# plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300)


# ##### multiple performance after dropping ON & OFF

# In[136]:


with open('../MA/results/' + str(datatype) + '/results_app_multiple_droponoff.pickle', 'rb') as handle:
    res_drop = pickle.load(handle)


# In[137]:


fig_name = 'HitRate_' + str(datatype) + '_multiple_droponoff'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
legend = ['all', 'no ON & OFF']
df = res.copy(deep=True)
df.rename(columns={
    'HitRate@1': str(legend[0]) + '_HitRate@1',
    'HitRate@5': str(legend[0]) + '_HitRate@5',
    'HitRate@10': str(legend[0]) + '_HitRate@10',
    'HitRate@20': str(legend[0]) + '_HitRate@20',}, inplace=True)
df[str(legend[1]) + '_HitRate@1'] = res_drop['HitRate@1']
df[str(legend[1]) + '_HitRate@5'] = res_drop['HitRate@5']
df[str(legend[1]) + '_HitRate@10'] = res_drop['HitRate@10']
df[str(legend[1]) + '_HitRate@20'] = res_drop['HitRate@20']

fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)

undropped = [plt.bar(df['model'], df[str(legend[0]) + '_HitRate@20'],
                     align='edge', width= -0.2, color='lightgrey',
                     label='k = 20 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@10'],
                     align='edge', width= -0.2, color='darkgrey',
                     label='k = 10 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@5'],
                     align='edge', width= -0.2, color='dimgray',
                     label='k = 5 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@1'],
                     align='edge', width= -0.2, color='black',
                     label='k = 1 (' + str(legend[0]) + ')')]

dropped =   [plt.bar(df['model'], df[str(legend[1]) + '_HitRate@20'],
                     align='edge', width= +0.2, color='lightsteelblue',
                     label='k = 20 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@10'],
                     align='edge', width= +0.2, color='deepskyblue',
                     label='k = 10 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@5'],
                     align='edge', width= +0.2, color='royalblue',
                     label='k = 5 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@1'],
                     align='edge', width= +0.2, color='mediumblue',
                     label='k = 1 (' + str(legend[1]) + ')')]

ax.tick_params(axis='x', rotation=30)
ax.set_ylabel('HitRate@k')
ax.legend(loc='upper center', ncol=4)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')


# In[138]:


# fig_name = 'HitRate@1_' + str(datatype) + '_multiple_droponoff'

# legend = ['all data', 'ON & OFF dropped']
# df = res.copy(deep=True)
# df.rename(columns={'HitRate@1': legend[0]}, inplace=True)
# df[legend[1]] = res_drop['HitRate@1']
# plot = df.plot(x='model', y=legend, kind="bar", rot=45, color = ['grey', 'lightgrey'], figsize = (fig_width, fig_height))
# plot.set_xlabel("")
# plot.set_ylabel("HitRate@1")
# fig = plot.get_figure()
# fig.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300)


# ### sequence-level

# In[139]:


datatype = 'sequence-level'


# ##### multiple performance

# In[140]:


with open('../MA/results/' + str(datatype) + '/results_seq_multiple.pickle', 'rb') as handle:
    res = pickle.load(handle)


# In[141]:


res['model_cat'] = model_cat


# In[142]:


fig_name = 'HitRate_' + str(datatype) + '_multiple'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)
#fig.clf()
x = np.array([1, 5, 10, 20])
cols = ['HitRate@' + str(k) for k in x]
for model in res['model']:
    y = res[res['model'] == model][cols].values.flatten()
    ax.plot(x, y, label = model)
[ax.lines[i].set_color(colors[i]) for i in colors]

ax.set_xlabel('k')
ax.set_xticks([1, 5, 10, 20])
ax.set_xticklabels([1, 5, 10, 20])
ax.set_xlim(xmin=1, xmax=20)

ax.set_ylabel('HitRate@k')
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])


ax.legend(loc = 'lower right', ncol = 3)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')


# ##### multiple performance after dropping ON & OFF

# In[143]:


with open('../MA/results/' + str(datatype) + '/results_seq_multiple_droponoff.pickle', 'rb') as handle:
    res_drop = pickle.load(handle)


# In[144]:


fig_name = 'HitRate_' + str(datatype) + '_multiple_droponoff'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
legend = ['all', 'no ON-OFF']
df = res.copy(deep=True)
df.rename(columns={
    'HitRate@1': str(legend[0]) + '_HitRate@1',
    'HitRate@5': str(legend[0]) + '_HitRate@5',
    'HitRate@10': str(legend[0]) + '_HitRate@10',
    'HitRate@20': str(legend[0]) + '_HitRate@20',}, inplace=True)
df[str(legend[1]) + '_HitRate@1'] = res_drop['HitRate@1']
df[str(legend[1]) + '_HitRate@5'] = res_drop['HitRate@5']
df[str(legend[1]) + '_HitRate@10'] = res_drop['HitRate@10']
df[str(legend[1]) + '_HitRate@20'] = res_drop['HitRate@20']

fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)

undropped = [plt.bar(df['model'], df[str(legend[0]) + '_HitRate@20'],
                     align='edge', width= -0.2, color='lightgrey',
                     label='k = 20 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@10'],
                     align='edge', width= -0.2, color='darkgrey',
                     label='k = 10 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@5'],
                     align='edge', width= -0.2, color='dimgray',
                     label='k = 5 (' + str(legend[0]) + ')'),
             plt.bar(df['model'], df[str(legend[0]) + '_HitRate@1'],
                     align='edge', width= -0.2, color='black',
                     label='k = 1 (' + str(legend[0]) + ')')]

dropped =   [plt.bar(df['model'], df[str(legend[1]) + '_HitRate@20'],
                     align='edge', width= +0.2, color='lightsteelblue',
                     label='k = 20 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@10'],
                     align='edge', width= +0.2, color='deepskyblue',
                     label='k = 10 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@5'],
                     align='edge', width= +0.2, color='royalblue',
                     label='k = 5 (' + str(legend[1]) + ')'),
             plt.bar(df['model'], df[str(legend[1]) + '_HitRate@1'],
                     align='edge', width= +0.2, color='mediumblue',
                     label='k = 1 (' + str(legend[1]) + ')')]

ax.tick_params(axis='x', rotation=30)
ax.set_ylabel('HitRate@k')
ax.legend(loc='upper center', ncol=4)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')


# In[145]:


# fig_name = 'HitRate@1_' + str(datatype) + '_multiple_droponoff'

# legend = ['all data', 'ON & OFF dropped']
# df = res.copy(deep=True)
# df.rename(columns={'HitRate@1': legend[0]}, inplace=True)
# df[legend[1]] = res_drop['HitRate@1']
# plot = df.plot(x='model', y=legend, kind="bar", rot=45, color = ['grey', 'lightgrey'], figsize = (fig_width, fig_height))
# plot.set_xlabel("")
# plot.set_ylabel("HitRate@1")
# fig = plot.get_figure()
# fig.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300)


# ##### performance by position

# In[146]:


with open('../MA/results/' + str(datatype) + '/results_seq_multiple_pos_HR@1.pickle', 'rb') as handle:
    res_pos = pickle.load(handle)


# In[147]:


fig_name = 'HitRate@1_' + str(datatype) + '_multiple_by_position'
matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"
fig, ax = plt.subplots()
fig.set_figwidth(fig_width)
fig.set_figheight(fig_height)
#fig.clf()
x = np.linspace(1,10,10, dtype=int)
cols = ['position = ' + str(k) for k in x]
for model in res_pos['model']:
    y = res_pos[res_pos['model'] == model][cols].values.flatten()
    ax.plot(x, y, label = model)
[ax.lines[i].set_color(colors[i]) for i in colors]

ax.set_xlabel('Prediction Position in Test Sequence')
ax.set_xticks(np.linspace(1,10,10, dtype=int))
ax.set_xticklabels(np.linspace(1,10,10, dtype=int))
ax.set_xlim(xmin=1, xmax=10)

ax.set_ylabel('HitRate@1')
ax.set_yticks([0, 0.1, 0.2, 0.3])

ax.legend(loc = 'upper right', ncol = 3)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.savefig(folder_figures + fig_name + '.png', format = 'png', dpi=300, bbox_inches='tight')

