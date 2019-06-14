#!/usr/bin/env python
# coding: utf-8

# In[3]:


path= r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\bitly_usagov\example.txt'

open(path).readline()


# In[4]:


import json

records= [json.loads(line) for line in open(path)]

records


# In[6]:


records[0]['g']


# In[9]:


#Time zone Count

time_zones= [rec['tz'] for rec in records if 'tz' in rec]
time_zones


# In[12]:





# In[13]:


len(time_zones)


# In[26]:


from collections import Counter

counts= Counter(time_zones)

counts.most_common(10)


# In[28]:


from pandas import DataFrame, Series
import pandas as pd

frame= DataFrame(records)

frame


# In[32]:


frame['tz'].value_counts()[:10]


# In[34]:


clean_tz= frame['tz'].fillna("Missing")
clean_tz[clean_tz=='']= 'Unknown'

clean_tz.value_counts()[:10]


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')

clean_tz.value_counts()[:10].plot(kind='barh', rot=0)


# In[37]:


frame['a'][1]


# In[38]:


frame['a'][10]


# In[43]:


results= Series([x.split()[0] for x in frame.a.dropna()])
results.value_counts()[:8]


# In[46]:


cframe=frame[frame.a.notnull()]

import numpy as np
os= np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
os


# In[54]:


tz_os_grouping= cframe.groupby(['tz', os])
agg_counts=tz_os_grouping.size().unstack().fillna(0)
agg_counts


# In[69]:


indexer=agg_counts.sum(1).argsort()


# In[72]:


count_subset=agg_counts.take(indexer)[-10:]


# In[73]:


count_subset


# In[76]:


count_subset.plot(kind='barh', stacked=True)


# In[77]:


normed_subset= count_subset.div(count_subset.sum(1), axis=0)

normed_subset.plot(kind='barh', stacked=True)


# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[85]:


unames= ['user_id', 'gender', 'age', 'occupation', 'zip']
users= pd.read_table(r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\movielens\users.dat', 
                     sep='::', names=unames, header=None)

users[:10]


# In[91]:


rnames= ['user_id', 'movie_id', 'rating', 'timestamp']
ratings= pd.read_table(r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\movielens\ratings.dat', 
                     sep='::', names=rnames, header=None)
ratings[:10]


# In[92]:


mnames= ['movie_id', 'title', 'genres']
movies= pd.read_table(r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\movielens\movies.dat', 
                     sep='::', names=mnames, header=None)
movies[:10]


# In[97]:


data= pd.merge(pd.merge(ratings, users), movies)

data[:10]


# In[114]:


mean_ratings= data.pivot_table('rating', 'title', 'gender', aggfunc='mean')
mean_ratings[:10]


# In[109]:


ratings_by_title= data.groupby('title').size()
ratings_by_title[:10]


# In[113]:


active_titles= ratings_by_title.index[ratings_by_title>=250]
active_titles[:10]


# In[118]:


mean_ratings= mean_ratings.ix[active_titles]
mean_ratings


# In[120]:


top_female_ratings= mean_ratings.sort_index(by='F', ascending=False)
top_female_ratings[:10]


# In[124]:


mean_ratings['diff']= mean_ratings['M'] - mean_ratings['F']
sorted_by_diff= mean_ratings.sort_index(by='diff')
sorted_by_diff[:10]


# In[127]:


sorted_by_diff[::-1][:10]


# In[137]:


rating_std_by_title= data.groupby('title')['rating'].std()

rating_std_by_title= rating_std_by_title.ix[active_titles]


# In[138]:


rating_std_by_title.order(ascending=False)[:10]


# In[ ]:




