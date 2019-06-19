#!/usr/bin/env python
# coding: utf-8

# In[1]:


path= r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\bitly_usagov\example.txt'

open(path).readline()


# In[2]:


import json

records= [json.loads(line) for line in open(path)]

records


# In[3]:


records[0]['g']


# In[4]:


#Time zone Count

time_zones= [rec['tz'] for rec in records if 'tz' in rec]
time_zones


# In[ ]:





# In[5]:


len(time_zones)


# In[6]:


from collections import Counter

counts= Counter(time_zones)

counts.most_common(10)


# In[7]:


from pandas import DataFrame, Series
import pandas as pd

frame= DataFrame(records)

frame


# In[8]:


frame['tz'].value_counts()[:10]


# In[9]:


clean_tz= frame['tz'].fillna("Missing")
clean_tz[clean_tz=='']= 'Unknown'

clean_tz.value_counts()[:10]


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')

clean_tz.value_counts()[:10].plot(kind='barh', rot=0)


# In[11]:


frame['a'][1]


# In[12]:


frame['a'][10]


# In[13]:


results= Series([x.split()[0] for x in frame.a.dropna()])
results.value_counts()[:8]


# In[14]:


cframe=frame[frame.a.notnull()]

import numpy as np
os= np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
os


# In[15]:


tz_os_grouping= cframe.groupby(['tz', os])
agg_counts=tz_os_grouping.size().unstack().fillna(0)
agg_counts


# In[16]:


indexer=agg_counts.sum(1).argsort()


# In[17]:


count_subset=agg_counts.take(indexer)[-10:]


# In[18]:


count_subset


# In[19]:


count_subset.plot(kind='barh', stacked=True)


# In[20]:


normed_subset= count_subset.div(count_subset.sum(1), axis=0)

normed_subset.plot(kind='barh', stacked=True)


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


unames= ['user_id', 'gender', 'age', 'occupation', 'zip']
users= pd.read_table(r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\movielens\users.dat', 
                     sep='::', names=unames, header=None)

users[:10]


# In[23]:


rnames= ['user_id', 'movie_id', 'rating', 'timestamp']
ratings= pd.read_table(r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\movielens\ratings.dat', 
                     sep='::', names=rnames, header=None)
ratings[:10]


# In[24]:


mnames= ['movie_id', 'title', 'genres']
movies= pd.read_table(r'C:\Users\soumyama\Documents\Python Scripts\DataScienceBasics\pydata-book-2nd-edition\datasets\movielens\movies.dat', 
                     sep='::', names=mnames, header=None)
movies[:10]


# In[25]:


data= pd.merge(pd.merge(ratings, users), movies)

data[:10]


# In[26]:


mean_ratings= data.pivot_table('rating', 'title', 'gender', aggfunc='mean')
mean_ratings[:10]


# In[27]:


ratings_by_title= data.groupby('title').size()
ratings_by_title[:10]


# In[28]:


active_titles= ratings_by_title.index[ratings_by_title>=250]
active_titles[:10]


# In[29]:


mean_ratings= mean_ratings.ix[active_titles]
mean_ratings


# In[30]:


top_female_ratings= mean_ratings.sort_index(by='F', ascending=False)
top_female_ratings[:10]


# In[31]:


mean_ratings['diff']= mean_ratings['M'] - mean_ratings['F']
sorted_by_diff= mean_ratings.sort_index(by='diff')
sorted_by_diff[:10]


# In[32]:


sorted_by_diff[::-1][:10]


# In[33]:


rating_std_by_title= data.groupby('title')['rating'].std()

rating_std_by_title= rating_std_by_title.ix[active_titles]


# In[35]:


#Child birth data


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


names1980= pd.read_csv(r".\pydata-book-2nd-edition\datasets\babynames\yob1880.txt", names=["name", "sex", "births"])


# In[40]:


names1980[:10]


# In[42]:


names1980.groupby("sex").births.sum()


# In[51]:


years= range(1980, 2011)

columns=['name', 'sex', 'births']
pieces=[]

for year in years:
    path= './pydata-book-2nd-edition/datasets/babynames/yob%d.txt'% year
    frame= pd.read_csv(path, names=columns)
    frame['year']=year
    pieces.append(frame)
    names= pd.concat(pieces, ignore_index=True)


# In[52]:


names


# In[59]:


tot_births= names.pivot_table('births',  'year', 'sex', aggfunc=sum)
tot_births


# In[61]:


tot_births.plot(title= 'Total births by Sex and Year')


# In[63]:


def add_prop(group):
    births= group.births.astype('float')
    
    group['prop']= births/births.sum()
    return group

names= names.groupby(['year', 'sex']).apply(add_prop)


# In[69]:


tot_births= names.pivot_table('prop',  'year', 'sex', aggfunc=sum)

tot_births.plot(title= 'Total births by Sex and Year')


# In[76]:


names.columns.values


# In[78]:


np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)


# In[80]:


def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]

grouped= names.groupby(['year', 'sex'])
top_1000= grouped.apply(get_top1000)
top_1000


# In[82]:


boys= top_1000[top_1000.sex=='M']
girls= top_1000[top_1000.sex=='F']


# In[85]:


tot_births= top_1000.pivot_table('births', 'year', 'name', aggfunc=sum)
tot_births


# In[88]:


subset= tot_births[['John', 'Mary', 'Harry', 'Marilyn']]

subset.plot(subplots=True, figsize=(10,10), grid=False, title='Number of births per year')


# In[90]:


table= top_1000.pivot_table('prop', 'year', 'sex', aggfunc=sum)

table.plot(title='Sum of table1000.prop by year and sex', yticks= np.linspace(0,1.2, 13), xticks=range(1980, 2020, 10))


# In[91]:


np.linspace(0,1.2, 13)


# In[92]:


df= boys[boys.year==2010]


# In[96]:


prop_cumsum=df.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum.tail(10)


# In[97]:


prop_cumsum.searchsorted(0.5)


# In[98]:


df= boys[boys.year==1990]
prop_cumsum=df.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum.tail(10)


# In[111]:


prop_cumsum.searchsorted(0.5)


# In[112]:


def get_quantile_count(group, q=0.5):
    group= group.sort_index(by= 'prop', ascending=False)
    return group.prop.cumsum().searchsorted(q)+1


# In[124]:


diversity= top_1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity= diversity.unstack('sex')
diversity.head()


# In[125]:


#

diversity['M'] = diversity['M'].str[0]
diversity['F'] = diversity['F'].str[0]


# In[127]:


diversity.plot(title="Number of popular names in top 50%")


# In[130]:


get_last_letter= lambda x:x[-1]

last_letters= names.name.map(get_last_letter)
last_letters.name= 'last_letter'


# In[138]:


table= names.pivot_table('births', last_letters, ["sex", "year"], aggfunc=sum)
table


# In[152]:


subtable=table.reindex(columns=[1990,2000,2008,2010], level='year')
subtable.head()


# In[153]:


subtable.sum()


# In[156]:


letter_prop= subtable/subtable.sum().astype(float)
letter_prop.head()


# In[158]:


fig, axes= plt.subplots(2,1, figsize=(8,8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female')


# In[160]:


letter_prop= table/table.sum().astype(float)


# In[163]:


dny_ts= letter_prop.loc[["d", "n", "y"], "M"].T
dny_ts.head()


# In[164]:


dny_ts.plot()


# In[165]:


all_names= top_1000.name.unique()

mask= np.array(["lesl" in x.lower() for x in all_names])

lesley_like= all_names[mask]

lesley_like


# In[167]:


filtered= top_1000[top_1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()


# In[169]:


table= filtered.pivot_table("births", "year", "sex", aggfunc=sum)
table


# In[173]:


table.plot(style={'M':'k-', 'F':'k--'})


# In[175]:





# In[ ]:




