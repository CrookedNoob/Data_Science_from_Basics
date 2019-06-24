#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# In[2]:


obj= Series([2,4,-9, 1])
obj


# In[3]:


obj.values


# In[4]:


obj.index


# In[5]:


obj= Series([1,2,3,4], index=['x', 'y', 'z', 'ez'])
obj


# In[6]:


obj.index


# In[7]:


obj['ez']


# In[8]:


obj[['ez', 'x','z']]


# In[9]:


np.exp(obj[obj>2]*2)


# In[10]:


'a' in obj


# In[11]:


'z' in obj


# In[12]:


4 in obj.values


# In[13]:


sdata= {'Ohio': 3000, 'Texas': 2400, 'Utah': 4000, 'Oregon': 1590}
sdata


# In[14]:


obj2= Series(sdata)
obj2


# In[15]:


states= ['Ohio', 'Utah', 'Oregon', 'Texas', "Cal"]
obj3= Series(sdata, states)


# In[16]:


obj3


# In[17]:


pd.isnull(obj3)


# In[18]:


pd.notnull(obj3)


# In[19]:


obj2+obj3


# In[20]:


obj3.name='Population'


# In[21]:


obj3.index.name='State'


# In[22]:


obj3


# In[23]:


data= {'state':['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year':[2000, 2001, 2002, 2001, 2002], 
      'population':[1.5, 1.6, 1.7, 2.4, 2.9]}
data


# In[24]:


frame= DataFrame(data)
frame


# In[25]:


DataFrame(data, columns=['year', 'state', 'population'])


# In[26]:


frame2= DataFrame(data, index=['one', 'two', 'three', 'four', 'five'])
frame2


# In[27]:


frame2.columns


# In[28]:


frame2= DataFrame(data, columns=['year', 'state', 'population', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
frame2


# In[29]:


frame2.state


# In[30]:


frame2.year


# In[31]:


frame2.loc["three"]


# In[32]:


frame2.debt= np.arange(5)+1
frame2


# In[33]:


val= Series([-1.2, -1.5, -1.7], index=['two', 'three', 'five'])

frame2.debt= val


# In[34]:


frame2


# In[35]:


frame2['eastern']= frame2.state=='Ohio'
frame2


# In[36]:


del frame2["eastern"]
frame.columns


# In[37]:


pop= {'Nevada':{2001: 2.4, 2002:2.9}, 'Ohio':{2000:1.5, 2001:1.7, 2002:3.6}}


# In[38]:


frame3= DataFrame(pop)
frame3


# In[39]:


frame3.T


# In[40]:


DataFrame(frame3, index=[2002, 2001, 2000])


# In[41]:


pdata={'Ohio': frame3['Ohio'][:-1], 'Nevada':frame3['Nevada'][2:]}
DataFrame(pdata)


# In[42]:


frame3.index.name='year'; frame3.columns.name='state'

frame3


# In[43]:


frame3.values


# In[44]:


frame2.values


# In[45]:


obj= Series(range(3), index=['a', 'b', 'c'])
obj


# In[46]:


index=obj.index
index


# In[47]:


index[1:]


# In[48]:


index[1]='d'


# In[49]:


index= pd.Index(np.arange(3))
index


# In[50]:


obj2= Series([1,1.5,2.5], index=index)
obj2


# In[51]:


'Ohio' in frame3.columns


# In[52]:


2019 in frame3.index


# In[53]:


obj= Series(range(5), index=['d', 'b', 'a', 'c', 'e'])
obj


# In[54]:


obj2=obj.reindex(['a', 'b', 'c', 'd', 'e', 'f'])
obj2


# In[55]:


obj3=Series(["Blue", "Red", "Green"], index=[0, 2,4])
obj3


# In[56]:


obj3.reindex(range(6), method='ffill')


# In[57]:


frame= DataFrame(np.arange(9).reshape(3,3), columns=["Cal", "Ohio", "Texas"], index=["a", "c", "d"])
frame


# In[58]:


frame2= frame.reindex(["a", "b", "c", "d"])
frame2


# In[59]:


states=["Cal", "Texas", "Utah"]

frame.reindex(columns=states)


# In[60]:


frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=states)


# In[61]:


obj= Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj= obj.drop('c')
new_obj


# In[62]:


data= DataFrame(np.arange(16).reshape(4,4), index=["Ohio", "Cal", "Texas", "Nevada"], columns=['a', 'b', 'c', 'd'] )
data


# In[63]:


data.drop(["Cal", "Nevada"])


# In[64]:


data.drop(['a','b'], axis=1)


# In[65]:


data.drop('a', axis=1)


# In[66]:


obj= Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj


# In[67]:


obj['b']


# In[68]:


obj[1]


# In[69]:


obj[1:3]


# In[70]:


obj[["a", "b", "d"]]


# In[71]:


obj[[1,3,2]]


# In[72]:


obj["b":"d"]=5
obj


# In[73]:


data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

data


# In[74]:


data['one']


# In[75]:


data[['three', 'one']]


# In[76]:


data[:1]


# In[77]:


data.drop(['two', 'four'], axis=1)


# In[78]:


obj= Series(np.arange(4.), index=['a', 'b', 'c', 'd'])


# In[79]:


obj


# In[80]:


obj[1]


# In[81]:


obj['b']


# In[82]:


obj[2:4]


# In[85]:


obj[['b', 'a', 'd']]


# In[86]:


obj[[1,3,2]]


# In[87]:


obj[obj<2]


# In[88]:


obj['a':'c']


# In[90]:


data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], 
                 columns=['one', 'two', 'three', 'four'])
data


# In[97]:


data[:2]


# In[99]:


data[data>=3]


# In[100]:


data<5


# In[102]:


data.loc['Colorado', ['two', 'three']]


# In[108]:


data.ix[['Colorado', 'Utah'], [3,0,1]]


# In[110]:


data.iloc[2]


# In[113]:


data.loc[:'Utah', 'two']


# In[118]:


data.ix[data.three>5, :3]


# In[119]:


s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e']) 
s1


# In[121]:


s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s2


# In[122]:


s1+s2


# In[123]:


df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),index=['Ohio', 'Texas', 'Colorado'])
df1


# In[124]:


df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df2


# In[125]:


df1+df2


# In[126]:


df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df1


# In[127]:


df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df2


# In[128]:


df1+df2


# In[130]:


df1.add(df2, fill_value=0)


# In[131]:


df1.reindex(columns=df2.columns, fill_value=0)


# In[132]:


arr = np.arange(12.).reshape((3, 4))
arr


# In[133]:


arr[0]


# In[134]:


arr-arr[0]


# In[137]:


arr[0:2]


# In[150]:


frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[141]:


series= frame.iloc[0]
series


# In[142]:


frame-series


# In[146]:


series2= Series(range(3), index=['b','f','g'])
frame-series2


# In[147]:


frame+series2


# In[152]:


series3=frame['d']
series3


# In[154]:


frame.sub(series3, axis=0)


# In[155]:


frame = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[156]:


np.abs(frame)


# In[159]:


f= lambda x: x.max()-x.min()
frame.apply(f)


# In[160]:


frame.apply(f, axis=1)


# In[161]:


def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])

frame.apply(f)


# In[162]:


frame.apply(f, axis=1)


# In[163]:


format= lambda x: '%.2f'%x

frame.applymap(format)


# In[164]:


frame['e'].map(format)


# In[169]:


obj= Series(range(4), index=['b', 'd', 'a', 'c'])
obj


# In[170]:


obj.sort_index()


# In[174]:


frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame


# In[175]:


frame.sort_index()


# In[176]:


frame.sort_index(axis=1)


# In[177]:


frame.sort_index(axis=1, ascending=False)


# In[181]:


obj= Series([4,6,1,0, -4])
obj.sort_values(ascending=False)


# In[183]:


obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj


# In[184]:


obj.sort_values()


# In[190]:


frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame


# In[191]:


frame.sort_values(by='b')


# In[192]:


frame.sort_values(by=['a','b'])


# In[193]:


obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()


# In[194]:


obj.rank(method='first')


# In[198]:


obj.rank(ascending=False, method='max')


# In[199]:


frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
frame


# In[201]:


frame.rank(axis=1)


# In[202]:


obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj


# In[204]:


obj.index.is_unique


# In[205]:


obj['a']


# In[206]:


df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df


# In[207]:


df.loc['b']


# In[208]:


df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
df


# In[209]:


df.sum()


# In[220]:


df.sum(axis=1)


# In[221]:


df.mean(axis=1, skipna=False)


# In[223]:


df.idxmin()


# In[224]:


df.idxmax()


# In[226]:


df.cumsum()


# In[227]:


df.describe()


# In[229]:


obj = Series(['a', 'a', 'b', 'c'] * 4)
obj


# In[230]:


obj.describe()


# In[231]:


df.skew()


# In[233]:


from pandas_datareader import data as web

all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')


# In[237]:


price= DataFrame({tic: data['Adj Close'] for tic, data in all_data.items()})
volume= DataFrame({tic: data['Volume'] for tic, data in all_data.items()})


# In[239]:


price.head()


# In[241]:


returns= price.pct_change()
returns.tail()


# In[242]:


returns.MSFT.corr(returns.IBM)


# In[243]:


returns.GOOG.corr(returns.AAPL)


# In[244]:


returns.MSFT.cov(returns.IBM)


# In[245]:


returns.GOOG.cov(returns.AAPL)


# In[246]:


returns.corr()


# In[247]:


returns.cov()


# In[248]:


returns.corrwith(returns.GOOG)


# In[249]:


returns.corrwith(volume)


# In[251]:


obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj


# In[252]:


uniques= obj.unique()
uniques


# In[253]:


obj.value_counts()


# In[254]:


mask= obj.isin(['b', 'c'])
mask


# In[255]:


obj[mask]


# In[256]:


data= DataFrame(DataFrame({'Qu1': [1, 3, 4, 3, 4],
                           'Qu2': [2, 3, 1, 2, 3],
                           'Qu3': [1, 5, 2, 4, 4]}))

data


# In[258]:


result= data.apply(pd.value_counts).fillna(0)
result


# In[265]:





# In[266]:


string_data= Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data


# In[267]:


string_data[0]=None
string_data.isnull()


# In[269]:


from numpy import nan as NA

data = Series([1, NA, 3.5, NA, 7])
data


# In[270]:


data.dropna()


# In[271]:


data[data.notnull()]


# In[272]:


data = DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
cleaned= data.dropna()
cleaned


# In[273]:


data.dropna(how='all')


# In[275]:


data[4]=NA
data


# In[276]:


data.dropna(axis=1, how='all')


# In[277]:


df = DataFrame(np.random.randn(7, 3))
df


# In[278]:


df.loc[:4,1]=NA
df


# In[279]:


df.loc[:2,2]=NA
df


# In[280]:


df.dropna(thresh=3)


# In[282]:


df.fillna(0)


# In[285]:


df.fillna({1:0.5, 2:-1.5})


# In[284]:


df


# In[286]:


_= df.fillna(0, inplace=True)
df


# In[287]:


df = DataFrame(np.random.randn(6, 3))
df


# In[289]:


df.loc[2:, 1] = NA; df.loc[4:, 2] = NA
df


# In[290]:


df.fillna(method='ffill')


# In[291]:


df.fillna(method='ffill', limit=2)


# In[292]:


df.fillna(data.mean())


# In[294]:


data = Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], 
                                          [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data


# In[295]:


data.index


# In[296]:


data['b']


# In[297]:


data['b':'c']


# In[304]:


data.loc[['b','d']]


# In[305]:


data[:,2]


# In[310]:


data['c']


# In[311]:


data.unstack()


# In[312]:


data.unstack().stack()


# In[313]:


frame = DataFrame(np.arange(12).reshape((4, 3)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], 
                  columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
frame


# In[315]:


frame.index.names=['key1', 'key2']
frame.columns.names=['state', 'color']
frame


# In[316]:


frame['Ohio']


# In[320]:


frame.swaplevel('key1', 'key2')


# In[322]:


frame.sort_index(level=1)


# In[326]:


frame.swaplevel(0,1).sort_index(1)


# In[327]:


frame.sum(level='key2')


# In[328]:


frame.sum(level='color', axis=1)


# In[329]:


frame = DataFrame({'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'], 
                   'd': [0, 1, 2, 0, 1, 2, 3]})
frame


# In[330]:


frame2= frame.set_index(['c', 'd'])
frame2


# In[331]:


frame.set_index(['c', 'd'], drop=False)


# In[332]:


frame2.reset_index()


# In[334]:


ser = Series(np.arange(3.))
ser


# In[336]:


ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]


# In[337]:


ser.loc[:1]


# In[338]:


ser3 = Series(range(3), index=[-5, 1, 3])
ser3


# In[346]:


from pandas_datareader import data as web

pdata = pd.Panel(dict((stk, web.get_data_yahoo(stk)) 
                      for stk in ['AAPL', 'GOOG', 'MSFT', 'DELL']))
pdata


# In[347]:


pdata= pdata.swapaxes('items', 'minor')


# In[349]:


pdata['Adj Close']


# In[350]:


pdata.loc[:, '6/1/2012', :]


# In[351]:


pdata.loc['Adj Close', '5/22/2012':, :]


# In[352]:


stacked = pdata.ix[:, '5/30/2012':, :].to_frame()
stacked.head()


# In[354]:


stacked.to_panel()


# In[ ]:




