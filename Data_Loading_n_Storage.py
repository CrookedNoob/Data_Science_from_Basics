#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np


# In[2]:


pd.read_csv("./pydata-book-2nd-edition/datasets/fec/P00000001-ALL.csv")


# In[3]:


pd.read_csv("./pydata-book-2nd-edition/datasets/fec/P00000001-ALL.csv", header=None)


# In[5]:


pd.read_table("./pydata-book-2nd-edition/datasets/fec/P00000001-ALL.csv", sep=',')


# In[6]:


pd.read_csv("./pydata-book-2nd-edition/datasets/fec/P00000001-ALL.csv", 
            names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p'])


# In[23]:


parsed= pd.read_csv("./pydata-book-2nd-edition/datasets/fec/P00000001-ALL.csv", index_col=['cmte_id', 'cand_id'])
parsed.head()


# In[26]:


parsed= pd.read_csv("./pydata-book-2nd-edition/datasets/fec/P00000001-ALL.csv", skiprows=[1,2,3])
parsed.head()


# In[28]:


result= pd.read_csv("./pydata-book-2nd-edition/examples/ex6.csv")
result


# In[29]:


result= pd.read_csv("./pydata-book-2nd-edition/examples/ex6.csv", nrows=5)
result


# In[44]:


result.to_csv('out.csv')


# In[48]:


dates= pd.date_range('1/9/1990', periods=7)
dates


# In[49]:


ts= Series(np.arange(7), index=dates)
ts


# In[50]:


import csv

f= open("./pydata-book-2nd-edition/examples/ex7.csv")

reader= csv.reader(f)
reader


# In[51]:


for line in reader:
    print(line)


# In[53]:


lines= list(csv.reader(open("./pydata-book-2nd-edition/examples/ex7.csv")))
lines


# In[54]:


header, values= lines[0], lines[1:]
header


# In[55]:


values


# In[56]:


data_dict= {h:v for h, v in zip(header, zip(*values))}
data_dict


# In[59]:


obj = """
{"name": "Wes",
"places_lived": ["United States", "Spain", "Germany"],
"pet": null,
"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
{"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

obj


# In[60]:


import json


# In[61]:


result= json.loads(obj)
result


# In[63]:


as_json= json.dumps(result)
as_json


# In[64]:


siblings= DataFrame(result['siblings'], columns=['name', 'age'])
siblings


# In[66]:


from lxml.html import parse
from urllib.request import urlopen


# In[67]:


parsed= parse(urlopen('https://finance.yahoo.com/quote/AAPL/options?ltr=1'))
parsed


# In[69]:


doc= parsed.getroot()
doc


# In[71]:


links= doc.findall('.//a')


# In[72]:


links[5:10]


# In[73]:


links[30].get('href')


# In[74]:


links[30].text_content()


# In[76]:


urls= [link.get('href') for link in doc.findall('.//a')]
urls


# In[88]:


tables= doc.findall('.//table')
calls=tables[0]
puts= tables[1]


# In[89]:


rows = calls.findall('.//tr')


# In[90]:


def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]


# In[91]:


_unpack(rows[0], kind='th')


# In[92]:


_unpack(rows[1], kind='td')


# In[95]:


from pandas.io.parsers import TextParser

def parse_options_data(table):
    rows= table.findall('.//tr')
    header= _unpack(rows[0], kind='th')
    data= [_unpack(r) for r in rows[1:]]
    return TextParser(data, names=header).get_chunk()


# In[97]:


call_data= parse_options_data(calls)
call_data[:10]


# In[98]:


put_data= parse_options_data(puts)
put_data[:10]


# In[99]:


from lxml import objectify

path= './pydata-book-2nd-edition/datasets/mta_perf/Performance_MNR.xml'

parsed= objectify.parse(open(path))
parsed


# In[100]:


root= parsed.getroot()
root


# In[103]:


data=[]
skip_fields=['PARENT_SEQ', 'INDICATOR_SEQ', 'DESIRED_CHANGE', 'DECIMAL_PLACES']

for elt in root.INDICATOR:
    el_data={}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag]= child.pyval
    data.append(el_data)


# In[104]:


perf= DataFrame(data)
perf[:10]


# In[106]:


from io import StringIO

tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()

root


# In[107]:


root.get('href')


# In[109]:


root.text


# In[110]:


frame= pd.read_csv("./pydata-book-2nd-edition/examples/ex1.csv")
frame


# In[113]:


import requests


# In[114]:


url = 'http://search.twitter.com/search.json?q=python%20pandas'


# In[115]:


resp= requests.get(url)


# In[116]:


resp


# In[118]:


import json

data= json.loads(resp.text)
data.keys()


# In[ ]:




