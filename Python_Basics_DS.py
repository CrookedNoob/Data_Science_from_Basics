#!/usr/bin/env python
# coding: utf-8

# In[1]:


for i in [1,2,3,4,5]:
    print(i)
    for j in [1,2,3,4,5]:
        print(j)
        print(i+j)
    print(i)
print("looping finished")


# In[3]:


x= 2+3
x


# In[4]:


import re

re.compile("[0-9]", re.I)


# In[5]:


from collections import defaultdict, Counter

lookup= defaultdict(int)
my_counter= Counter


# In[10]:


y= lambda x:x*2
print(y(3))


# In[12]:


def minus(a=0, b=0):
    return a-b

print(minus(10, 5))
print(minus(a=5))
print(minus(b=10))


# In[13]:


#list

int_lst= [1,2,3,4]
het_lst= ["string", 1.0, False]
combo_lst= [het_lst, int_lst, []]

print(combo_lst)
print(len(het_lst))
print(int_lst)


# In[20]:


x= list(range(10))
zero=x[0]
nine= x[-1]
x[0]=-1
print(zero)
print(nine)
print(x)


# In[21]:


1 in [1,2,3]


# In[22]:


0 in [1,2,3]


# In[23]:


x[0:3]


# In[24]:


x[:4]


# In[25]:


x[2:]


# In[26]:


x[1:-1]


# In[27]:


y=x
y


# In[28]:


y=x[:]


# In[29]:


y


# In[42]:


x=[1,2,3]
x.extend([4,5,6])
x


# In[39]:


y= x+[7,8,9]
y


# In[44]:


print(x)
x.append(y)
print(x)


# In[45]:


len(x)


# In[52]:


x,y,z, _=[1,2,3,5]


# In[53]:


print(x)
print(y)
print(z)


# In[54]:


lst=[1,2,3]
tup=(4,5)
oth_tup= 3,4
lst[1]=5
lst


# In[55]:


try:
    tup[1]=3
except TypeError:
    print("cant modify tuple")


# In[59]:


def sum_prod(x,y):
    return (x+y), (x*y)

print(sum_prod(4,5))
print(sum_prod(9,10))


# In[61]:


a,b=3,4
a,b=b,a
print(a,b)


# In[62]:


#Diictionaries

empt_dict={}
empt_dict2=dict()
game= {"div2":1, "div":2, "dest": 100}


# In[68]:


div2_rank= game["div2"]
div2_rank


# In[69]:


try:
    tlou_rank= game["tlou"]
except KeyError:
    print("TLOU not ARPG")


# In[71]:


tlou_rank_exist= "tlou" in game
print(tlou_rank_exist)


# In[72]:


dest_rank_exist= "dest" in game
print(dest_rank_exist)


# In[74]:


game.get('div2',100)


# In[75]:


game.get('tlou',999)


# In[77]:


game.get('batman')


# In[78]:


game["witchr3"]=3
game["anthm"]=10
game


# In[79]:


len(game)


# In[80]:


twt= {"usr":"dip_soumyadip", "tweet":"division 2 is addictive", 
      "likes": 40, "retwts": 5, "hashtags":["#division2", "#ubisoft", "#ps4", "arpg"]}


# In[81]:


twt


# In[82]:


twt.keys()


# In[83]:


twt.values()


# In[84]:


twt.items()


# In[85]:


"usr" in twt.keys()


# In[86]:


"usr" in twt


# In[91]:


from collections import defaultdict

word_count= defaultdict(int)
for words in document:
    word_counts[word]=1


# In[94]:


dict_list= defaultdict(list)
dict_list["kaboom"].append([3,2,1])
dict_list


# In[96]:


dict_dict= defaultdict(dict)
dict_dict["kaboom"]["bazinga"]="bbt"
dict_dict


# In[98]:


pair= defaultdict(lambda: [0,0])
pair[2][1]=1
pair


# In[107]:


from collections import Counter
c= Counter([0, 1, 2, 3, 1])
c


# In[108]:


#Set

s= set()
s.add(0)
s.add(0)
s.add(2)
s.add(3)
s.add(1)
len(s)


# In[109]:


s


# In[110]:


2 in s


# In[111]:


lst=list(range(1000000))
len(lst)


# In[113]:


x=0
while x<10:
    print(x)
    x+=1


# In[114]:


for x in range(10):
    print(x)


# In[129]:


x = None
x is None


# In[ ]:


x = None


# In[4]:


#Sorting

x=[8,5,7,1,4,3,9]
y= sorted(x)
print(y)
print(x)
x.sort()
print(x)


# In[7]:


x= sorted([9,1,-4,5,2], reverse=True, key=abs)
x


# In[8]:


x= sorted([9,1,-4,5,2], reverse=True)
x


# In[9]:


[x for x in range(5) if x%2==0]


# In[10]:


[x*x for x in range(5)]


# In[13]:


{x: x*x for x in range(5) if x%2==0}


# In[14]:


{x*x for x in [-2, 2]}


# In[20]:


[(x,y) for x in range(10) for y in range(10)]


# In[23]:


[(x,y) for x in range(10) for y in range(x+2, 10)]


# In[25]:


# Random numbers

import random

{random.random() for _ in range(4)}


# In[26]:


random.seed(10)
print(random.random())


# In[27]:


random.seed(10)
print(random.random())


# In[ ]:




