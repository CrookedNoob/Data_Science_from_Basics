#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Algebra


# In[1]:


x=[20, 45, 67]
y=[12, 10, 78]

def vec_add(a,b):
    return [i+j for i, j in zip(a,b)]


print(vec_add(x,y))

def vec_sub(a, b):
    return [i-j for i, j in zip(a,b)]

print(vec_sub(x,y))


# In[2]:


def scal_mul(c, v):
    return [c*i for i in v]

scal_mul(2,x)


# In[3]:


def avg(v):
    return sum(v)/len(v)

avg(x)


# In[4]:


def dot(a,b):
    return sum(i*j for i, j in zip(a,b))

dot(x,y)


# In[6]:


def sum_of_squares(a):
    return dot(a,a)

sum_of_squares(x)


# In[7]:


import math

def mag(a):
    return math.sqrt(sum_of_squares(a))

mag(x)


# In[9]:


def sq_dist(a,b):
    return sum_of_squares(vec_sub(a,b))

sq_dist(x,y)


# In[10]:


def dist(a,b):
    return mag(vec_sub(a,b))

dist(x,y)


# In[11]:


# Matrix

A= [[1,2,3, 4],
   [5,6,7,8]]

B= [[1,2],
   [3,4],
   [5,6],
   [7,8]]


# In[16]:


def mat_shape(m):
    num_rows= len(m)
    num_cols= len(m[0]) if m else 0
    return num_rows, num_cols

mat_shape(B)


# In[22]:


def get_row(m, i):
    return m[i]

get_row(A, 1)


# In[25]:


def get_col(m,i):
    return [m_i[i] for m_i in m]

get_col(A, 2)


# In[29]:


def make_mat(nr,nc, fn):
    return [[fn(i,j) for j in range(nc)] for i in range(nr)]

def diag(i,j):
    return 1 if i==j else 0

make_mat(5,5,diag)


# In[31]:


friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # user 0
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # user 1
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # user 2
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # user 3
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # user 4
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # user 5
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 6
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 7
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # user 8
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # user 9

[ i for i, is_f in enumerate(friendships[4]) if is_f]


# In[40]:





# In[ ]:




