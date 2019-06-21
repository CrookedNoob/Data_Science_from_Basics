#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Numpy

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data1=[2,5,7,6]

arr1= np.array(data1)
arr1


# In[4]:


data2=[[1, 2.5, 6.4, 7, 9],[21,4,1.9,4,5]]
arr2= np.array(data2)
arr2


# In[5]:


arr2.ndim


# In[6]:


arr2.shape


# In[7]:


arr1.shape


# In[8]:


arr2.dtype


# In[9]:


np.zeros(5)


# In[10]:


np.zeros((5,2))


# In[11]:


np.ones((1,2))


# In[12]:


np.empty((2,2))


# In[13]:


np.arange(10)


# In[14]:


np.arange(5,10)


# In[15]:


np.ones_like(arr2)


# In[16]:


x=np.eye(3)
x


# In[17]:


np.identity(3)


# In[18]:


arr1= np.array([1,2,3], dtype='float32')
arr2= np.array([1,2,3], dtype='int32')


# In[19]:


arr1.dtype


# In[20]:


arr2.dtype


# In[21]:


arr2.astype('float32')


# In[22]:


st= np.array(['1', '2', '3'], dtype=np.string_)

flt= st.astype(np.float64)


# In[23]:


flt.dtype


# In[24]:


flt= np.arange(5,10)
flt.astype(arr2.dtype)
flt.dtype


# In[25]:


arr= np.array([[1, 2.5, 6.4, 7, 9],[21,4,1.9,4,5]])
1/arr


# In[26]:


flt[2]


# In[27]:


flt[1:4]


# In[28]:


lala= flt[2:4]
lala[1:3]=34
lala


# In[29]:


lala[:]


# In[30]:


arr= np.arange(10)
arr


# In[31]:


arr_sl= arr[5:8]
arr_sl


# In[32]:


arr_sl[1]=123
arr_sl


# In[33]:


arr


# In[34]:


arr_sl[:]=64
arr_sl


# In[35]:


arr


# In[36]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# In[37]:


arr2d[2][2]


# In[38]:


arr2d[1,0]


# In[39]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d


# In[40]:


arr3d[0]


# In[41]:


arr3d[0,1,2]


# In[42]:


arr[1:6]


# In[43]:


arr2d[:2]


# In[44]:


arr2d[:2,1:]


# In[45]:


arr2d


# In[46]:


arr2d[1, :2]


# In[47]:


arr2d[:2, 1:] = 0


# In[48]:


arr2d


# In[49]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names


# In[50]:


data= np.random.randn(7,4)
data


# In[51]:


names=='Bob'


# In[52]:


data[names=='Bob']


# In[53]:


data[names=='Bob', 2:]


# In[54]:


data[names=='Bob', 3]


# In[55]:


data[names!='Bob']


# In[56]:


data[data<0]=0
data


# In[57]:


data[names!='Joe']=7
data


# In[58]:


arr= np.empty((8,4))
arr


# In[59]:


for i in range(8):
    arr[i]=i+1
    
arr


# In[60]:


arr[[4,3,1,2]]


# In[61]:


arr[[-4,-1,-8]]


# In[62]:


arr= np.arange(32).reshape(8,4)
arr


# In[63]:


arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# In[64]:


arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]


# In[65]:


arr[np.ix_([1,5,7,2],[0,3,1,2])]


# In[66]:


arr= np.arange(12).reshape(4,3)
arr


# In[67]:


arr.T


# In[68]:


np.dot(arr, arr.T)


# In[69]:


arr=arr.reshape(2,2,3)
arr


# In[70]:


arr.transpose((1,0,2))


# In[71]:


arr.swapaxes(1,1)


# In[72]:


x= np.random.randn(10)
y= np.random.randn(10)


# In[73]:


x


# In[74]:


y


# In[75]:


np.sqrt(x)


# In[76]:


np.maximum(x,y)


# In[77]:


arr= np.random.randn(5)*5
arr


# In[78]:


np.modf(arr)


# In[79]:


np.abs(arr)


# In[80]:


points= np.arange(-5,5, 0.01)


# In[81]:


xs, ys= np.meshgrid(points, points)


# In[ ]:





# In[82]:


z= np.sqrt(xs**2 +ys**2)
z


# In[83]:


plt.imshow(z)


# In[84]:


plt.imshow(xs)


# In[85]:


plt.imshow(ys)


# In[86]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[87]:


result= [(x if c else y) for x,y,c in zip(xarr,yarr, cond)]
result


# In[88]:


np.where(cond, xarr, yarr)


# In[89]:


arr=np.random.randn(4,4)
np.where(arr>0, 2, -2)


# In[90]:


np.where(arr>0, 2, arr)


# In[91]:


arr= np.random.randn(10)


# In[92]:


arr.mean()


# In[93]:


np.mean(arr)


# In[94]:


arr.sum()


# In[95]:


arr= np.random.randn(5,4)
arr


# In[96]:


arr.sum(axis=0)


# In[97]:


arr.sum(axis=1)


# In[98]:


arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr


# In[99]:


arr.cumsum(0)


# In[100]:


arr.cumsum(axis=1)


# In[101]:


arr.cumprod(axis=1)


# In[102]:


arr= np.random.randn(100)
(arr>0).sum()


# In[103]:


bools= np.array([False, False, True, False])
bools.any()


# In[104]:


bools.all()


# In[105]:


arr= np.random.randn(5)
arr


# In[106]:


arr.sort()


# In[107]:


arr


# In[108]:


arr= np.random.randn(5,3)
arr


# In[109]:


arr.sort(1)
arr


# In[110]:


arr.sort(0)
arr


# In[111]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])


# In[112]:


np.unique(names)


# In[113]:


ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# In[114]:


sorted(set(names))


# In[115]:


values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [1,3,6])


# In[116]:


arr= np.array(np.arange(10)).reshape(5,2)
arr


# In[117]:


np.in1d(arr, [1,3,6])


# In[118]:


np.save('some_array', arr)


# In[119]:


np.load('some_array.npy')


# In[120]:


np.savez('array_archive.npz', ints=ints, arr=arr)


# In[121]:


arch=np.load('array_archive.npz')


# In[122]:


arch['ints']


# In[123]:


arr= np.loadtxt('./pydata-book-2nd-edition/examples/array_ex.txt', delimiter=',')
arr


# In[124]:


x= np.array([[1.,2., 3.], [4., 5., 6.]])
y= np.array([[7.,8., 9.], [10., 11., 12.]])

x.dot(y.T)


# In[125]:


np.dot(x, np.ones(3))


# In[126]:


from numpy.linalg import inv, qr

X= np.random.randn(5,5)


# In[127]:


mat= X.T.dot(X)


# In[128]:


mat


# In[129]:


inv(mat)


# In[130]:


mat.dot(inv(mat))


# In[131]:


q,r= qr(mat)


# In[132]:


q


# In[133]:


r


# In[134]:


arr= np.random.normal(size=(4,4))
arr


# In[135]:


arr= np.random.normal(range(0,100))
plt.plot(arr)
arr


# In[136]:


import random

position=0
walk=[position]
steps=1000
for i in range(steps):
    step= 1 if random.randint(0,1) else -1
    position+=step
    walk.append(position)
    

plt.plot(walk)
    


# In[137]:


nsteps=1000
draws= np.random.randint(0,2, size=nsteps)
steps= np.where(draws>0, 1, -1)

walks= steps.cumsum()
plt.plot(walks)


# In[138]:


walks.min()


# In[139]:


walks.max()


# In[140]:


np.abs(walks>=7).argmax()


# In[141]:


nwalks=5000

nsteps=1000
draws= np.random.randint(0,2, size=(nwalks, nsteps))

steps= np.where(draws>0, 1,-1)

walks= steps.cumsum(1)


# In[142]:


walks.max()


# In[143]:


walks.min()


# In[146]:


hit30= np.abs(walks>=30).any(1)
hit30


# In[147]:


hit30.sum()


# In[ ]:




