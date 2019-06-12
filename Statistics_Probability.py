#!/usr/bin/env python
# coding: utf-8

# In[72]:


#Statistics

from collections import Counter
num_friends = [100, 49, 41, 40, 25]
mins= [10, 40, 55, 70, 100]

friends_count= Counter(num_friends)
friends_count


# In[13]:


xs= range(101)

ys= [friends_count[i] for i in xs]


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.bar(xs, ys)
plt.show()


# In[14]:


num_points= len(num_friends)
num_points


# In[17]:


largest_value= max(num_friends)
print(largest_value)
smallest_value= min(num_friends)
print(smallest_value)


# In[21]:


sorted_vals= sorted(num_friends)
sorted_vals


# In[23]:


sorted_vals[0]


# In[25]:


sorted_vals[-2]


# In[27]:


def mean(x):
    return sum(x)/len(x)


mean(num_friends)


# In[41]:


def median(vec):
    n= len(vec)
    sorted_vec= sorted(vec)
    mid_point= n//2
    
    if n % 2==1:
        med= sorted_vec[mid_point]
        return med
    else:
        lo= mid_point-1
        hi= mid_point
        med= (sorted_vec[lo]+sorted_vec[hi])/2
        return med

x=[1,2,5,6,7,8]
median(x)


# In[42]:


median(num_friends)


# In[44]:


def quantile(x, p):
    p_index= int(p*len(x))
    return sorted(x)[p_index]

quantile(num_friends, .10)


# In[45]:


quantile(num_friends, .30)


# In[46]:


quantile(num_friends, .70)


# In[55]:


#Dispersion

def data_range(x):
    return max(x)-min(x)


# In[56]:


data_range(num_friends)


# In[66]:


def dot(a,b):
    return sum(i*j for i,j in zip(a,b))

def sum_of_squares(a):
    return dot(a,a)

def de_mean(x):
    x_bar= mean(x)
    return [x_i -x_bar for x_i in x]

def variance(x):
    n=len(x)
    deviations= de_mean(x)
    return sum_of_squares(deviations)/n-1


# In[67]:


variance(x)


# In[69]:


import math

def std_dev(x):
    return math.sqrt(variance(x))

std_dev(x)


# In[70]:


def int_quar_rng(x):
    return quantile(x, 0.75)- quantile(x, 0.25)

int_quar_rng(x)


# In[71]:


def covariance(x, y):
    l= len(x)
    return dot(de_mean(x), de_mean(y))/l-1


# In[74]:


covariance(num_friends, mins)


# In[85]:


def correlation(x,y):
    std_dev_x= std_dev(x)
    std_dev_y= std_dev(y)
    if std_dev_x > 0 and std_dev_y > 0:
        return covariance(x, y)/ std_dev_x/std_dev_y
    else:
        return 0


# In[86]:


correlation(num_friends, mins)


# In[3]:


# Probability
import random

def random_kid():
    return random.choice(["boy", "girl"])

both_girls=0
older_girl=0
either_girl=0

random.seed(0)

for _ in range(1000):
    younger= random_kid()
    older= random_kid()
    if older=="girl":
        older_girl+=1
    if older=="girl" and younger=="girl":
        both_girls+=1
    if older=="girl" or younger=="girl":
        either_girl+=1
        
print("P(both|older):", both_girls/older_girl)
print("P(both|either):", both_girls/either_girl)


# In[13]:


import math


def norm_dist(x, mu, sigma):
    sq_rt_two_pi_sigma= math.sqrt(2*math.pi)*sigma
    return math.exp(-(x-mu)**2/(2*(sigma**2)))/sq_rt_two_pi_sigma


# In[22]:


xs= [x/10.0 for x in range(-50, 50)]

import matplotlib.pyplot as plt

plt.plot(xs, [norm_dist(x, 0, 1) for x in xs], '-', label="mu=0, sigma=1")
plt.plot(xs, [norm_dist(x, 0, 2) for x in xs], '--', label="mu=0, sigma=2")
plt.plot(xs, [norm_dist(x, 1, 2) for x in xs], '.', label="mu=1, sigma=1")
plt.plot(xs, [norm_dist(x, -1, 2) for x in xs], '.', label="mu=1, sigma=1")
plt.plot(xs, [norm_dist(x, -1, 2) for x in xs], ':', label="mu=1, sigma=1")
plt.plot(xs, [norm_dist(x, 0, 0.5) for x in xs], '-.', label="mu=0, sigma=.5")
plt.legend()
plt.show()


# In[25]:


def norm_cum_dist(x, mu, sigma):
    return (1+math.erf((x-mu)/math.sqrt(2)))/2


# In[26]:


plt.plot(xs, [norm_cum_dist(x, 0, 1) for x in xs], '-', label="mu=0, sigma=1")
plt.plot(xs, [norm_cum_dist(x, 0, 2) for x in xs], '--', label="mu=0, sigma=2")
plt.plot(xs, [norm_cum_dist(x, 1, 2) for x in xs], '.', label="mu=1, sigma=1")
plt.plot(xs, [norm_cum_dist(x, -1, 2) for x in xs], '.', label="mu=1, sigma=1")
plt.plot(xs, [norm_cum_dist(x, -1, 2) for x in xs], ':', label="mu=1, sigma=1")
plt.plot(xs, [norm_cum_dist(x, 0, 0.5) for x in xs], '-.', label="mu=0, sigma=.5")
plt.legend()
plt.show()


# In[38]:


def bernoulli_trial(p):
    return 1 if random.random()<p else 0

def binomial(n,p):
    return sum(bernoulli_trial(p) for _ in range(n))


# In[51]:


from collections import Counter

def make_hist(p, n, num_points):
    data= [binomial(n,p) for _ in range(num_points)]
    histogram= Counter(data)
    plt.bar([x-0.4 for x in histogram.keys()], [v/num_points for v in histogram.values()], 0.8, color='red')
    
    mu= p*n
    sigma= math.sqrt(n*p*(1-p))
    xs= range(min(data), max(data)+1)
    ys= [norm_cum_dist(i+0.5, mu,sigma) - norm_cum_dist(i-0.5, mu,sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Dist vs Norm Approx")
    plt.show()


# In[52]:


make_hist(0.75, 100, 10000),


# In[41]:





# In[ ]:




