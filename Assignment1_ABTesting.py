#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats as st
import math
import random


# In[2]:


df = pd.read_csv("AB_test_data.csv")

# inspect data frame
print(df.head())


# ## Question 1

# In[3]:


# for variant A - what % purchased
p_pop = len(df[(df['Variant'] == "A") & (df['purchase_TF'] == True)]) / len(df[df['Variant'] == "A"])
print("Conversion_A: %.3f" % p_pop)


# In[4]:


# for variant B - what % purchase
p_sam = len(df[(df['Variant'] == "B") & (df['purchase_TF'] == True)]) / len(df[df['Variant'] == "B"])
print("Conversion_B: %.3f" % p_sam)


# In[5]:


# sample size - how many people got variant B
n = len(df[df['Variant'] == "B"])
print("n: %d" % n)


# In[6]:


z = (p_sam - p_pop) / math.sqrt( (p_pop*(1-p_pop)) / n )
print("z-score: %.2f" % z)


# In[7]:


alpha = 0.05

z_alpha = st.norm.ppf(1-(alpha/2))

print("z_alpha2: %.2f" % z_alpha)


# In[9]:


print("Reject Null Hypothesis: %s" % (z > z_alpha))


# ## Question 2

# In[10]:


beta = 1-0.8
z_beta = st.norm.ppf(1-beta)
print("z_beta: %.2f" % z_beta)


# In[15]:


delta = (p_sam - p_pop) #/ p_pop

print("Delta: %.3f" % delta)


# In[16]:


p_hat = (p_sam+p_pop)/2

print("p-hat: %.2f" % p_hat)


# In[17]:


n_opt = ((z_alpha*math.sqrt(2*p_hat*(1-p_hat)) + z_beta * math.sqrt(p_pop*(1-p_pop) + p_sam *(1-p_sam)))**2 ) / delta**2

print("Optimal sample size: %d" % math.ceil(n_opt))


# In[19]:


# create 10 samples according to size n_opt
samples = []

x=0
while x < 10:
    samples.append(random.sample(df[df['Variant'] == "B"]['id'].tolist(),math.ceil(n_opt)))
    x += 1


# In[20]:


# test the samples
list_p_sample = []
list_z = []
list_reject_null = []

for i in samples:
    # subset data to just the samples
    df_sample = df[df['id'].isin(i)]
    
    # calculate p sample
    p_sample = len(df_sample[df_sample['purchase_TF'] == True]) / len(df_sample)
    
    # calculate z
    z = (p_sample - p_pop) / math.sqrt( (p_pop*(1-p_pop)) / math.ceil(n_opt) )
    
    # do i reject the null hypothesis
    reject_null = abs(z) > z_alpha
    
    list_p_sample.append(p_sample)
    list_z.append(z)
    list_reject_null.append(reject_null)
    
df_sampleresults = pd.DataFrame({'Sample_Number': range(0,len(samples)), 
                                 'p_sample': list_p_sample,
                                'z_score': list_z,
                                'reject_null': list_reject_null})

print(df_sampleresults)


# ## Question 3

# In[21]:


# establish boundaries
upper_bound = math.log(1/alpha)
lower_bound = math.log(beta)

print("Upper Bound: %.2f" % upper_bound)
print("Lower Bound: %.2f" % lower_bound)


# In[23]:


stopping_iteration = []
stopping_reason = []

for sample_n in samples:
    results = df[df['id'].isin(sample_n)]['purchase_TF'].values

    log_gamma = 0
    count = 0
    while (log_gamma > lower_bound) & (log_gamma < upper_bound):
        if results[count] == True:
            log_gamma = log_gamma + math.log(p_sam / p_pop)
        else:
            log_gamma = log_gamma + math.log( (1-p_sam) / (1-p_pop))

        count += 1

    stopping_iteration.append(count)
    
    if log_gamma < lower_bound:
        stopping_reason.append('Lower bound')
    else:
        stopping_reason.append('Upper bound')
    
print(stopping_iteration)
print(stopping_reason)
print("Average iterations: %.2f" % (sum(stopping_iteration) / len(stopping_iteration)))

