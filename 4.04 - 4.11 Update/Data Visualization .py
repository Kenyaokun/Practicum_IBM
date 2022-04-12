#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import statsmodels.api as sm 
import statsmodels.formula.api as smf
import os
import math
import sklearn
import matplotlib.pyplot as plt
import warnings
import seaborn as sns


# In[3]:


path = '/Users/luhan/Desktop/'
df = pd.read_csv(path+'lung_cancer_all_dummified.csv', sep = '\,', engine = 'python')
df.head(10)


# In[5]:


plt.hist(df['Overall Survival (Months)'])
plt.xlabel("Overall Survival (Months)")
plt.ylabel("Age")


# In[7]:


import seaborn as sns
sns.regplot(x="Overall Survival (Months)", y="Age", data=df, color='r')


# In[10]:


sns.scatterplot(df.index,df['Overall Survival (Months)'],alpha=0.8,hue=df['Age'],palette='rocket')


# #Takeways: First, I put Overall Survival (Months) and Age in the histgram to compare, we can see at heatmap, the age between 90-70s have less month of Overall Survival, it fits my guess,beacuse when incresing 1% of age, the Overall Survival (Months)will decrease.

# In[8]:


plt.hist(df['Overall Survival (Months)'])
plt.xlabel("Overall Survival (Months)")
plt.ylabel("Smoking Status")


# In[9]:


import seaborn as sns
sns.regplot(x="Overall Survival (Months)", y="Smoking Status", data=df, color='r')


# In[54]:


sns.scatterplot(df.index,df['Overall Survival (Months)'],alpha=0.8,hue=df['Smoking Status'],palette='rocket')


# #Takeways: Second, I put Overall Survival (Months) and Smoking Status in the histgram to compare, we can see at heatmap, the smoking Status=1, means people who already have cancer have less month of Overall Survival.

# In[12]:


plt.hist(df['Overall Survival (Months)'])
plt.xlabel("Overall Survival (Months)")
plt.ylabel("Cancer Type Detailed")


# In[13]:


import seaborn as sns
sns.regplot(x="Overall Survival (Months)", y="Cancer Type Detailed", data=df, color='r')


# In[14]:


sns.scatterplot(df.index,df['Overall Survival (Months)'],alpha=0.8,hue=df['Cancer Type Detailed'],palette='rocket')


# #Takeways: Second, I put Overall Survival (Months) and Cancer Type Detailed in the histgram to compare, we can see at heatmap, the smoking Status=1, means  Small Cell Lung Cancer has lesss month of Overall Survival. It fits my guess.

# In[25]:


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Set up function parameters for different cross validation strategies
kfold = KFold(n_splits=5, shuffle=True)
skfold = StratifiedKFold(n_splits=5, shuffle=True) 


# In[26]:


y = df['Overall Survival (Months)']
X = pd.DataFrame(df, columns = ['Age',
                                'Sex',
                                'Cancer Type Detailed'])   


# In[27]:


# Use train_test_split(X,y) to create four new data sets, defaults to .75/.25 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

print(X.shape)
X_train.shape


# In[28]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[32]:


from sklearn import preprocessing
# Here, I standardize by X data using StandardScalar
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))


# In[36]:


print("Training set score: {:.5f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.5f}".format(lr.score(X_test, y_test)))

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5)
skfold = StratifiedKFold(n_splits=5, shuffle=True) 
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1) 

print(np.mean(cross_val_score(LinearRegression(), X_train, y_train, cv=kfold, scoring="r2")))


# In[37]:


X_train_new = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_new ).fit()

model.summary()


# In[38]:


#### Ridge Regressions with Cross-Validations (alpha=1, 10, 0.1 respectively)


# In[48]:


from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.5f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.5f}".format(ridge.score(X_test, y_test)))


# In[49]:


ridgecoef=ridge.coef_


# In[50]:


ridge_scores=np.mean(cross_val_score(ridge, X_train,y_train, scoring='r2',cv=kfold))
ridge_scores


# In[51]:


ridge10 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.5f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.5f}".format(ridge10.score(X_test, y_test)))


# In[52]:


ridge10_scores=np.mean(cross_val_score(ridge10, X_train,y_train, scoring='r2',cv=kfold))
ridge10_scores


# In[53]:


ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.5f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.5f}".format(ridge01.score(X_test, y_test)))


# In[ ]:


ridge01_scores=np.mean(cross_val_score(ridge01, X_train,y_train, scoring='r2',cv=kfold))
ridge01_scores

