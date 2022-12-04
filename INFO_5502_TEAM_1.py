#!/usr/bin/env python
# coding: utf-8

# ### <p style="background-color:white;font-family:timesnewroman;color:lightseagreen;font-size:200%;border-radius:20px 60px;">Introduction</p> 
#  
# #### Predicting future sales for a company is one of the most important aspects of strategic planning.
# 
# In this kernel, we will wanted analyze in depth how internal and external factors of one of the biggest companies in the US can affect their Weekly Sales in the future.
# 
# This module contains complete analysis of data , includes time series analysis , identifies the best performing stores , performs sales prediction with the help of multiple linear regression.
# 
# The data collected ranges from 2010 to 2012, where 45 Walmart stores across the country were included in this analysis. It is important to note that we also have external data available like CPI, Unemployment Rate and Fuel Prices in the region of each store which, hopefully, help us to make a more detailed analysis.

# ## <p style="background-color:white;font-family:newtimeroman;color:coral;font-size:170%;text-align:center;border-radius:20px 60px;">Let's dive in . . .</p>
# 

# 
#  #### In Retail Industry and chain of stores one of the biggest issue they face are supply chain management. The component of supply chain management (SCM) involved with determining how best to fulfill the requirements created from the Demand Plan. 
#  
#  *It's objective is to balance supply and demand in a manner that achieves the financial and service objectives of the enterprise.*
#  
#   If we look into the case of a retail chain stores one of the basic case is to know the demand of products that are sold in the store. If the decision making authority know whats the demand of each products for a week or month, they would be able to plan the supply chain accordingly. If that is possible this would save a lot of money for them because they don't have to overstock or can plan their Logistics accordingly.

#  
# ### <p style="background-color:white;font-family:calibri;color:lightseagreen;font-size:200%;border-radius:20px 60px;">Data</p> 
# 
# ### There are 3 Datasets :

# #### Stores:¶
# 
# - Store: The store number. Range from 1–45.
# - Type: Three types of stores ‘A’, ‘B’ or ‘C’.
# - Size: Sets the size of a Store would be calculated by the no. of products available in the particular store ranging from 34,000 to 210,000.
# 
# ***primary key is Store***
# #### Sales:¶
# 
# - Date: The date of the week where this observation was taken.
# - Weekly_Sales: The sales recorded during that Week.
# - store: The store which observation in recorded 1–45
# - Dept: One of 1–99 that shows the department.
# - IsHoliday: Boolean value representing a holiday week or not.
# 
# ***primary key is a combination of (Store,Dept,Date)***
# #### Features:
# 
# - store: The store which observation in recorded 1–45
# - Date: The date of the week where this observation was taken.
# - Temperature: Temperature of the region during that week.
# - Fuel_Price: Fuel Price in that region during that week.
# - MarkDown1:5 : Represents the Type of markdown and what quantity was available during that week.
# - CPI: Consumer Price Index during that week.
# - Unemployment: The unemployment rate during that week in the region of the store.
# - IsHoliday: Boolean value representing a holiday week or not.
# 
# ***primary key here is a combination of (Store,Date)***
# 
# 
# 
# 
#     
#  

# # <p style="background-color:coral;font-family:newtimeroman;color:white;font-size:150%;text-align:center;border-radius:20px 60px;">Dataset Importing and Querying</p>
# 
#     We will load all 3 datasets and merge them  into one big dataset that gives whole data. 
# 
# ### Note :
# 
# *Since here we are only predicting Store level sales we will group the dataframes such that Department level data gets eliminated and take the sum of department level sales to give the store level sales.*

# In[1]:


import warnings 
warnings.filterwarnings('ignore')

import time
t = time.time()

print('Importing startred...')

# base libraries
import os
import numpy as np
import pandas as pd
import re
from scipy import stats
from random import randint
from datetime import datetime


# visualization libraries
import matplotlib.pyplot as plt
import matplotlib 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#import missingno as msno
import plotly.express as px


# preprocessing libraries
from sklearn.model_selection import (TimeSeriesSplit,
                                     GridSearchCV,
                                     RandomizedSearchCV,
                                     train_test_split, 
                                     KFold, 
                                     StratifiedKFold,
                                    cross_val_score)

from sklearn.preprocessing import (LabelEncoder,
                                   StandardScaler, 
                                   MinMaxScaler, 
                                   OrdinalEncoder)
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel


# metrics
from sklearn.metrics import (mean_squared_error, 
                             r2_score, 
                             mean_absolute_error)
from sklearn.metrics import make_scorer


# modeling algos
from sklearn.linear_model import (LogisticRegression,
                                  Lasso, 
                                  ridge_regression,
                                  LinearRegression)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, 
                              RandomForestRegressor,
                              VotingRegressor, 
                              GradientBoostingRegressor)
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import (LGBMRegressor,
                      early_stopping)

from sklearn.base import clone ## sklearn base models for stacked ensemble model
from sklearn.pipeline import make_pipeline


print('Done, All the required modules are imported. Time elapsed: {}sec'.format(time.time()-t))


# #### Load and read data

# In[2]:


walmart_train = pd.read_csv('train.csv')
walmart_feature = pd.read_csv('features.csv')
walmart_store = pd.read_csv('stores.csv')
walmart_test = pd.read_csv('test.csv')


# In[3]:


print(walmart_train.columns)
print(walmart_feature.columns)
print(walmart_test.columns)


# In[4]:


walmart_test.head()


# In[5]:


walmart_train.head()


# #### Merging all the datasets into one place for easier test and analysis.

# In[6]:


result = pd.merge(walmart_train, walmart_store, how='inner', on='Store', left_on=None, right_on=None,
        left_index=False, right_index=False, sort=False,
        suffixes=('_x', '_y'), copy=True, indicator=False)

data = pd.merge(result, walmart_feature, how='inner', on=['Store','Date','IsHoliday'], left_on=None, right_on=None,
        left_index=False, right_index=False, sort=False,
        suffixes=('_x', '_y'), copy=False, indicator=False)


# In[7]:


result.head()


# In[8]:


result1 = pd.merge(walmart_test, walmart_store, how='inner', on='Store', left_on=None, right_on=None,
        left_index=False, right_index=False, sort=False,
        suffixes=('_x', '_y'), copy=True, indicator=False)

test = pd.merge(result1, walmart_feature, how='inner', on=['Store','Date','IsHoliday'], left_on=None, right_on=None,
        left_index=False, right_index=False, sort=False,
        suffixes=('_x', '_y'), copy=False, indicator=False)


# # <p style="background-color:coral;font-family:newtimeroman;color:white;font-size:120%;text-align:center;border-radius:20px 60px;">Data Cleaning</p>
# 
# <p style="background-color:white;font-family:calibri;color:lightseagreen;font-size:200%;border-radius:20px 60px;">Now let's look through the data and do some basic data cleaning</p> 
# 

# In[9]:


print(data.head())
print(test.head())


# In[10]:


print(data.shape)
print(test.shape)


# In[11]:


# % of missing.
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[12]:


# FINDING UNIQUE VALUE FOR CATEGORIAL COLUMNS IN TRAIN
for i in ['Store','Dept','Fuel_Price','Unemployment']:#,'Day','Month','Year']:
    print(f'{i}: {data[i].nunique()}')


# In[13]:


# FINDING UNIQUE VALUE FOR CATEGORIAL COLUMNS IN TEST
for i in ['Store','Dept','Fuel_Price','Unemployment']:#,'Day','Month','Year']:
    print(f'{i}: {test[i].nunique()}')


# In[14]:


# GETTING THE DATA TYPES+
data.info()


# In[15]:


# GETTING THE DATA TYPES+
test.info()


# In[16]:


#let's encode the categorical column : IsHoliday

data['IsHoliday'] = data['IsHoliday'].apply(lambda x: 1 if x == True else 0)
# Will convert the bool to 1 and 0 for easier use later.
#data.IsHoliday=data.IsHoliday.map(lambda x: 1 if x==True else 0)

test['IsHoliday'] = test['IsHoliday'].apply(lambda x: 1 if x == True else 0)


# In[17]:


# Lets look into the null values
print(data.isnull().sum())
print(test.isnull().sum())


# **Wow!! Now that's huge. More Than 65% of value are missing in MarkDown values**
# 
# *We can impute sum values as of now for the missing  and will later decide  whether to use Markdown for modeling or should take some other approach for imputing or whether to discard MarkDowns completely*
# 

# For the test data, we have null values in CPI and Unemployment as well

# In[18]:


## setting all missing values in markdown columns to -500 for now. We will treat them later while performing Feature scaling
data['MarkDown1'].fillna(-500, inplace=True)
data['MarkDown2'].fillna(-500, inplace=True)
data['MarkDown3'].fillna(-500, inplace=True)
data['MarkDown4'].fillna(-500, inplace=True)
data['MarkDown5'].fillna(-500, inplace=True)


# In[19]:


## setting all missing values in markdown columns to -500 for now. We will treat them later while performing Feature scaling
test['MarkDown1'].fillna(-500, inplace=True)
test['MarkDown2'].fillna(-500, inplace=True)
test['MarkDown3'].fillna(-500, inplace=True)
test['MarkDown4'].fillna(-500, inplace=True)
test['MarkDown5'].fillna(-500, inplace=True)


# In[20]:


#Irregular Data (Outliers) 

columns = ['Store','Dept' ,'Size','Temperature' ,'Fuel_Price' ,'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Unemployment','CPI']

for i in columns:
    plt.figure()
    plt.tight_layout()
    sns.set(rc={"figure.figsize":(8, 5)})
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True)
    plt.gca().set(xlabel= i,ylabel='Frequency')
    sns.boxplot(data[i], ax=ax_box , linewidth= 1.0)
    sns.histplot(data[i], ax=ax_hist , bins = 10,kde=True)


# In[21]:


# converting the date to the required format
data['Date'] = pd.to_datetime(data['Date'])
#add a 'week' column to the dataset for further analysis
data['Week'] = data['Date'].dt.week
#add a 'quarter' column to the dataset for further analysis
data['quarter'] = data['Date'].dt.quarter
#add a 'year',  'month','Day' column to the dataset for further analysis
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day


# In[22]:


# converting the date to the required format
test['Date'] = pd.to_datetime(test['Date'])
#add a 'week' column to the dataset for further analysis
test['Week'] = test['Date'].dt.week
#add a 'quarter' column to the dataset for further analysis
test['quarter'] = test['Date'].dt.quarter
#add a 'year',  'month','Day' column to the dataset for further analysis
test['year'] = test['Date'].dt.year
test['month'] = test['Date'].dt.month
test['day'] = test['Date'].dt.day


# In[23]:


data.head()


# In[24]:


test.head()


# In[25]:


# FINDING UNIQUE VALUE FOR CATEGORIAL COLUMNS IN TRAIN
    
    
for col in ['Week','quarter','year','month','day']:
    print(f'{col}: {data[col].unique()}')
    


# In[26]:


# FINDING UNIQUE VALUE FOR CATEGORIAL COLUMNS IN Test
    
    
for col in ['Week','quarter','year','month','day']:
    print(f'{col}: {test[col].unique()}')
    


# In[27]:


print(data.describe())#result is numeric because there are no categorial columns
print(test.describe())


# # EXPLORATORY DATA ANALYSIS(EDA)
# 

# **A histogram is representation of the distribution of numerical data, where the data are binned and the count for each bin is represented. More generally, in Plotly a histogram is an aggregated bar chart, with several possible aggregation functions (e.g. sum, average, count...) which can be used to visualize data on categorical and date axes as well as linear axes.**

# In[28]:


fig = px.histogram(data, x='Temperature', y ='Weekly_Sales', color='IsHoliday')#,color = "red")#, marginal='box')
fig.show()
# corelating the temperatue and weekly sales


# Low temperature and High temperature are effecting the sales and holiday doesn't have huge impact on the sales based on the temparature.

# In[29]:


fig = px.histogram(data, x='Unemployment', y ='Weekly_Sales',color='IsHoliday', title = 'How Unemployment affect sales')
fig.show()


# During the low Unemployment and high unemployment rate - weekly sales is less,assuming the average median value for unemployment(7.874) there is no proper correlation can be done.Further analysis is done under Time series analysis.

# In[30]:


px.histogram(data, x='Fuel_Price', y ='Weekly_Sales', color='IsHoliday')


# In[31]:


px.histogram(data, x='CPI', y ='Weekly_Sales', color='IsHoliday')


# In[32]:


weekly_sales = data['Weekly_Sales'].groupby(data['Store']).mean()
plt.figure(figsize=(20,8))
plt.style.use('default')
sns.barplot(weekly_sales.index, weekly_sales.values)
plt.grid()
plt.title('Average Sales - per Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.show()


# In[33]:


weekly_sales = data['Weekly_Sales'].groupby(data['Dept']).mean()
plt.figure(figsize=(20,8))
plt.style.use('default')
sns.barplot(weekly_sales.index, weekly_sales.values)
plt.grid()
plt.title('Average Sales - per Department', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Department', fontsize=16)
plt.show()


# In[34]:


#Create a dataframe for heatmap
data_heatmap_df=data.copy()

# Eliminating all the columns that are not continuous/binary  variables from the heatmap section.
data_heatmap_df.drop(['Store','day','month','year','Date','Type'], axis=1,inplace=True)


# Lets look the correlation matrix and heat map of the 

## Correlation Heat map
def correlation_heat_map(df):
    corrs = df.corr()

    # Set the default matplotlib figure size:
    fig, ax = plt.subplots(figsize=(12,8))

    # Generate a mask for the upper triangle (taken from seaborn example gallery)
    mask = np.zeros_like(corrs, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Plot the heatmap with seaborn.
    # Assign the matplotlib axis the function returns. This will let us resize the labels.
    ax = sns.heatmap(corrs, mask=mask, annot=True, cmap='Pastel1_r')

    # Resize the labels.
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=90)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)

    # If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
    plt.show()


# 
# <p style="background-color:white;font-family:timesnewroman;color:lightseagreen;font-size:200%;border-radius:20px 60px;"> Statistical analysis and correlations</p>

# In[35]:


correlation_heat_map(data_heatmap_df)

#inference: By checking the direct correlation of features there is no much promising correlations. 
#           There are no much correlation within the features as well. In a way this is good because 
#           there won't be multicollinearity that we have to take care while running models.


# In[36]:


sns.set(style="white")

corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title('Correlation Matrix', fontsize=18)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()


# In[37]:


def scatter(dataset, column):
    plt.figure()
    plt.scatter(data[column] , data['Weekly_Sales'], color = 'turquoise')
    plt.ylabel('Weekly Sales')
    plt.xlabel(column)

scatter(data, 'Fuel_Price')
scatter(data, 'Size')
scatter(data, 'CPI')
scatter(data, 'Type')
scatter(data, 'IsHoliday')
scatter(data, 'Unemployment')
scatter(data, 'Temperature')
scatter(data, 'Store')
scatter(data, 'Dept')


# In[38]:


plt.figure(figsize=(18,8))
sns.lineplot( data = data, x = 'Size', y =  'Weekly_Sales', hue = 'IsHoliday');


# **The distplot figure factory displays a combination of statistical representations of numerical data, such as histogram, kernel density estimation or normal curve, and rug plot**

# In[39]:


import plotly.graph_objs as go
from plotly import tools

fig = go.Figure()
df_weeks = data.groupby('Week').sum()
#fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['Weekly_Sales'], name = 'Weekly Sales', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown1'], name = 'MarkDown1', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown2'], name = 'MarkDown2', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown3'], name = 'MarkDown3', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown4'], name = 'MarkDown4', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown5'], name = 'MarkDown5', mode = 'lines') )
fig.update_layout(title = 'Sales vs Markdown', xaxis_title = 'Weeks')


# In[40]:


weekly_sales_2010 = data[data.year==2010]['Weekly_Sales'].groupby(data['Week']).mean()
weekly_sales_2011 = data[data.year==2011]['Weekly_Sales'].groupby(data['Week']).mean()
weekly_sales_2012 = data[data.year==2012]['Weekly_Sales'].groupby(data['Week']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1, 53, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Average Weekly Sales - Per Year', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Week', fontsize=16)
plt.show()


# In[41]:


monthly_sales = data['Weekly_Sales'].groupby(data['month']).sum()
print(monthly_sales)


# 
# # <p style="background-color:coral;font-family:newtimeroman;color:white;font-size:150%;text-align:center;border-radius:20px 60px;">Detailed Time-Series Analysis</p>
# 

# In[42]:


monthly_sales_2010 = data[data.year==2010]['Weekly_Sales'].groupby(data['month']).mean()
monthly_sales_2011 = data[data.year==2011]['Weekly_Sales'].groupby(data['month']).mean()
monthly_sales_2012 = data[data.year==2012]['Weekly_Sales'].groupby(data['month']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(monthly_sales_2010.index, monthly_sales_2010.values)
sns.lineplot(monthly_sales_2011.index, monthly_sales_2011.values)
sns.lineplot(monthly_sales_2012.index, monthly_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1, 13, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Average Monthly Sales - Per Year', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.show()


# #### Note :
# 
# **As we can see, there is one important Holiday not included in 'IsHoliday'. It's the Easter Day. It is always in a Sunday, but can fall on different weeks.**
# 
# In 2010 is in Week 13
# 
# In 2011, Week 16
# 
# Week 14 in 2012
# 
# Week 13 in 2013 for **Test set**
# 
# **So, we can change to 'True' these Weeks in each Year.**

# In[43]:


weekly_sales_mean = data['Weekly_Sales'].groupby(data['Date']).mean()
weekly_sales_median = data['Weekly_Sales'].groupby(data['Date']).median()
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values, color = 'indigo')
sns.lineplot(weekly_sales_median.index, weekly_sales_median.values, color = 'tomato')
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Weekly Sales - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# **# converting days and months from numerics to categories**

# In[44]:


days = {0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5: 'Friday',6:'Saturday'}
data['day'] = data['day'].map(days)
months={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'Novemenber',12:'December'}
data['month']= data['month'].map(months)
data.head()


# In[45]:


quarter_sales_2010 = data[data.year==2010]['Weekly_Sales'].groupby(data['quarter']).mean()
quarter_sales_2011 = data[data.year==2011]['Weekly_Sales'].groupby(data['quarter']).mean()
quarter_sales_2012 = data[data.year==2012]['Weekly_Sales'].groupby(data['quarter']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(quarter_sales_2010.index, quarter_sales_2010.values)
sns.lineplot(quarter_sales_2011.index, quarter_sales_2011.values)
sns.lineplot(quarter_sales_2012.index, quarter_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1, 5, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Average Quarter Sales - Per Year', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Quarter', fontsize=16)
plt.show()


# In[46]:


yearly_sales = data.groupby('year').sum()
yearly_sales['Weekly_Sales']


# In[47]:


plt.figure(figsize=(20,8))
plt.style.use('default')
sns.barplot(x = yearly_sales.index, y = yearly_sales['Weekly_Sales'])
plt.grid()
plt.title('Average Sales - per Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('year', fontsize=16)
plt.show()
# a seasonal pattern can be analysied so we will dig deep into timeseries forecasting


# In[48]:


plt.pie(data.groupby('year')['Weekly_Sales'].sum(),labels=data['year'].unique(),autopct='%1.2f%%',colors=['hotpink','green','violet'])
plt.title('Annual Sales')


# In[49]:


df2 = data.groupby('day')['Weekly_Sales'].sum().reset_index()
plt.figure(figsize=(10,8))
plt.pie(df2['Weekly_Sales'],labels= df2['day'],autopct='%1.2f%%')


# In[50]:


plt.figure(figsize=(10,10))
df3 = data.groupby('month')['Weekly_Sales'].sum().reset_index()
plt.pie(df3['Weekly_Sales'],labels=df3['month'],autopct='%1.2f%%')


# In[51]:


df4 = data.groupby('IsHoliday')['Weekly_Sales'].sum().reset_index()
plt.pie(df4['Weekly_Sales'],labels= ['Non Special Holiday Week','Special Holiday Week'],autopct='%1.2f%%',startangle=90,explode=[0,0.3],shadow=True,colors=['violet','pink'])


# In[52]:


sns.barplot(data["month"], data["Weekly_Sales"] )
plt.show()


# In[53]:


monthly_temp_2010 = data[data.year==2010]['Temperature'].groupby(data['month']).mean()
monthly_temp_2011 = data[data.year==2011]['Temperature'].groupby(data['month']).mean()
monthly_temp_2012 = data[data.year==2012]['Temperature'].groupby(data['month']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(monthly_temp_2010.index, monthly_temp_2010.values)
sns.lineplot(monthly_temp_2011.index, monthly_temp_2011.values)
sns.lineplot(monthly_temp_2012.index, monthly_temp_2012.values)
plt.grid()
plt.xticks(np.arange(0, 12, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Average Monthly Temperature - Per Year', fontsize=18)
plt.ylabel('Temperature', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.show()


# In[54]:


monthly_Unemp_2010 = data[data.year==2010]['Unemployment'].groupby(data['month']).mean()
monthly_Unemp_2011 = data[data.year==2011]['Unemployment'].groupby(data['month']).mean()
monthly_Unemp_2012 = data[data.year==2012]['Unemployment'].groupby(data['month']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(monthly_Unemp_2010.index, monthly_Unemp_2010.values)
sns.lineplot(monthly_Unemp_2011.index, monthly_Unemp_2011.values)
sns.lineplot(monthly_Unemp_2012.index, monthly_Unemp_2012.values)
plt.grid()
plt.xticks(np.arange(0, 12, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Average Monthly Unemployment - Per Year', fontsize=18)
plt.ylabel('Unemployment', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.show()


# In[55]:



fig = px.line( data_frame = df_weeks, x = df_weeks.index, y = 'Weekly_Sales', labels = {'Weekly_Sales' : 'Weekly Sales', 'x' : 'Weeks' }, title = 'Sales over weeks')
fig.update_traces(line_color='blue', line_width=3)


# ##### Sales is less
# **January(Week1 - Week4 ) , Week 44: October 30 to November 5, 2022 Week 45: November 6 to November 12, 2022 Week 46: November 13 to November 19, 2022 ( week before the Thanks giving ) , Week 48 ( Week after thanks giving ) , Week 52 ( Year End or Week after Christmas )**
# 
# 

# In[56]:


data.head()


# In[57]:


plt.figure(figsize=[20,10])
df_quarter = data.groupby(["Store","Dept","year","quarter"])[["Weekly_Sales"]].sum().reset_index()


# In[58]:


plt.figure(figsize=(20,8))
plt.style.use('default')
base_color = sns.color_palette('plasma',n_colors=5)
sns.barplot(data=df_quarter, x='year', y='Weekly_Sales',hue='quarter', palette = base_color)
plt.grid()
plt.title('Weekly Sales - per year ', fontsize=18)
plt.ylabel('sales', fontsize=16)
plt.xlabel('year', fontsize=16)
plt.show()


# for i in range(1,df_quarter["Store"].nunique()+1):   
#     store = df_quarter.where(df_quarter["Store"]==i)
#     #print(store)
#     #print(store)
#     dt = "Department wise Weekly Sales - per year in Store number" + str(i)
#     dept_quarter = store.groupby(["Dept","year"])[["Weekly_Sales"]].sum().reset_index()
#     #print(dept_quarter)
#     plt.figure(figsize=(20,8))
#     plt.style.use('default')
#     base_color = sns.color_palette('pastel',n_colors=5)
#     sns.barplot(data=dept_quarter, x='year', y='Weekly_Sales',hue='Dept')
#     plt.grid()
#     plt.title(dt, fontsize=18)
#     plt.ylabel('sales', fontsize=16)
#     plt.xlabel('year', fontsize=16)
#     plt.show()
#     #plt.tight_layout
#     

# # Modelling

# # DATA PROCESSING FOR MODEL FIT

# In[59]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


# In[60]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True)
plt.title('Correlation Matrix')


# In[61]:


data = data.drop(['CPI','Unemployment','Date','year','day'],axis=1)
test = test.drop(['CPI','Unemployment','Date','year','day'],axis=1)


print(data.head())
print(test.head())


# In[62]:


months={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'Novemenber':11,'December':12}
data['month']= data['month'].map(months)


# In[63]:


Type={'A':13,'B':14,'C':15}
data['Type']= data['Type'].map(Type)
test['Type']= test['Type'].map(Type)


# In[64]:


x_train,x_test,y_train,y_test=train_test_split(data.drop(['Weekly_Sales'],axis=1),data['Weekly_Sales'],test_size=0.30,random_state=0)


# In[65]:


models=[DecisionTreeRegressor(),LinearRegression(),RandomForestRegressor(),KNeighborsRegressor(n_neighbors = 5),BaggingRegressor(),LinearSVR()]
model_names=['DecisionTreeRegressor','LinearRegression','RandomForestRegressor','KNeighborsRegressor','BaggingRegressor','LinearSVR']
rmse=[]
d={}
acc_rf=[]

for model in range (len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    test_pred=clf.predict(x_test)
    #rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
    acc_rf.append(round(clf.score(x_train, y_train) * 100, 2))
    rmse.append(np.sqrt(mean_squared_error(y_test,test_pred)))
d={'Modelling Algo':model_names,'RMSE':rmse,'Acc':acc_rf}  
MS=pd.DataFrame(d)
MS


# In[66]:


clf=RandomForestRegressor()
clf.fit(x_train,y_train)
test_pred=clf.predict(test)
d={'Store':test['Store'],'Dept':test['Dept'],'Week':test['Week'],'Weekly_sales':test_pred}
ans=pd.DataFrame(d)
print(ans)


# In[67]:


storedata=[]
deptdata=[]
weekdata=[]
salesdata=[]

for i in range(1,ans["Store"].nunique()+1):
    store = ans.where(ans["Store"]==i)
    dept_sales_id = max(store[['Weekly_sales']].idxmax())    
    deptnum=ans.iloc[dept_sales_id,1]
    week=ans.iloc[dept_sales_id,2]
    max_sale = ans.iloc[dept_sales_id,3]    
    storedata.append(i)
    deptdata.append(deptnum)
    weekdata.append(week)
    salesdata.append(max_sale)
d={'StoreNumber':storedata,'DepartmentNumber':deptdata,'Week':weekdata,'Max_Weekly_sales':salesdata}
final=pd.DataFrame(d)
    #print("Max Sales in Department" +str(s)+" in Week "+str(week)+"is "+str(max_sale))

final

