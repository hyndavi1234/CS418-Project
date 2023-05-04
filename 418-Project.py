#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# * The Electricity Usage Forecasting & Price Prediction project aims to develop a machine learning model that can accurately forecast electricity consumption and predict its price.
# * The project will involve collecting historical data on electricity usage and pricing from data sources. This data will then be pre-processed and analyzed to identify patterns and trends in electricity consumption and pricing.
# * Next, various machine learning algorithms such as time-series analysis, regression analysis, and neural networks will be applied to the data to create a predictive model. The model will be trained and validated using historical data, and its accuracy will be tested against new, unseen data.
# * This could be used by energy companies, policymakers, and consumers to make informed decisions about energy usage, pricing, and resource planning.
# 
# 
# ## Data
# The source of the data is the following link: [LINK](https://data.world/houston/houston-electricity-bills)
# 
# There are 4 files, they are:
# 1. July 2011 to June 2012 excel file - 57,430 rows and 24 columns
# 2. May 2012 to April 2013 excel file - 65,806 rows and 24 columns
# 3. July 2012 to June 2013 excel file - 66,776 rows and 24 columns
# 4. July 2013 to June 2014 excel file - 67,838 rows and 24 columns
# 
# The following is a brief summary of the data cleaning steps we performed:
# * First, we identified missing data and decided how to handle it, either by imputing the missing values or excluding the observations entirely based on the respective columns. 
# * Next, identified and corrected any errors and inconsistencies in the data, such as incorrect values, and formatting the date column. 
# * We also removed duplicate data and standardized the format of data across different tables, since we were working with multiple tables and there was overlap between the time period of the datasets which we had to account for.
# 
# The data tables contain information regarding the building address, location, service number, billing dates, total amount due. 
# 
# Description of each column 
# 1. Reliant Contract No: A unique identifier for each contract. 
# 2. Service Address: Address for the service location
# 3. Meter No: Meter number for the service location.
# 4. ESID: Electric Service Identifier for the service location.
# 5. Business Area: Business area code for the service location.
# 6. Cost Center: Cost center code for the service location.
# 7. Fund: Fund code for the service location.
# 8. Bill Type: Type of bill (e.g. "T" for "Total", "P" for "Partial", etc.). 
# 9. Bill Date: Date the bill was generated. 
# 10. Read Date: Date the meter was read. 
# 11. Due Date: Due date for the bill. 
# 12. Meter Read: Meter reading for the service location. 
# 13. Base Cost: TBase cost for the service. 
# 14. T&D Discretionary: Transmission and Distribution Discretionary charge for the service. 
# 15. T&D Charges: Transmission and Distribution charge for the service. 
# 16. Current Due: Current due amount for the service.
# 17. Index Charge: Index charge for the service. 
# 18. Total Due: Total due amount for the service. 
# 19. Franchise Fee: Franchise fee for the service. 
# 20. Voucher Date: Date the voucher was issued for the service. 
# 21. Billed Demand: Billed demand for the service in KVA. 
# 22. kWh Usage: Kilowatt-hour usage for the service. 
# 23. Nodal Cu Charge:  Nodal Cu Charge for the service. 
# 24. Adder Charge:  Adder Charge for the service.
# 
# Statistical Data Type of Each Column 
# 1. Reliant Contract No: integer (ratio)
# 2. Service Address: string (nominal)
# 3. Meter No: integer (nominal)
# 4. ESID: integer (nominal)
# 5. Business Area: integer (ratio))
# 6. Cost Center: integer (ratio)
# 7. Fund: integer (ratio)
# 8. Bill Type: string (nominal)
# 9. Bill Date: date (nominal)
# 10. Read Date: date (nominal)
# 11. Due Date: date (nominal)
# 12. Meter Read: integer (ratio)
# 13. Base Cost: float (nominal)
# 14. T&D Discretionary: float (nominal)
# 15. T&D Charges: float (nominal)
# 16. Current Due: float (nominal)
# 17. Index Charge: float (nominal)
# 18. Total Due: float (nominal)
# 19. Franchise Fee: float (nominal)
# 20. Voucher Date: date (nominal)
# 21. Billed Demand (KVA): integer (nominal)
# 22. kWh Usage: integer (nominal)
# 23. Nodal Cu Charge: float (nominal)
# 24. Adder Charge: float (nominal)
# 
# ## Problem
# The key issue in generating electricity is to determine how much capacity to generate in order to meet future demand. 
# 
# Electricity usage forecasting involves predicting the demand for electricity over a specific eriod. This process has several uses, including energy procurement, where it helps suppliers purchase the right amount of energy to ensure a steady supply.
# 
# The advancement of smart infrastructure and integration of distributed renewable power has raised future supply, demand, and pricing uncertainties. This unpredictability has increased interest in price prediction and energy analysis.
# 
# ## Research Questions
# 1. Previous electricity usage data can be used for predicting the usage for future (Time-Series) - Hyndavi 
# 2. Group areas based on their energy consumption (Clustering) - Sunil
# 3. Electricity usage can be predicted by using correlated features (Regression) - Sourabh
# 4. Classification of bill type can be done using features in the data (Classification) - Sharmisha

# ## Import Statements

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
import requests,urllib,os,pickle
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from IPython import display

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from tqdm import tqdm_notebook
from itertools import product
import math 
from statistics import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

pd.options.display.max_columns=25

import warnings
warnings.filterwarnings('ignore')


# ## Data FY 2012 - Hyndavi

# In[ ]:


data_2012 = pd.read_excel(
'houston-houston-electricity-bills/coh-fy2012-ee-bills-july2011-june2012.xls'
)
orig_shape_2012 = data_2012.shape[0]

data_2012.shape


# In[ ]:


data_2012.head(5)


# ### Checking Nulls

# In[ ]:


data_2012.isna().sum()


# ### Checking Adjustment ($) column

# In[ ]:


data_2012['Adjustment ($)'].value_counts(dropna=False)


# The column does not have any relevant information based on the above reported values. Electing to drop the column.

# In[ ]:


data_2012.drop(columns=['Adjustment ($)'], inplace=True)


# ### Checking Unique Number of Customers

# There are quite a few columns in the dataset that signify relating to a unique person/house/business. Checking the unique counts of such columns.

# In[ ]:


check_unique_columns = ['Reliant Contract No', 'Service Address ', 'Meter No', 
                        'ESID', 'Business Area', 'Cost Center',]

for col in check_unique_columns:
    print(f'Number of Unique Values in {col}: {data_2012[col].nunique()}')


# Based on the above reported values and further research online:
# 
# ESID signifies a unique ID provided to each customer subscribed to the electricity board. It would be best to choose ESID and Service Address columns going forward as these would provide number of unique customers and the areas (streets) where higher usage of electricity occurs.
# 
# Business Area signifies a grouping a number of buildings which covers a certain area. This would be useful usage patterns grouped by certain zones in the city.

# ### Checking Bill Type

# In[ ]:


data_2012['Bill Type'].value_counts(dropna=False)


# Bill Type could signify the type of the connection given. Since commercial, residential and government spaces would have different type of pricing and needs this column could be capturing that information.

# In[ ]:


(
    data_2012['Service Address '].nunique(), 
    data_2012['Meter No'].nunique(), 
    data_2012['ESID'].nunique()
)


# The next 3 columns are: Bill Date, Read Date and Due Date. Of these it would be best to choose the Bill date across all the data files to keep the data consistent. 

# ### Electricity Usage Statistics

# In[ ]:


data_2012[['Meter Read', 'Billed Demand ', 'kWh Usage']].describe()


# There are 3 columns that denote the amount of electricity: Meter Read, Billed Demand, kWh Usage.
# 
# Using kWh Usage as a standard unit of measurement.

# In[ ]:


data_2012[[
    'Base Cost ($)', 'T&D Discretionary ($)', 'T&D Charges ($)', 
    'Current Due ($)', 'Total Due ($)', 'Franchise Fee ($)', 
    'Nodal Cu Charge ($)', 'Reliability Unit Charge ($)'
     ]].describe()


# Reliability Unit Charge does not contain any useful information. Electing to drop that column.
# 
# The columns other than Current Due or Total Due are adding up the value present in these two columns. Going forward choosing the column Total Due ($). 
# Based on the above statistics the columns Current Due and Total Due represent the same value. 

# ### Selecting and Filtering Columns

# In[ ]:


data_2012.columns


# Based on the above analysis of the dataset choosing the following columns:
# 
# 1. ESID
# 2. Business Area
# 3. Service Address 
# 3. Bill Type
# 4. Bill Date
# 5. Total Due ($)
# 6. kWh Usage

# In[ ]:


data_2012 = data_2012[[
    'ESID', 'Business Area', 'Service Address ', 'Bill Type',
    'Bill Date', 'Total Due ($)', 'kWh Usage'
]]


# In[ ]:


rename_cols = {
    'ESID': 'esid',
    'Business Area': 'business_area',
    'Service Address ': 'service_address',
    'Bill Type': 'bill_type',
    'Bill Date': 'bill_date',
    'Total Due ($)': 'total_due',
    'kWh Usage': 'kwh_usage'
}

data_2012_main = data_2012.rename(columns=rename_cols)


# Checking for Nulls again and dtypes

# In[ ]:


data_2012_main.isna().sum()


# In[ ]:


data_2012_main.dtypes


# In[ ]:


data_2012_main.shape


# In[ ]:


zscore_2012 = stats.zscore(data_2012_main[['total_due', 'kwh_usage']])

zscore_2012


# Each zscore value signifies how many standard deviations away an individual value is from the mean. This is a good indicator to finding outliers in the dataframe.
# 
# Usually z-score=3 is considered as a cut-off value to set the limit. Therefore, any z-score greater than +3 or less than -3 is considered as outlier which is pretty much similar to standard deviation method

# In[ ]:


# data_2012_main = data_2012_main[(np.abs(zscore_2012) < 3).all(axis=1)]

data_2012_main.shape


# The number of rows has decreased from 57,430 to 57,025. So 405 rows were outliers based on the data.

# In[ ]:


data_2012_main.head(5)


# In[ ]:


orig_shape_2012 - data_2012_main.shape[0]


# In[ ]:


data_2012.to_csv('electricity_usage_data_2012.csv', index=False)


# The trend graph of both the cost and energy usage is the same as the value of cost = energy usage times the cost per unit.

# ## Performing a Similar Analysis on FY 2013-1, FY 2013-2, and FY 2014 before merging datasets

# ## Data FY 2013-1 - Sourabh

# The code for the cleaning performed in this section is in the IPYNB: 'July 2012 to June 2013.ipynb'

# In[ ]:


data_2013 = pd.read_excel(
'houston-houston-electricity-bills/coh-fy2013-ee-bills-july2012-june2013.xlsx'
)
orig_shape_2013 = data_2013.shape[0]

data_2013.shape


# In[ ]:


data_2013.head(5)


# ### Checking Nulls

# In[ ]:


data_2013.isna().sum()


# ### Checking Index Charge ($) column - This was previously Adjustment

# In[ ]:


data_2013['Index Charge ($)'].value_counts(dropna=False)


# The column does not have any relevant information based on the above reported values. Electing to drop the column.

# In[ ]:


data_2013.drop(columns=['Index Charge ($)'], inplace=True)


# ### Checking Unique Number of Customers

# There are quite a few columns in the dataset that signify relating to a unique person/house/business. Checking the unique counts of such columns.

# In[ ]:


check_unique_columns = ['Reliant Contract No', 'Service Address ', 'Meter No', 
                        'ESID', 'Business Area', 'Cost Center',]

for col in check_unique_columns:
    print(f'Number of Unique Values in {col}: {data_2013[col].nunique()}')


# Based on the above reported values and further research online:
# 
# ESID signifies a unique ID provided to each customer subscribed to the electricity board. It would be best to choose ESID and Service Address columns going forward as these would provide number of unique customers and the areas (streets) where higher usage of electricity occurs.
# 
# Business Area signifies a grouping a number of buildings which covers a certain area. This would be useful usage patterns grouped by certain zones in the city.

# ### Checking Bill Type

# In[ ]:


data_2013['Bill Type'].value_counts(dropna=False)


# Bill Type could signify the type of the connection given. Since commercial, residential and government spaces would have different type of pricing and needs this column could be capturing that information.

# In[ ]:


(
    data_2013['Service Address '].nunique(), 
    data_2013['Meter No'].nunique(), 
    data_2013['ESID'].nunique()
)


# The next 3 columns are: Bill Date, Read Date and Due Date. Of these it would be best to choose the Bill date across all the data files to keep the data consistent. 

# ### Electricity Usage Statistics

# In[ ]:


data_2013[['Meter Read', 'Billed Demand (KVA)', 'kWh Usage']].describe()


# There are 3 columns that denote the amount of electricity: Meter Read, Billed Demand, kWh Usage.
# 
# Using kWh Usage as a standard unit of measurement.

# In[ ]:


data_2013[[
    'Base Cost ($)', 'T&D Discretionary ($)', 'T&D Charges ($)', 
    'Current Due ($)', 'Total Due ($)', 'Franchise Fee ($)', 
    'Nodal Cu Charge ($)', 'Adder Charge ($)'
     ]].describe()


# Adder Charge ($) does not contain any useful information. Electing to drop that column. Previously this column was Reliability Unit Charge.
# 
# The columns other than Current Due or Total Due are adding up the value present in these two columns. Going forward choosing the column Total Due ($). 
# Based on the above statistics the columns Current Due and Total Due represent the same value. 

# Based on the above analysis of the dataset choosing the following columns:
# 
# 1. ESID
# 2. Business Area
# 3. Service Address 
# 3. Bill Type
# 4. Bill Date
# 5. Total Due ($)
# 6. kWh Usage

# ### Selecting and Filtering Columns

# In[ ]:


data_2013 = data_2013[[
    'ESID', 'Business Area', 'Service Address ', 'Bill Type',
    'Bill Date', 'Total Due ($)', 'kWh Usage'
]]


# In[ ]:


rename_cols = {
    'ESID': 'esid',
    'Business Area': 'business_area',
    'Service Address ': 'service_address',
    'Bill Type': 'bill_type',
    'Bill Date': 'bill_date',
    'Total Due ($)': 'total_due',
    'kWh Usage': 'kwh_usage'
}

data_2013_main = data_2013.rename(columns=rename_cols)


# Checking for Nulls again and dtypes

# In[ ]:


data_2013_main.isna().sum()


# In[ ]:


data_2013_main.dropna(subset=['kwh_usage'], inplace=True)


# In[ ]:


data_2013_main.isna().sum()


# In[ ]:


data_2013_main.dtypes


# In[ ]:


data_2013_main.shape


# In[ ]:


zscore_2013 = stats.zscore(data_2013_main[['total_due', 'kwh_usage']])

zscore_2013


# Each zscore value signifies how many standard deviations away an individual value is from the mean. This is a good indicator to finding outliers in the dataframe.
# 
# Usually z-score=3 is considered as a cut-off value to set the limit. Therefore, any z-score greater than +3 or less than -3 is considered as outlier which is pretty much similar to standard deviation method

# In[ ]:


# data_2013_main = data_2013_main[(np.abs(zscore_2013) < 3).all(axis=1)]

data_2013_main.shape


# The number of rows has decreased from 66,775 to 66,360. So 415 rows were outliers based on the data.

# In[ ]:


data_2013_main.head(5)


# In[ ]:


orig_shape_2013 - data_2013_main.shape[0]


# In[ ]:


data_2013_main.to_csv('electricity_usage_data_2013.csv', index=False)


# ## Data FY 2013-2 - Sunil

# In[ ]:


data_2013_2 = pd.read_excel(
'houston-houston-electricity-bills/coh-ee-bills-may2012-apr2013.xlsx'
)
orig_shape_2013_2 = data_2013_2.shape[0]

data_2013_2.shape


# In[ ]:


data_2013_2.head(5)


# ### Checking Nulls

# In[ ]:


data_2013_2.isna().sum()


# ### Checking Adjustment ($) column 

# This column was named Index Charge in the other FY 2013 electricity usage data file

# In[ ]:


data_2013_2['Adjustment ($)'].value_counts(dropna=False)


# The column does not have any relevant information based on the above reported values. Electing to drop the column.

# In[ ]:


data_2013_2.drop(columns=['Adjustment ($)'], inplace=True)


# ### Checking Unique Number of Customers

# There are quite a few columns in the dataset that signify relating to a unique person/house/business. Checking the unique counts of such columns.

# In[ ]:


check_unique_columns = [
    'Reliant Contract No', 'Service Address ', 'Meter No', 
    'ESID', 'Business Area', 'Cost Center',
]

for col in check_unique_columns:
    print(f'Number of Unique Values in {col}: {data_2013_2[col].nunique()}')


# Based on the above reported values and further research online:
# 
# ESID signifies a unique ID provided to each customer subscribed to the electricity board. It would be best to choose ESID and Service Address columns going forward as these would provide number of unique customers and the areas (streets) where higher usage of electricity occurs.
# 
# Business Area signifies a grouping a number of buildings which covers a certain area. This would be useful usage patterns grouped by certain zones in the city.

# ### Checking Bill Type

# In[ ]:


data_2013_2['Bill Type'].value_counts(dropna=False)


# Bill Type could signify the type of the connection given. Since commercial, residential and government spaces would have different type of pricing and needs this column could be capturing that information.

# In[ ]:


data_2013_2['Service Address '].nunique(), data_2013_2['Meter No'].nunique(), data_2013_2['ESID'].nunique()


# The next 3 columns are: Bill Date, Read Date and Due Date. Of these it would be best to choose the Bill date across all the data files to keep the data consistent. 

# ### Electricity Usage Statistics

# In[ ]:


data_2013_2[['Meter Read', 'Billed Demand (KVA)', 'kWh Usage']].describe()


# There are 3 columns that denote the amount of electricity: Meter Read, Billed Demand, kWh Usage.
# 
# Using kWh Usage as a standard unit of measurement.

# In[ ]:


data_2013_2[[
    'Base Cost ($)', 'T&D Discretionary ($)', 'T&D Charges ($)', 
    'Current Due ($)', 'Total Due ($)', 'Franchise Fee ($)', 
    'Nodal Cu Charge ($)', 'Reliability Unit Charge ($)'
     ]].describe()


# Reliability Unit Charge ($) does not contain any useful information. Electing to drop that column.
# 
# The columns other than Current Due or Total Due are adding up the value present in these two columns. Going forward choosing the column Total Due ($). 
# Based on the above statistics the columns Current Due and Total Due represent the same value. 

# Based on the above analysis of the dataset choosing the following columns:
# 
# 1. ESID
# 2. Business Area
# 3. Service Address 
# 3. Bill Type
# 4. Bill Date
# 5. Total Due ($)
# 6. kWh Usage

# ### Selecting and Filtering Columns

# In[ ]:


data_2013_2 = data_2013_2[[
    'ESID', 'Business Area', 'Service Address ', 'Bill Type',
    'Bill Date', 'Total Due ($)', 'kWh Usage'
]]


# In[ ]:


rename_cols = {
    'ESID': 'esid',
    'Business Area': 'business_area',
    'Service Address ': 'service_address',
    'Bill Type': 'bill_type',
    'Bill Date': 'bill_date',
    'Total Due ($)': 'total_due',
    'kWh Usage': 'kwh_usage'
}

data_2013_2_main = data_2013_2.rename(columns=rename_cols)


# Checking for Nulls again and dtypes

# In[ ]:


data_2013_2_main.isna().sum()


# In[ ]:


data_2013_2_main.dropna(subset=['kwh_usage'], inplace=True)


# In[ ]:


data_2013_2_main.isna().sum()


# In[ ]:


data_2013_2_main.dtypes


# In[ ]:


data_2013_2_main.shape


# In[ ]:


zscore_2013_2 = stats.zscore(data_2013_2_main[['total_due', 'kwh_usage']])

zscore_2013_2


# Each zscore value signifies how many standard deviations away an individual value is from the mean. This is a good indicator to finding outliers in the dataframe.
# 
# Usually z-score=3 is considered as a cut-off value to set the limit. Therefore, any z-score greater than +3 or less than -3 is considered as outlier which is pretty much similar to standard deviation method

# In[ ]:


# data_2013_2_main = data_2013_2_main[(np.abs(zscore_2013_2) < 3).all(axis=1)]

data_2013_2_main.shape


# The number of rows has decreased from 65,805 to 65,388. So 417 rows were outliers based on the data.

# In[ ]:


data_2013_2_main.head(5)


# In[ ]:


orig_shape_2013_2 - data_2013_2_main.shape[0]


# In[ ]:


data_2013_2_main.to_csv('electricity_usage_data_2013_2.csv', index=False)


# ## Data FY 2014 - Sharmisha

# In[ ]:


data_2014 = pd.read_excel(
'houston-houston-electricity-bills/coh-fy2014-ee-bills-july2013-june2014.xlsx'
)
orig_shape_2014 = data_2014.shape[0]

data_2014.shape


# In[ ]:


data_2014.head(5)


# ### Checking Nulls

# In[ ]:


data_2014.isna().sum()


# ### Checking Index Charge ($) column - This was previously Adjustment

# In[ ]:


data_2014['Index Charge ($)'].value_counts(dropna=False)


# The column does does have information regarding a certain price. Since we are using the total due amount at the end, Index Charge ($) does not need to be present again, as it would be included in the total due amount.

# In[ ]:


data_2014.drop(columns=['Index Charge ($)'], inplace=True)


# ### Checking Unique Number of Customers

# There are quite a few columns in the dataset that signify relating to a unique person/house/business. Checking the unique counts of such columns.

# In[ ]:


check_unique_columns = [
    'Reliant Contract No', 'Service Address ', 'Meter No', 
    'ESID', 'Business Area', 'Cost Center'
]

for col in check_unique_columns:
    print(f'Number of Unique Values in {col}: {data_2014[col].nunique()}')


# NOTE: Compared to previous years, there is one less business area.
# 
# Based on the above reported values and further research online:
# 
# ESID signifies a unique ID provided to each customer subscribed to the electricity board. It would be best to choose ESID and Service Address columns going forward as these would provide number of unique customers and the areas (streets) where higher usage of electricity occurs.
# 
# Business Area signifies a grouping a number of buildings which covers a certain area. This would be useful usage patterns grouped by certain zones in the city.

# ### Checking Bill Type

# In[ ]:


data_2014['Bill Type'].value_counts(dropna=False)


# Bill Type could signify the type of the connection given. Since commercial, residential and government spaces would have different type of pricing and needs this column could be capturing that information.
# 
# Previously there were 3 types of Bills. T, P, and C. But in year 2014 there are only 2 types. 

# In[ ]:


(
    data_2014['Service Address '].nunique(), 
    data_2014['Meter No'].nunique(), 
    data_2014['ESID'].nunique()
)


# The next 3 columns are: Bill Date, Read Date and Due Date. Of these it would be best to choose the Bill date across all the data files to keep the data consistent. 

# ### Electricity Usage Statistics

# In[ ]:


data_2014[['Meter Read', 'Billed Demand (KVA)', 'kWh Usage']].describe()


# There are 3 columns that denote the amount of electricity: Meter Read, Billed Demand, kWh Usage.
# 
# Using kWh Usage as a standard unit of measurement.

# In[ ]:


data_2014[[
    'Base Cost ($)', 'T&D Discretionary ($)', 'T&D Charges ($)', 
    'Current Due ($)', 'Total Due ($)', 'Franchise Fee ($)', 
    'Nodal Cu Charge ($)', 'Adder Charge ($)'
     ]].describe()


# Adder Charge ($) does not contain any useful information. Electing to drop that column. Previously this column was Reliability Unit Charge.
# 
# The columns other than Current Due or Total Due are adding up the value present in these two columns. Going forward choosing the column Total Due ($). 
# Based on the above statistics the columns Current Due and Total Due represent the same value. 

# Based on the above analysis of the dataset choosing the following columns:
# 
# 1. ESID
# 2. Business Area
# 3. Service Address 
# 3. Bill Type
# 4. Bill Date
# 5. Total Due ($)
# 6. kWh Usage

# ### Selecting and Filtering Columns

# In[ ]:


data_2014 = data_2014[[
    'ESID', 'Business Area', 'Service Address ', 'Bill Type',
    'Bill Date', 'Total Due ($)', 'kWh Usage'
]]


# In[ ]:


rename_cols = {
    'ESID': 'esid',
    'Business Area': 'business_area',
    'Service Address ': 'service_address',
    'Bill Type': 'bill_type',
    'Bill Date': 'bill_date',
    'Total Due ($)': 'total_due',
    'kWh Usage': 'kwh_usage'
}

data_2014_main = data_2014.rename(columns=rename_cols)


# Checking for Nulls again and dtypes

# In[ ]:


data_2014_main.isna().sum()


# In[ ]:


data_2014_main.dtypes


# In[ ]:


data_2014_main.shape


# In[ ]:


zscore_2014 = stats.zscore(data_2014_main[['total_due', 'kwh_usage']])

zscore_2014


# Each zscore value signifies how many standard deviations away an individual value is from the mean. This is a good indicator to finding outliers in the dataframe.
# 
# Usually z-score=3 is considered as a cut-off value to set the limit. Therefore, any z-score greater than +3 or less than -3 is considered as outlier which is pretty much similar to standard deviation method

# In[ ]:


# data_2014_main = data_2014_main[(np.abs(zscore_2014) < 3).all(axis=1)]

data_2014_main.shape


# The number of rows has decreased from 67,838 to 67,427. So 411 rows were outliers based on the data.

# In[ ]:


data_2014_main.head(5)


# In[ ]:


orig_shape_2014 - data_2014_main.shape[0]


# In[ ]:


data_2014_main.to_csv('electricity_usage_data_2014.csv', index=False)


# ## Merging the DataFrames - Sourabh

# In[ ]:


df_list = [data_2012_main, data_2013_main, data_2013_2_main, data_2014_main]

data = pd.concat(df_list)
data.shape


# Since we cleaned each df separetely there should not be any NANs, but checking for it nonetheless.

# In[ ]:


data.isna().sum()


# ### Checking for Duplicate Rows

# Since there is an overlap in time period for the CSV files it is important not to have repeating rows.
# 
# We can check for this in the following way: A particular ESID should be billed only once. Therefore by taking a subset of ESID, Business Area and Bill Date we can know if a particular customer's billing info has been repeated in the df or not.

# In[ ]:


dup_rows_index = data.duplicated(
    subset=['esid', 'business_area', 'service_address', 'bill_date']
)
(dup_rows_index).sum()


# This confirms the doubt that the overlap with the FY 2012 and FY 2013 with the CSV file has generated duplicate rows in the data.
# We need to remove these columns.

# In[ ]:


data_main = data[~(dup_rows_index)]

data_main.shape


# In[ ]:


data_main.to_csv('Electricity_Usage_Data.csv', index=False)


# ## Visualizations

# In[ ]:


import time
from pprint import pprint


# In[ ]:


data_main = pd.read_csv('Electricity_Usage_Data.csv')


# In[ ]:


data_main[['bill_date']] = data_main[['bill_date']].apply(pd.to_datetime)


# In[ ]:


data_main.loc[:,'bill_date'] = data_main['bill_date'].apply(
    lambda x: pd.to_datetime(f'{x.year}-{x.month}-01')
)


# In[ ]:


viz_df = data_main.set_index('bill_date')


# In[ ]:


viz_df.head()


# ### Visualization #1 - Hyndavi

# In[ ]:


address_enc = LabelEncoder()
bill_type_enc = LabelEncoder()

data_main['address_enc'] = address_enc.fit_transform(
    data_main['service_address']
)
data_main['bill_type_enc'] = bill_type_enc.fit_transform(
    data_main['bill_type']
)
data_main['year'] = data_main['bill_date'].apply(lambda x:x.year)
data_main['month'] = data_main['bill_date'].apply(lambda x:x.month)


# In[ ]:


sns.heatmap(data_main.corr())


# There do not seem to be any features that have high correlation with kwh usage except total due. But this is to be expected since the amount of energy used is directly proportional to the cost.
# 
# It might be difficult to use the features as they are for ML modeling.

# In[74]:


# Pie chart for 'Bill Type'
explode = (0, 0.4, 0.2)
plt.figure(figsize=(5, 4))
print('value_counts:\n', data_main['bill_type'].value_counts())
data_main['bill_type'].value_counts().plot(kind='pie', explode=explode) #, autopct='%1.10f%%')
plt.title('Bill Type Pie Chart')
plt.xlabel('Type of Bill')
plt.ylabel('Frequency')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# In[67]:


# Bar chart for 'Business Area'
plt.figure(figsize=(8, 6))
ax = data_main['business_area'].value_counts().plot(kind='bar')
plt.title('business_area')
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# The business area 2000 is the most populous area based on the frequency plot.
# 
# And the most common type of Bill type is T.

# ### Visualization #2 - Sunil

# In[ ]:


viz_df_2 = data_main.groupby('bill_date').agg(
    {'kwh_usage':'mean'}
).reset_index()

temp_list = [0 for _ in range(12)]

date_dict = {
    '2011': temp_list.copy(),
    '2012': temp_list.copy(),
    '2013': temp_list.copy(),
    '2014': temp_list.copy(),
}

for date, usage in viz_df_2.values:
    date_dict[str(date.year)][date.month-1] = usage

usage_2011 = date_dict['2011']
usage_2012 = date_dict['2012']
usage_2013 = date_dict['2013']
usage_2014 = date_dict['2014']


plt.figure(figsize=(8,6))
plt.plot(range(12), usage_2011, label='2011')
plt.plot(range(12), usage_2012, label='2012')
plt.plot(range(12), usage_2013, label='2013')
plt.plot(range(12), usage_2014, label='2014')
plt.xlabel('Month')
plt.ylabel('kWh Usage')
plt.title('Average Monthly Usage of Electricity')
plt.legend()
plt.show()


# In 2011 the first 6 months and  in 2014 the last 6 months have the value 0. This is due to the fact that the data has been collected for each year from July of the current year to June of the next year (Financial Year).
# 
# Similar to the previous plot, we can see the same trend. But one thing to be observed is that the trend across all these years has remained relatively same across the months even in these 4 years.

# ### Visualization #3 - Sourabh

# The code for the visualizations performed in this section is in the IPYNB: 'Viz_Sourabh.ipynb'

# In[ ]:


def plotbox(df, column):
    plot_features = df.groupby(pd.Grouper(freq=str(60)+'T')).mean().copy()
    plot_features[column] = [eval('x.%s'%column) for x in plot_features.index]
    plot_features.boxplot('kwh_usage', by=column, figsize=(12, 8), grid=False)
    plt.ylabel('kWh Usage')
    plt.xlabel(column)
    plt.show()


# In[ ]:


plotbox(viz_df, 'month')


# Based on the above box plot, we can see that the highest energy is consumed in the months of June-September. That is in the summer/fall season energy usage is quite high.

# In[ ]:


plotbox(viz_df, 'year')


# It was expected that energy consumption would increase through the years. It was surprising to see that the energy use in fact decreased. Further analysis needs to be done in order to see why this might have happened.

# In[ ]:


data_main_agg = data_main.groupby('business_area').agg({
    'esid': pd.Series.nunique,
    'total_due': 'mean',
    'kwh_usage': 'mean',
})

data_main_agg.sort_values(by='kwh_usage', ascending=False)


# In[ ]:


plt.plot(
    data_main_agg.index, 
    data_main_agg['total_due'], 
    color='b', 
    label='Average Cost'
)
plt.plot(
    data_main_agg.index, 
    data_main_agg['kwh_usage'], 
    color='r', 
    label='Average kwh Usage'
)
plt.title('Average Energy Usage by Business Area')
plt.legend()
plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
fig.suptitle('Average Energy Usage by Business Area')

axes[0].plot(
    data_main_agg.index, 
    data_main_agg['total_due'], 
    color='b', 
    label='Average Cost'
)
axes[0].legend()

axes[1].plot(
    data_main_agg.index, 
    data_main_agg['kwh_usage'], 
    color='r', 
    label='Average kwh Usage'
)
axes[1].legend()

fig.tight_layout()

plt.show()


# This indicates that both the cost and kwh usage are indicating usage of electricity and the only difference between is the scale (a multiplicative factor). 
# 
# Therfore, analyzing just the kwh usage or just the price might be enough and both should be used in predictive tasks, i.e., if we are predicting kwh usage then price should not be present in the train dataset, as these two are different representations of the same thing which is usage.

# ### Visualization #4 - Sharmisha

# In[ ]:


monthly_en = viz_df.resample('M', label = 'left')['kwh_usage'].max()
plt.figure(figsize = (12,6))

#plotting the max monthly energy consumption
plt.plot(monthly_en)
plt.xlim(monthly_en.index.min(), monthly_en.index.max())
locator = mdates.MonthLocator(bymonthday = 1, interval = 2)
fmt = mdates.DateFormatter('%m-%y') 
X = plt.gca().xaxis
# Setting the locator
X.set_major_locator(locator)
# Specify formatter
X.set_major_formatter(fmt)
plt.xticks(rotation = 45)
plt.ylabel('Max Energy consumption in kWh')
plt.xlabel('Date')
plt.title('Peak energy usage of a month over the years 2011-14')
plt.show()


# There is a noticable trend in the usage of energy across the months starting from 2011 to 2014.

# In[ ]:


sns.pairplot(data_main)


# In[141]:


# Stacked bar chart to visualize the distribution of bill types across different business areas
business_areas = data_main['business_area'].unique()
if len(business_areas) > 0:
    bill_types = data_main['bill_type'].unique()

    business_area_counts = []
    for ba in business_areas:
        counts = data_main.loc[data_main['business_area'] == ba]['bill_type'].value_counts()
        business_area_counts.append(counts)

    bill_type_counts_by_business_area = pd.concat(business_area_counts, axis=1, keys=business_areas)
    print('bill_type_counts_by_business_area:')

    bill_type_counts_by_business_area.plot(kind='bar', stacked=True)

    # y-axis label
    plt.ylabel('Number of Bills')

    # Chart title
    plt.title('Distribution of Bill Types by Business Area')

    plt.show()    


# ## ML Models

# In[ ]:


data_main = pd.read_csv('Electricity_Usage_Data.csv')


# In[ ]:


data_main[['bill_date']] = data_main[['bill_date']].apply(pd.to_datetime)


# In[ ]:


data_main.loc[:,'bill_date'] = data_main['bill_date'].apply(
    lambda x: pd.to_datetime(f'{x.year}-{x.month}-01')
)


# In[ ]:


address_enc = LabelEncoder()
bill_type_enc = LabelEncoder()

data_main['address_enc'] = address_enc.fit_transform(
    data_main['service_address']
)
data_main['bill_type_enc'] = bill_type_enc.fit_transform(
    data_main['bill_type']
)
data_main['year'] = data_main['bill_date'].apply(lambda x:x.year)
data_main['month'] = data_main['bill_date'].apply(lambda x:x.month)


# ### Regression - Task: Predicting Energy Usage - Sourabh
# 
# Models Proposed:
# 1. Linear Regression - Simple and easy to understand model. Using this to set a baseline. assumes linear relationship between thhe input features and the target variable.
# 2. Gradient Boosting Regressor - Since this model is an ensemble model which combines, multiple decision tree. Expecting good accuracy from this model. can be computationally expensive and may require more resources to other models
# 3. Decision Tree Regressor - This is a simple, interpretable model. can handle non-linear relationships between the input features. Can be prone to overfitting on the train data

# The code for the ML/Stats performed in this section is in the IPYNB: 'Regression_Sourabh.ipynb'

# In[ ]:


data_main.head()

Q1 = data_main['kwh_usage'].quantile(0.25)
Q3 = data_main['kwh_usage'].quantile(0.75)
IQR = Q3 - Q1

Q1, Q3, IQR
data_main_filt = data_main[~(
    (data_main['kwh_usage'] < (Q1 - 1.5 * IQR)) | 
    (data_main['kwh_usage'] > (Q3 + 1.5 * IQR))
)]

data_main.shape, data_main_filt.shape


# Therefore, 33,755 rows have values that are considered outliers based on the IQR Method.

# In[ ]:


def regression_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f'Train R2 Score: {r2_score(y_train, y_train_pred)}')
    print(f'Test R2 Score: {r2_score(y_test, y_test_pred)}')

    print(f'Train MSE Score: {mean_squared_error(y_train, y_train_pred)}')
    print(f'Test MSE Score: {mean_squared_error(y_test, y_test_pred)}')


# In[ ]:


X = data_main[[
    'business_area', 'address_enc', 'bill_type_enc', 'year', 'month'
]]
y = data_main[['kwh_usage']]

X_filt = data_main_filt[[
    'business_area', 'address_enc', 'bill_type_enc', 'year', 'month'
]]
y_filt = data_main_filt[['kwh_usage']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_filt, X_test_filt, y_train_filt, y_test_filt = train_test_split(
    X_filt, y_filt, test_size=0.2, random_state=42
)


# #### Linear Regression

# In[ ]:


reg = LinearRegression().fit(X_train, y_train)
reg_filt = LinearRegression().fit(X_train_filt, y_train_filt)


# In[ ]:


print('With Outliers')
regression_metrics(reg, X_train, X_test, y_train, y_test)

print('='*50)

print('Without Outliers')
regression_metrics(
    reg_filt, X_train_filt, X_test_filt, y_train_filt, y_test_filt
)


# #### Gradient Boosting Regression

# In[ ]:


gbr = GradientBoostingRegressor().fit(X_train, np.ravel(y_train))
gbr_filt = GradientBoostingRegressor().fit(
    X_train_filt, np.ravel(y_train_filt)
)


# In[ ]:


print('With Outliers')
regression_metrics(
    gbr, X_train, X_test, np.ravel(y_train), np.ravel(y_test)
)

print('='*50)

print('Without Outliers')
regression_metrics(
    gbr_filt, X_train_filt, X_test_filt, np.ravel(y_train_filt), np.ravel(y_test_filt)
)


# #### Decision Tree Regressor

# In[ ]:


dtr = DecisionTreeRegressor().fit(X_train, np.ravel(y_train))
dtr_filt = DecisionTreeRegressor().fit(X_train_filt, np.ravel(y_train_filt))


# In[ ]:


print('With Outliers')
regression_metrics(
    dtr, X_train, X_test, np.ravel(y_train), np.ravel(y_test)
)

print('='*50)

print('Without Outliers')
regression_metrics(
    dtr_filt, X_train_filt, X_test_filt, np.ravel(y_train_filt), np.ravel(y_test_filt)
)


# ### Classification - Task: Predicting Type of Bill - Sharmisha
# 
# Models proposed:
# 1. Logistic Regression - widely used interpretable model which can be used for setting a baseline accuracy. This model assumed linear relationship between the variables, so mighht give bad results
# 
# 2. Decision Tree Classifier - It can handle the non-linear relationships well between input and target variable. Can be prone to overfitting on the train data.
# 
# 3. Random Forest Classifier - ensemble model, takes advantage of multiple decision trees to create a powerful model. But this model is not easy to interpret and requires more computational resource to run.

# In[ ]:


def classification_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(
        f'Train F1 Score: {f1_score(y_train, y_train_pred, average="macro")}'
    )
    print(
        f'Test F1 Score: {f1_score(y_test, y_test_pred, average="macro")}'
    )

    print(f'Train Accuracy Score: {accuracy_score(y_train, y_train_pred)}')
    print(f'Test Accuract Score: {accuracy_score(y_test, y_test_pred)}')


# In[ ]:


X = data_main[[
    'business_area', 'address_enc', 'kwh_usage', 'year', 'month'
]]
y = data_main[['bill_type_enc']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# #### Logistic Regression

# In[ ]:


lreg = LogisticRegression().fit(X_train, np.ravel(y_train))


# In[ ]:


classification_metrics(lreg, X_train, X_test, y_train, y_test)


# #### Decision Tree Classifier

# In[ ]:


dtc = DecisionTreeClassifier().fit(X_train, y_train)


# In[ ]:


classification_metrics(dtc, X_train, X_test, y_train, y_test)


# #### Random Forest Classifier

# In[ ]:


rfc = RandomForestClassifier().fit(X_train, np.ravel(y_train))


# In[ ]:


classification_metrics(
    dtc, X_train, X_test, np.ravel(y_train), np.ravel(y_test)
)


# ### Time-Series Analysis - Hyndavi
# 
# Proposed Models:
# 1. VAR - can model multiple time series variables simultaneously and capture complex relationships between multiple time series variables. But this model can be sensitive to the number of lags used in the model.
# 2. ARIMA - Can capture the autocorrelation and trends in the time series data as well as seasonality. But it may not perform well with long term forecasting and requires turning to make it optimal
# 3. LSTM - can model complex relationships between time series data such as non-stationary and non-linear time series data. But it requires a lot of computational resources compared to the other models.
# 
# Currently still working on time-series analysis.

# In[4]:


# Mount drive to colab file
from google.colab import drive
drive.mount('/content/drive')

# Insert, change the directory 
import sys
sys.path.insert(0,'/content/drive/MyDrive/CS418-Project-main')
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/CS418-Project-main')


# In[1]:


# !pip install pmdarima


# Load all the data from the files, which was cleaned and pre-processed by all the team members. 

# In[5]:


data_2012_main = pd.read_csv('electricity_usage_data_2012.csv')
data_2013_main = pd.read_csv('electricity_usage_data_2013.csv')
data_2013_2_main = pd.read_csv('electricity_usage_data_2013_2.csv')
data_2014_main = pd.read_csv('electricity_usage_data_2014.csv')


# In[9]:


# Remove outliers in data
zscore_2012 = stats.zscore(data_2012_main[['total_due', 'kwh_usage']])
print('data_2012_main shape before removing outliers: {}'.format(data_2012_main.shape))
data_2012_main = data_2012_main[(np.abs(zscore_2012) < 3).all(axis=1)]
print('data_2012_main shape after removing outliers: {}'.format(data_2012_main.shape), '\n')

zscore_2013 = stats.zscore(data_2013_main[['total_due', 'kwh_usage']])
print('data_2013_main shape before removing outliers: {}'.format(data_2013_main.shape))
data_2013_main = data_2013_main[(np.abs(zscore_2013) < 3).all(axis=1)]
print('data_2013_main shape after removing outliers: {}'.format(data_2013_main.shape), '\n')

zscore_2013_2 = stats.zscore(data_2013_2_main[['total_due', 'kwh_usage']])
print('data_2013_2_main shape before removing outliers: {}'.format(data_2013_2_main.shape))
data_2013_2_main = data_2013_2_main[(np.abs(zscore_2013_2) < 3).all(axis=1)]
print('data_2013_2_main shape after removing outliers: {}'.format(data_2013_2_main.shape), '\n')

zscore_2014 = stats.zscore(data_2014_main[['total_due', 'kwh_usage']])
print('data_2014_main shape before removing outliers: {}'.format(data_2014_main.shape))
data_2014_main = data_2014_main[(np.abs(zscore_2014) < 3).all(axis=1)]
print('data_2014_main shape after removing outliers: {}'.format(data_2014_main.shape), '\n')


# Verify the data to check nulls, duplicate rows, and save final data into csv file

# In[10]:


df_list = [data_2012_main, data_2013_main, data_2013_2_main, data_2014_main]

data = pd.concat(df_list)
print('data.shape', data.shape, '\n')

# Checking nulls in the data
print('Nulls in the data:\n', data.isna().sum(), '\n')

# Checking for duplicate rows
dup_rows_index = data.duplicated(subset=['esid', 'business_area', 'service_address', 'bill_date'])
print('duplicate rows', (dup_rows_index).sum(), '\n')

# Removing the duplicates
data_main = data[~(dup_rows_index)]
print('data_main.shape', data_main.shape, '\n')
# last result - data_main.shape (190848, 7) 

# saving into csv files
data_main.to_csv('Electricity_Usage_Data.csv', index=False)


# ### Prepocessing data for Model

# In[12]:


plt.rcParams.update({'font.size': 12})

data_df = pd.read_csv('Electricity_Usage_Data.csv')
address_enc = LabelEncoder()
bill_type_enc = LabelEncoder()

data_df['bill_date']=pd.to_datetime(data_df['bill_date'])
data_df['year'] = data_df['bill_date'].apply(lambda x: x.year)
data_df['month'] = data_df['bill_date'].apply(lambda x: x.month)
data_df['year_month'] = data_df['bill_date'].dt.date.apply(lambda x: x.strftime('%Y-%m'))
data_df['week'] = data_df.apply(lambda row: row['bill_date'].week+52*(int(row['year'])-2011),axis=1)

data_df.head()


# In[13]:


df = data_df[['kwh_usage', 'week']]
df = df.groupby(by=['week']).mean()


# In[14]:


plt.figure(figsize=(35, 7))

plt.grid()
plt.plot(df)

plt.title('Average Weekly Consumption of Electricity', fontsize=25)
plt.xlabel('Week', fontsize=20)
plt.ylabel('Kwh Usage', fontsize=20)

plt.tight_layout()


# ### **Check whether the series is stationary?**
# Stationary time series is the one whose satistical properties(mean, var, etc.) donot change over time. \
# 
# We need to perform additional check to find if the series is stationary?
# 
# We'll use rolling statistics first, followed by Dickey-Fuller test to check if the series is stationary and make it stationary if not.

# ### Rolling Statistics Method

# In[15]:


rolling_mean = df.rolling(2).mean()
rolling_std = df.rolling(2).std()


# In[16]:


plt.figure(figsize=(35, 8))
plt.grid()

plt.plot(df, color="blue",label="Original Usage")
plt.plot(rolling_mean, color="red", label="Rolling Mean")
plt.plot(rolling_std, color="black", label = "Rolling Standard Deviation")

plt.title('Electricity Time Series, Rolling Mean, Standard Deviation', fontsize=25)
plt.xlabel('Week', fontsize=20)
plt.ylabel('Energy Usage in Kwh', fontsize=20)

plt.legend(loc="upper left")
plt.tight_layout()


# We see that statistics are not constant over the time, but to confirm we'll perform additional statistical test using augmented Dickey-Fuller method.

# ### Augmented Dickey-Fuller Test:
# H0 = Null-hypothesis => It has unit root, the series is non-stationary \
# H1 = Alternate-hypothesis => No unit root, the series is stationary
# 
# If p-value < critical value [0.05] -> We reject the null-hypothesis H0 \
# If p-value > critical value [0.05] -> We fail to reject null-hypothesis H0

# In[17]:


def aug_dickey_fuller_test(df):
  adft = adfuller(df, autolag="AIC")
  output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']], 
                            "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
  print(output_df)


# In[18]:


aug_dickey_fuller_test(df)


# ### Dickey-Fuller Test Result:
# 
# As (Test Statistics -1.83 > -2.88 critical value (5%)), p-value > 0.05, we fail to reject the null-hypothesis, and thus the time series is non-stationary.

# ### **Make the Series Stationary**

# In[19]:


# First we'll perform differencing on the data to see if it becomes stationary
diff_df = df.diff()

plt.figure(figsize=(32, 8))
plt.grid()
plt.plot(diff_df)

plt.title('Electricity Usage after Differencing', fontsize=25)
plt.xlabel('Week', fontsize=20)
plt.ylabel('Kwh Usage', fontsize=20)

plt.tight_layout()


# In[20]:


# Confirm with the dickey-fuller test
aug_dickey_fuller_test(diff_df.dropna())


# Here, (Test Statistics = -4.55 <  critical value (5%) of -2.88), p-value is < 0.05 so we reject the null hypothesis and accept the alternate hypothesis, hence considers the **time series is stationary** for order difference of 1 (d).

# ### ARIMA - AutoRegressive Integrated Moving Average

# In[22]:


# week wise data split
train_data = df.loc['0':'160']
test_data = df.loc['160':]

plt.figure(figsize=(35, 7))
plt.grid()

plt.plot(train_data, c='blue', label='Train kwh_usage')
plt.plot(test_data, c='orange', label='Test kwh_usage')
plt.legend(loc='upper left', prop={'size':20})

plt.title('Average Weekly Consumption of Electricity for Train, Test data', fontsize=25)
plt.xlabel('Week', fontsize=20)
plt.ylabel('Kwh Usage', fontsize=20)

plt.tight_layout()


# In[24]:


# Find the order of the ARIMA model
order_df = auto_arima(df, trace=True, suppress_warnings=True)
order_df.summary()


# In[64]:


# With the optimised (p, d, q) based on the above auto_arima results, and as we also know (d=1) which is identified while making the series stationary
model = ARIMA(train_data, order=(3, 1, 3)).fit()

# Prediction
pred = model.predict(start=len(train_data)-1,end=(len(df)-1))

# Model Summary
model.summary()


# In[65]:


# Evaluation
print('Mean Absolute Error: %.2f' % mean_absolute_error(test_data['kwh_usage'].values, pred))
print('Root Mean Squared Error: %.2f' % np.sqrt(mean_squared_error(test_data['kwh_usage'], pred)))


# In[26]:


pred = pd.Series(list(pred.values), index=list(test_data.index))


# In[27]:


plt.figure(figsize=(35, 8))
plt.grid()

plt.plot(train_data, label = 'Train kwh_usage')
plt.plot(test_data, label = 'Test kwh_usage')
plt.plot(pred, label = 'Predicted kwh_usage')

plt.title('Weekly Electricity Usage Forecasting', fontsize=25)
plt.xlabel('Week', fontsize=20)
plt.ylabel('Average Energy Usage Consumption in Kwh', fontsize=20)
plt.legend(loc='upper left', prop={'size': 20})
plt.tight_layout()


# ### VAR - Vector AutoRegressive

# In[29]:


# week wise data
var_df = data_df[['week', 'total_due', 'kwh_usage']]
group_df = var_df.groupby(['week']).mean()


# In[30]:


fig, axes = plt.subplots(nrows=2, ncols=1, dpi=120, figsize=(20,6))
for i, ax in enumerate(axes.flatten()):
    # data = var_df['bill_date', var_df.columns[i]].groupby('bill_date').mean()
    data = group_df[group_df.columns[i]]
    ax.plot(data, color='blue', linewidth=1)
    
    # Decorations
    ax.set_title(group_df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[31]:


print('kwh_usage augmented dickey-fuller test:')
aug_dickey_fuller_test(group_df['kwh_usage'])
print('\n')
print('total_due augmented dickey-fuller test:')
aug_dickey_fuller_test(group_df['total_due'])


# In[34]:


# Perform one order differecing and see if it becomes stationary and confirm with the dickey-fuller test
print('kwh_usage adf test on diff data:')
aug_dickey_fuller_test(group_df['kwh_usage'].diff().dropna())
print('\n')
print('total_due adf test on diff data:')
aug_dickey_fuller_test(group_df['total_due'].diff().dropna())


# One order differencing of kwh_usage, total_due made the series stationary

# In[35]:


print('kwh_usage causes total_due?\n')
print('------------------')
granger_1 = grangercausalitytests(group_df[['kwh_usage', 'total_due']], 4)

print('\n total_due causes kwh_usage?\n')
print('------------------')
granger_2 = grangercausalitytests(group_df[['total_due', 'kwh_usage']], 4)


# In[37]:


# week wise data split
train_data = group_df.loc['0':'160']
test_data = group_df.loc['160':]


# In[38]:


model = VAR(train_data)

order = model.select_order(maxlags=20)
print(order.summary())


# Based on the above results, the minimum vaues of AIC, FPE are observed at lag-13, so the lag to be choosen is 13.

# In[39]:


# As VARMAX takes care of stationarity using the property enforce_stationarity, we don't need to explicity handle it
var_model = VARMAX(train_data, order=(13, 0), enforce_stationarity= True)
fitted_model = var_model.fit(disp=False)
print(fitted_model.summary())


# In[40]:


pred = fitted_model.get_prediction(start=len(train_data)-1,end=len(group_df)-1)
predictions = pred.predicted_mean

predictions.columns=['total_due', 'kwh_usage']
predictions.index = test_data.index


# In[43]:


plt.figure(figsize=(20, 6))
plt.plot(train_data['kwh_usage'], label = 'Train kwh_usage')
plt.plot(train_data['total_due'], label = 'Train total_due')
plt.plot(test_data['kwh_usage'], label = 'Test kwh_usage')
plt.plot(test_data['total_due'], label = 'Test total_due')
plt.plot(predictions['kwh_usage'], label = 'Predicted kwh_usage')
plt.plot(predictions['total_due'], label = 'Predicted total_due')

plt.legend(loc='upper left')
plt.title('Weekly Electricity Usage Forecasting', fontsize=20)
plt.xlabel('Week', fontsize=12)
plt.ylabel('Average Energy Usage Consumption in Kwh', fontsize=12)
plt.tight_layout()


# In[44]:


# Calculating the root mean squared error
rmse_kwh_usage = math.sqrt(mean_squared_error(predictions['kwh_usage'],test_data['kwh_usage']))
print('Mean value of kwh_usage is : {}. Root Mean Squared Error is :{}'.format(mean(test_data['kwh_usage']), rmse_kwh_usage))

rmse_total_due = math.sqrt(mean_squared_error(predictions['total_due'],test_data['total_due']))
print('Mean value of total_due is : {}. Root Mean Squared Error is :{}'.format(mean(test_data['total_due']), rmse_total_due))


# ### LSTM - Long Short Term Memory

# In[46]:


lstm_df = data_df[['kwh_usage', 'week']]
lstm_df = lstm_df.groupby('week').mean()
lstm_df.shape, lstm_df.head()


# In[48]:


res = seasonal_decompose(lstm_df['kwh_usage'].dropna(), period=1)
fig = res.plot()
fig.set_size_inches((11, 7))
fig.tight_layout()
plt.show()


# In[49]:


train_data = lstm_df.loc['0':'160']
test_data = lstm_df.loc['160':]


# In[51]:


lstm_df.head(2), lstm_df.tail(2)


# In[52]:


scaler = MinMaxScaler()

scaler.fit(train_data)
scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)


# In[53]:


# define generator - using past 15 values
n_input = 15
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
     
X,y = generator[0]
print(f'Given Array: \n{X.flatten()}')
print(f'Need to Predict (y): \n {y}')


# In[55]:


# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[56]:


# fit model
model.fit(generator, epochs=50)


# In[57]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

plt.title('Plot of LSTM Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')


# In[58]:


test_preds = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):
    # get pred value for 1st batch
    current_pred = model.predict(current_batch)[0]
    # add preds into test_predictions[]
    test_preds.append(current_pred) 
    # use pred to update the batch and remove 1st value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[60]:


true_predictions = scaler.inverse_transform(test_preds)
test_data['predictions'] = true_predictions


# In[61]:


plt.figure(figsize=(20, 6))
plt.plot(train_data, label='Train kwh_usage')
plt.plot(test_data['kwh_usage'], label='Test kwh_usage')
plt.plot(test_data['predictions'], label='Predicted kwh_usage')

plt.legend(loc='upper left')
plt.title('Weekly Average Electricity Consumption Forecast')
plt.xlabel('Week')
plt.ylabel('Average Consumption Usage in Kwh')
plt.show()


# In[63]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test_data['kwh_usage'],test_data['predictions']))
print('Root Mean Squared Error of kwh_usage is :{}'.format(rmse))


# ### Clustering - Sunil
# 
# Models proposed:
# 
# 1. KMeans - simple and easy model which is computationally efficient. requires the number of clusters to be specified and can be sensitive to initialization of centroids
# 2. DBSCAN - can handle arbitrary shapes of clusters and noise points. But this model can be quite computationally expensive for big datasets.
# 3. Hierarchical - can provide hierarchy of clusters which is useful fpr understanding relationships between the clusters. But this model too is computationally expensive.

# In[ ]:


data_main = pd.read_csv('Electricity_Usage_Data.csv')


# In[ ]:


data_main[['bill_date']] = data_main[['bill_date']].apply(pd.to_datetime)


# In[ ]:


data_main.loc[:,'bill_date'] = data_main['bill_date'].apply(lambda x: pd.to_datetime(f'{x.year}-{x.month}-01'))


# In[ ]:


address_enc = LabelEncoder()
bill_type_enc = LabelEncoder()

data_main['address_enc'] = address_enc.fit_transform(data_main['service_address'])
data_main['bill_type_enc'] = bill_type_enc.fit_transform(data_main['bill_type'])
data_main['year'] = data_main['bill_date'].apply(lambda x:x.year)
data_main["week"] = data_main.apply(lambda row: row["bill_date"].week+52*(row["year"]-2011) ,axis=1)
data_main['month'] = data_main['bill_date'].apply(lambda x:x.month)


# In[ ]:


## Loading the extracted geolocations from saved pickle file
if os.path.isfile("locations.pkl"):
    locations = pickle.load(open("locations.pkl","rb"))
    print("Total Geo Location data extracted by Addresses : ",len(locations.keys()))
else:
    print("Locations not founds, run Extract Geolocations.ipynb")


# In[ ]:


data_main["lat"] = data_main["service_address"].apply(lambda x : locations[x][0] \ 
                                                      if x in locations.keys() else locations["Houston"][0])
data_main["lon"] = data_main["service_address"].apply(lambda x : locations[x][1]  \
                                                      if x in locations.keys() else locations["Houston"][1])


# In[ ]:


## Remove the rows for which the geo locations are not extraceted, which was assigned to houston city location
data_main = data_main[(data_main["lat"]!='29.7589382')&(data_main["lon"]!='-95.3676974')]


# In[ ]:


X = data_main[[
    'business_area', 
    'address_enc', 
    'bill_type_enc', 
    'year', 
    'month',
    'week',
    'kwh_usage',
]]


# In[ ]:


X.dtypes


# In[ ]:


def calculate_WSS(points, kmax):
    sse = []
    for k in range(10, kmax+1,10):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
    
        sse.append(kmeans.inertia_)
        print("sse for kmeans with {} clusters is : {}".format(k,kmeans.inertia_))
    return sse


# In[ ]:


sse = calculate_WSS(X.values,100)


# In[ ]:


plt.plot([i for i in range(10,101,10)],sse)
plt.xlabel("Number of clusters")
plt.ylabel("Sum Squared Error")
plt.show()


# ### From the above plot of sse vs #clusters, by elbow method we can take 40 as the best number of clusters

# In[ ]:


## Training the kmeans on best k-value
kmeans = KMeans(n_clusters=40, random_state=0).fit(X)
print("Value Counts of Cluster obtained from kmeans : ",pd.Series(kmeans.labels_).value_counts())


# In[ ]:


data_main["Cluster"] = kmeans.labels_


# In[ ]:


data_main["lat"].value_counts()


# In[ ]:


fig = px.scatter_mapbox(data_main, lat="lat", lon="lon", hover_name="service_address", \
                        hover_data=["service_address","Cluster","kwh_usage"],color="Cluster", \
                        color_discrete_sequence=["fuchsia"], zoom=3, height=300, \
                        center=dict(lat=29.7, lon=-95.3),)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### The above plot might not able to load because of memory, so a screenshot is provided in the below cells.

# In[ ]:


display.Image("Kmeans2.png")


# ## DBSCAN

# In[ ]:


db = DBSCAN(eps=0.3, min_samples=3)
db.fit(X)

y_pred = db.fit_predict(X)
print("Number of Clusters Obtained from DBSCAN : ",len(set(y_pred)))


# In[ ]:


data_main["DBS Cluster"] = y_pred


# In[ ]:


fig = px.scatter_mapbox(data_main, lat="lat", lon="lon", hover_name="service_address", \
                        hover_data=["service_address","DBS Cluster","kwh_usage"], \
                        color="DBS Cluster",color_discrete_sequence=["fuchsia"], \
                        zoom=3, height=300,center=dict(lat=29.7, lon=-95.3),)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:


display.Image("DBSCAN.png")


# ## Agglomerative Clustering

# In[ ]:


pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']


# In[ ]:


X_principal.shape


# In[ ]:


plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal.iloc[:300], method ='ward')))


# - As Agglomerative clustering is highly computational, we have limited the data to 10000 points for this model.
# - From the Dendrogram, we can take best number of clusters for the model as 2.

# In[ ]:


## Reducing number of rows as agglomerative clustering is unable to run on large data
num_points = 10000
clustering = AgglomerativeClustering(n_clusters=2).fit(X.iloc[:num_points])


# In[ ]:


data_aggc = data_main.iloc[:num_points]
data_aggc["Aggc Cluster"] = clustering.labels_


# In[ ]:


fig = px.scatter_mapbox(data_aggc, lat="lat", lon="lon", hover_name="service_address", \
                        hover_data=["service_address","Aggc Cluster","kwh_usage"], \
                        color="Aggc Cluster", color_discrete_sequence=["fuchsia"], \
                        zoom=3, height=300,center=dict(lat=29.7, lon=-95.3),)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:


display.Image("Aggc.png")


# ## Results
# 
# 1. The hardest part of the project so far was the initial data cleaning process and making sense of the columns. So far the correlation between the features is still quite low so generating useful insights from these features is proving challenging. 
# 
# 2. During Summer/Fall months there is quite high usage of energy. As well as there are few business areas where there is more usage of energy. This has been visualized as well.
# 
# 3. The results so far are preliminary. We need to iterate further on these results and the observations in order to verify the conclusions befere we can confidently say that our observations are indeed correct.
# 
# 4. The data might prove insufficient for few of the ML tasks that we are planning on performing. Therefore finding another data source which can be combined with these datasets is going to prove difficult in case the need arises.
# 
# 5. Yes, we believe we are on track with the project. We have a plan to work on the models that have not yet been implemented and we are going to verify the work we have done so far before concluding the work.
# 
# 6. The data as it stands right now is good as it has been useful to give the insights we have given so far. Once we are able to figure out the time-series and Clustering analysis part as well we would be able to prove that the data is able to forecast as well and cluster high energy needing areas into groups.
# 
# 
# ## Data Science Questions:
# ### Previous electricity usage data can be used for predicting the usage for future (Time-Series) - Hyndavi
# Perfomed time series analysis using three models - ARIMA, VAR, and LSTM.
# * As we have data spread accross 4 financial years, the monthly average energy usage forecasting was poor due to the less amount of data observations, so we had to perform analysis week wise to make the model more robust in predictions. 
# * There was no seasonality observed in the data, however we have seen a trend on average usage of electricity. Trained the models with first 160 weeks and rest for test the data to evaluate the performance.
# * Based on the metrics, we can say that ARIMA performed better. However, if we have more than one correlated variables then it wouldn't be a good choice as it is a univariate analyser.
# * The VAR with the help of VARMAX class was able to enforce the stationary and LSTM model is capable of handling non-stationary data.
# * VAR and LSTM provided similar results in terms on error metrics. The results would have been more concrete if we have more amount of data and few other correlated attributes.
# * Hence, using time series analysis we are able to predict the usage for future and based on previous consumption usage data.
# 
# 
# ### Group areas based on their energy consumption (Clustering) - Sunil
# 
# Out of all the three clustering models, kmeans performed better. It has clearly differentiated between the clusters and was able to give some conclusions out of geo plotted clusters
# 
# #### Kmeans
# * The locations with high usage of electricity are located at the center of the city (blue color) and also some have high usage (blue color) far away from the city center this could * *  probably because most industries/factoris are located outside the city which consume more electricity.
# *  The sub-urban areas have moderate usage(yellow color) of electricity.
# * Very few points have grouped into other than blue or yellow cluster, by viewing each of this points, we have noticed that they are outliers with unsual bill rates and usage.
# 
# #### DBSCAN
# * As DBSCAN is very sensitive to train and our data is not well suited for the clustering, DBSCAN algorith failed to cluster the data.
# * It gave two clusters, one of the clusters have very few points<0.2% which are found to be outliers.
# 
# #### Agglomerative Clustering
# * The Hierarchial model used gave two clusters, with one of clusters beign completely outliers.
# 
# 
# ### Electricity usage can be predicted by using correlated features (Regression) - Sourabh
# * Used three regression techniques: Linear Regression, Gradient Boosting Regression, and Decision Tree Regression.
# * The linear regression was too simple and was not able to capture the correlations within the dataset hence the poor performance on the train as well as test set. This can also be seen in the high RMSE value too.
# * Gradient Boosting Regression was able to capture the relationships between the variables but unfortunately it is overfitting as reflected by the R2 score and RMSE value. On the test the R2 score is too low and RMSE is high which indicates overfitting.
# * The Decision Tree Regressor gave the best performance of the 3 models. It averages 95% on the train and 90% on the test set (taking into account both the model trained on outlier datset and without outliers).
# * Therefore, using regression analysis, we are able to show that electricity usage can be predicted provided we have the features such as area of the connection, the type of the bill (connection taken), the year and month in which the usage is being predicted.
# 
# 
# 
# ### Classification of bill type can be done using features in the data (Classification) - Sharmisha
# * Evaluated the performance of three different classifiers: logistic regression, decision tree classifier, and random forest classifier. The evaluation metrics used are F1 score.
# * Based on the results, we can see that the logistic regression model has lower F1 scores compared to the other two models. This suggests that the logistic regression model may not be performing as well as the decision tree and random forest models in correctly predicting both positive and negative instances.
# * The decision tree and random forest models have very high F1 scores and accuracy scores for both the train and test sets. 
# * For predicting the type of bill, the Decision Tree Classifier and Random Forest Classifier models both performed similarly with an F1-score of 66%. The Logistic Regression model had a lower F1-score of 43%.
# 
# 
# ### Requirements:
# Since we are a team of 4 graduate students we had to provide 12 (3N) deliverables. They are:
# 
# * 4 Machine Learning Methods (Regression, Clustering, Classification, Time Series Analysis) - In each of these we implemented 3 models.
# * 9 Unique Visualizations. They are presented in this notebook above in their respective sections. They are:
# 1. Line Chart
# 2. Bar Chart
# 3. Pie Chart
# 4. Heat Map
# 5. Box Plot
# 6. Pairwise Plot
# 7. Seasonal Decompose Plot
# 8. Geographical plot to visualize clusters
# 9. Hierarchical clustering dendrogram
# 
# ## References
# 1. [sklearn](https://scikit-learn.org/stable/)
# 2. [pandas](https://pandas.pydata.org/docs/)
# 3. [scipy](https://docs.scipy.org/doc/scipy/)
# 4. [data](https://data.world/houston/houston-electricity-bills)
