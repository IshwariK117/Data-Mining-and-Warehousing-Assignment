#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
heart_data = pd.read_csv("C:\\Users\\sai\\Desktop\\heart.csv")

# Exclude non-numeric columns from calculation
numeric_data = heart_data.select_dtypes(include=['int64', 'float64'])

# Calculate standard deviation and variance
std_dev = numeric_data.std()
variance = numeric_data.var()

print("Standard Deviation:")
print(std_dev)
print("\nVariance:")
print(variance)

print("-----------------------------------------------------------------------")


# b) Find covariance and correlation coefficient
covariance_matrix = numeric_data.cov()
correlation_matrix = numeric_data.corr()
print("\nCovariance Matrix:\n", covariance_matrix)
print("\nCorrelation Matrix:\n", correlation_matrix)



print("-----------------------------------------------------------------------")

# c) Number of independent features
num_independent_features = len(numeric_data.columns) - 1  # Exclude target variable
print("\nNumber of independent features:", num_independent_features)


print("-----------------------------------------------------------------------")

# d) Identify unwanted features (For illustration purpose, let's say features with low variance are unwanted)
unwanted_features = variance[variance < 0.1].index.tolist()
print("\nUnwanted features:", unwanted_features)


print("-----------------------------------------------------------------------")

# e) Data discretization using equi-frequency binning on age attribute
heart_data['Age_bin'] = pd.qcut(heart_data['Age'], q=5, labels=False)
print(heart_data.head())




print("-----------------------------------------------------------------------")

# f) Normalize RestBP, chol, and MaxHR attributes using different normalization techniques
attributes_to_normalize = ['RestBP', 'Chol', 'MaxHR']


# Min-Max Normalization
min_max_scaler = MinMaxScaler()
data_min_max_normalized = heart_data.copy()
data_min_max_normalized[attributes_to_normalize] = min_max_scaler.fit_transform(heart_data[attributes_to_normalize])
print(data_min_max_normalized[attributes_to_normalize])

print("-----------------------------------------------------------------------")




# Z-score Normalization
z_score_scaler = StandardScaler()
data_z_score_normalized = heart_data.copy()
data_z_score_normalized[attributes_to_normalize] = z_score_scaler.fit_transform(heart_data[attributes_to_normalize])
print(data_z_score_normalized[attributes_to_normalize] )
print("-----------------------------------------------------------------------")

# Decimal Scaling Normalization
data_decimal_normalized = heart_data.copy()
for attribute in attributes_to_normalize:
    scale = 10 ** (len(str(int(heart_data[attribute].max()))) - 1)
    data_decimal_normalized[attribute] = heart_data[attribute] / scale
print( data_decimal_normalized[attribute])




# In[ ]:





# In[ ]:




