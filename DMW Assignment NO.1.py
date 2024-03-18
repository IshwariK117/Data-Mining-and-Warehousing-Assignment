#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Heart.csv")

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())
print("________________________")



#  a) Count missing values in each column
print("Count of missing values in each column:")
print(data.isnull().sum())

print("________________________")

# b) Remove inconsistency (if any) in the dataset
# Since we don't have information on specific inconsistencies, let's assume there are none for this example.

    
# d) Draw histogram for any two suitable attributes
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Chol'], bins=20, kde=True, color='salmon')
plt.title('Histogram of Cholesterol')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

print("________________________")


#  e)Display the data types of each column
print("Data types of each column:")
print(data.dtypes)
print("________________________")


# f)Fill missing values with 0
df2 = data.fillna(value=0)
print("Dataset after filling missing values with 0:")
print(df2)
print("________________________")



# g) Find Mean age of patients considering above dataset
mean_age = data['Age'].mean()
print("\nMean age of patients:", mean_age)

print("________________________")

    
# h) Display the shape of the dataset (number of rows and columns)
print("Shape of the dataset:")
print(data.shape)
print("________________________")


# Detect missing values
df1 = data.isnull()
print("DataFrame indicating missing values:")
print(df1)

print("________________________")

# Count total missing values in the dataset
total_sum = df1.sum().sum()
print("Total count of missing values in the dataset:")
print(total_sum)
print("________________________")


# Fill missing values with forward fill
df3 = data.fillna(method='pad')
print("Dataset after forward fill:")
print(df3)

print("________________________")

# Fill missing values with backward fill
df4 = data.fillna(method='bfill')
print("Dataset after backward fill:")
print(df4)
print(data)
print("________________________")

# Fill missing values with forward fill along columns
df5 = data.fillna(method='pad', axis=1)
print("Dataset after forward fill along columns:")
print(df5)
print("________________________")

# Fill missing values with the mean of the 'ca' column
df6 = data.fillna(value=data['Ca'].mean())
print("Dataset after filling missing values with the mean of 'ca' column:")
print(df6)

print("________________________")

# Fill missing values with the minimum value of the 'ca' column
df7 = data.fillna(value=data['Ca'].min())
print("Dataset after filling missing values with the minimum value of 'ca' column:")
print(df7)





# In[ ]:




