#!/usr/bin/env python
# coding: utf-8

# ## Mission 4 - Predicting House Price with Multiple Linear Regression
# 
# New notebook

# ## Mission 4 - Predicting House Price with Multiple Linear Regression
# 
# Introduction
# In this mission, you will be working with a real estate dataset to perform Exploratory Data Analysis (EDA) and create a Multiple Linear Regression (MLR) model to predict the Price of a house based on various features such as Square Feet, Number of Bedrooms, Number of Bathrooms, and other factors.
# 
# The dataset includes the following columns:
# 
# **Square_Feet:** The total area of the house in square feet.
# 
# **Num_Bedrooms:** The number of bedrooms in the house.
# 
# **Num_Bathrooms:** The number of bathrooms in the house.
# 
# **Num_Floors:** The number of floors in the house.
# 
# **Garage_Size:** The size of the garage.
# 
# **Location_Score:** A score representing the desirability of the location.
# 
# **Distance_to_Center:** The distance (in km) to the city center.
# 
# **Price:** The price of the house (target variable).
# 

# ## Step 1: Loading and Understanding the Data
# 
# Start by loading the dataset into a Pandas DataFrame. Inspect the first few rows of the dataset and understand its structure.
# 
# Tasks:
# 
# Task 1.1. The data is stored in Azure Blob Storage. To connect to Azure Blob Storage, we will use a connection string that contains authentication details such as the account name and account key. See the credentials below.
# 
# 
# Task 1.2. Load the dataset into a Pandas DataFrame from the CSV file stored in Blob Storage. Name the dataframe as df_house_price
# 
# Task 1.3. Display the first few rows.
# 
# Task 1.4. Check the data types and null values.

# Assigned tasks: "Task 1, Task 2 - Samuel"         "Task 3, Task 4 - Manoj"          "Task 5 - Nancy"

# In[1]:


import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import StringIO

connect_str = "DefaultEndpointsProtocol=https;AccountName=level5data;AccountKey=cWgxWhPELQdrW0sIgA/eXM8/k9+xdaELukN8sg5imfhm4wVIJQFmypEph/Ppy500v41mYT+iCBbC+AStsfHbAQ==;EndpointSuffix=core.windows.net"
container_name = "businesscases"
blob_name = "house_sales_prediction.csv"

# Create a BlobServiceClient using the connection string
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Get the container client
container_client = blob_service_client.get_container_client(container_name)

# Get the blob client for the specific file
blob_client = container_client.get_blob_client(blob_name)

# Download the blob as a string
blob_data = blob_client.download_blob()
csv_content = blob_data.readall().decode('utf-8')

# Convert the CSV string to a pandas DataFrame
df_house_price = pd.read_csv(StringIO(csv_content))

# Display the first few rows
display(df_house_price.head())


# In[2]:


df_house_price.dtypes


# In[3]:


df_house_price.isnull().sum()


# ## Step 2: Exploratory Data Analysis (EDA)
# 
# In this step, you will perform some basic EDA to understand the relationships between the features and the target variable (Price). This includes visualising the data, checking for correlations, and identifying patterns.
# 
# Tasks:
# 
# Task 2.1. Check summary statistics for the dataset.
# 
# Task 2.2. Visualise the relationship between Square Feet and Price using a scatter plot.
# 
# Task 2.3.  Visualise the distribution of the Price variable using a histogram.
# 
# Task 2.4.  Compute and visualize the correlation matrix between the features.
# 
# Task 2.5. Write 1–2 sentences explaining the insights from each visualisation.

# In[4]:


df_house_price.describe()


# There are 500 houses in the dataset with an average house size of 174.64 square feet. The average price is $582210. There is considerable variation in prices of house ranging from $276892 ti $960678.

# 

# In[5]:


get_ipython().system('pip install seaborn matplotlib')

import seaborn as sns
import matplotlib.pyplot as plt

#scatterplot

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Square_Feet', y='Price', hue='Num_Bedrooms', data=df_house_price)
plt.title(' Relationship between Square Feet and Price')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.show()


# This shows our most frequent price range is between $500,000-$60,000 and that the most common price is $500,000 - $550,000. There is a positive relationship  between houses size and price,with larger houses generally having higher prices. At similar sizes, houses with more bedrooms also tend to have higher price, suggesting that both size and bedroom count may be useful predictors.

# In[6]:


plt.figure(figsize=(8, 6))
sns.histplot(df_house_price['Price'], bins=16, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Distribution')
plt.show()


# Hpuse prices have an approximately bell-shaped distribution, with most prices concentrated between $450,000-$750,000.This shows our most frequent price range is between $500,000-$60,000 and that the most common price is $500,000 - $550,000,while relatively few houses have very low or very high prices. 

# In[7]:


numeric_df = df_house_price.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

plt.title('Correlation Matrix')
plt.show()


# The correlation matrix shows that house size and number of bedrooms have the strongest positive relatonships with price. The remaining features have weaker individual linear relationships with price, and the predictors generally show weak correlations with each other. 
# 

# ## Step 3: Creating a Multiple Linear Regression (MLR) Model
# 
# In this step, we will use Multiple Linear Regression to predict the Price of the house based on the features. We will split the data into a training set and a testing set, fit the model, and evaluate its performance.
# 
# Task 3.1: Split the Data into Features and Target
# Split the dataset into features (X) and target variable (y).
# 
# Task 3.2: Split the Data into Training and Testing Sets
# Split the data into training (80%) and testing (20%) sets.
# 
# Task 3.3: Train the Multiple Linear Regression Model
# Train the Multiple Linear Regression (MLR) model using the training data.
# 
# Task 3.4: Visualise the Predicted vs Actual Prices
# Create a scatter plot comparing the actual prices and predicted prices for the test set.
# 
# Task 3.5: Evaluate the Model
# Evaluate the model using Root Mean Squared Error (RMSE) and interpret the results.
# 
# Task 3.6. Write 1–2 sentences explaining the insights from the visualisation.

# In[8]:


## Manoj Piyasena ##
df = df_house_price

display("DataFrame successfully connected!")
display(df.shape)
display(df.head())
from sklearn.linear_model import LinearRegression

#Task 3.1: Split the Data into Features and Target Split the dataset into features (X) and target variable (y).
X = df[['Square_Feet','Num_Bedrooms','Num_Bathrooms','Num_Floors','Garage_Size','Location_Score','Distance_to_Center']]
y = df['Price']

display("Features and Target successfully separated!")
display("X shape:", X.shape)
display("y shape:", y.shape)

# Task 3.2: Split the Data into Training and Testing Sets Split the data into training (80%) and testing (20%) sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

display(f"Training Data: {X_train.shape}")
display(f"Testing Data: {X_test.shape}")


# Task 3.3: Train the Multiple Linear Regression Model Train the Multiple Linear Regression (MLR) model using the training data.
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

display("Multiple Linear Regression model trained successfully!")


# Task 3.4: Visualise the Predicted vs Actual Prices Create a scatter plot comparing the actual prices and predicted prices for the test set.
results_df = pd.DataFrame({'Actual Price': y_test,'Predicted Price (MLR)': y_pred_lr})
display(results_df)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.title('Actual vs Predicted House Prices (MLR)')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')

plt.grid(True)
plt.show()

# Task 3.5: Evaluate the Model Evaluate the model using Root Mean Squared Error (RMSE) and interpret the results.
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

display(f"Root Mean Squared Error (RMSE) for MLR: {rmse_lr:.2f}")


## Task 3.6. Write 1–2 sentences explaining the insights from the visualisation.

## 1. The graph shows the model generally understands the relationship between houes features and their prices, 
## as most predictions follow the upward trend.
## 2. However, I noticed that the model sometimes overestimates lower- price properties and understimates some higher- priced proparties,
## showing that its predictions are not equally accurate accross all price ranges. 


# ## Step 4: Compare with Random Forest Regressor
# 
# Task 4.1: Train a Random Forest Model
# Train a Random Forest Regressor model and evaluate its performance.
# 
# Task 4.2: Visualise Actual vs Predicted Prices for Random Forest
# Create a scatter plot comparing the actual prices and predicted prices for the Random Forest model.
# 
# Task 4.3: Model Comparison
# Compare the RMSE values of both the MLR and Random Forest models. Which model performs better?

# In[9]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

## Task 4.1:  Train a Random Forest Regressor model and evaluate its performance.

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("Random Forest Model trained successfully!")

#Evaluating model performance by calculating RMSE
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Root Mean Squared Error (RMSE) for Random Forest: {rmse_rf:.2f}")


# The Random Forest model has RMSE $71733.36. A lower RMSE would indicate more accurate predictions, and this measure gives greater importance to large prediction errors.
# 

# 

# In[10]:


## Task 4.2 : Visualise Actual vs Predicted prices from random Forest Create a scatter plot 
## comparing the actual prices and predicted prices for the Random Forest model.

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.title('Actual vs Predicted House Prices (Random Forest)')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')

plt.grid(True)
plt.show()

results_df_rf = pd.DataFrame({'Actual Price': y_test,'Predicted Price (MLR)': y_pred_lr,'Predicted Price (Random Forest)': y_pred_rf})
print(results_df_rf.head())

# Calculate RMSE for Random Forest
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Root Mean Squared Error (RMSE) for Random Forest: {rmse_rf:.2f}")


# The predictions genrally follow the actual price trend. However, the model tends to overestimate lower priced houses and underestimate some higher-priced houses.

# In[10]:


## Task 4.3: Model Comparison Compare the RMSE values of both the MLR and Random Forest models. Which model performs better?

comparison_df = pd.DataFrame({'Model': ['Multiple Linear Regression', 'Random Forest'],'RMSE': [rmse_lr, rmse_rf]})
display(comparison_df)







# Multiple linear regression achieved a lower RMSE of 63952.38 than Random Forest which achieves a RMSE of 71733.35.Therefore, Multiple linear regression performed  better on the same test set. As Multiple linear regression model has produced smaller prediction errors overall, so it is the preferred model.

# ## Step 5: Conclusion and Insights
# 
# Task 5.1: Insights and Recommendations
# 
# Based on the RMSE and visualisations, summarise which model performs better and why.
# 
# Discuss any improvements that could be made to both models.

# In[13]:


if rmse_lr < rmse_rf:
    print("Multiple Linear Regression performs better because it has a lower RMSE.")   
elif rmse_rf < rmse_lr:
    print("Random Forest performs better because it has a lower RMSE.")
else:
    print("Both models have the same RMSE.")


# 

# The exploratory analysis showed that house size and number of bedrooms had the strongest positive correlations with price. House prices were approximately bell-shaped, with most of the prices between $450,000 and $750,000.
# 
# Multiple Linear Regression performed better on the test set, with an RMSE of $63,952.38, while Random Forest has $71,733.36. Both models captured the overall price trend, but tended to overestimate some lower-priced houses and underestimate some higher-priced houses. Based on these results, Multiple Linear Regression is the preferred model for the dataset.
# 
# Both models could be improved by collecting more data and adding features such as property age and condition. We could also try a different training and testing split, such as 70-30 instead of 80-20, to compare performance. For Random Forest, changing the number of trees and their maximum depth could also help improve predictions.
# 
# 
# 

# Good Work

# In[ ]:




