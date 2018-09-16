
# coding: utf-8

# # House Prices Prediction
# 
# I have a dataset containing some real estate data that I ca use to predict the estimate price of houses across United States. Although the data is not real but it is a great exercise to test my supervised learning and modelling skills.
# 
# The data contains the following columns:
# 
# * 'Avg. Area Income': Avg. Income of residents of the city house is located in.
# * 'Avg. Area House Age': Avg Age of Houses in same city
# * 'Avg. Area Number of Rooms': Avg Number of Rooms for Houses in same city
# * 'Avg. Area Number of Bedrooms': Avg Number of Bedrooms for Houses in same city
# * 'Area Population': Population of city house is located in
# * 'Price': Price that the house sold at
# * 'Address': Address for the house

# **Let's get started!**
# ## Check out the data
# I've been able to get some data for housing prices as a csv set, let's get the environment ready with the libraries I'll need and then import the data!
# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Check out the Data

# In[2]:


USAhousing = pd.read_csv('USA_Housing.csv')


# In[3]:


USAhousing.head()


# In[4]:


USAhousing.info()


# In[5]:


USAhousing.describe()


# In[6]:


USAhousing.columns


# # Exploratory Data Analysis
# 
# Let's create some simple plots to check out the data!

# In[7]:


sns.pairplot(USAhousing)


# In[8]:


sns.distplot(USAhousing['Price'])


# In[9]:


sns.heatmap(USAhousing.corr(), cmap='viridis')


# The above heatmap is a great way to see the correlations between different attributes present in data at once.

# ## Training a Linear Regression Model
# 
# Let's now begin to train a regression model! I will need to first split up the data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. I will toss out the Address column because it only has text info that the linear regression model can't use.
# 
# ### X and y arrays

# In[10]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# ## Train Test Split
# 
# Now let's split the data into a training set and a testing set. I will train out model on the training set and then use the test set to evaluate the model.

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Creating and Training the Model

# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lm = LinearRegression()


# In[15]:


lm.fit(X_train,y_train)


# ## Model Evaluation
# 
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[16]:


# print the intercept
print(lm.intercept_)


# In[17]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80 **.
# - Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15 **.
# 
# Does this make sense? Probably not because ** I made up this data.**
# 
# 

# ## Predictions from the Model
# 
# Let's grab predictions off our test set and see how well it did!

# In[18]:


predictions = lm.predict(X_test)


# In[19]:


plt.scatter(y_test,predictions)


# **Residual Histogram**

# In[20]:


sns.distplot((y_test-predictions),bins=50);


# ## Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because I want to minimize them.

# In[21]:


from sklearn import metrics


# In[22]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# 
# ## END!
