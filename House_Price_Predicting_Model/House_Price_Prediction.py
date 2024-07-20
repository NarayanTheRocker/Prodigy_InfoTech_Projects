#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd


# In[61]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[45]:


from sklearn.linear_model import LinearRegression


# In[46]:


import matplotlib.pyplot as plt


# In[47]:


ds = pd.read_csv("train.csv")


# In[48]:


ds


# In[49]:


features = ["GrLivArea","BedroomAbvGr","FullBath"]


# In[50]:


x = ds[features]


# In[51]:


y = ds["SalePrice"]


# In[52]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[53]:


model = LinearRegression()


# In[54]:


model.fit(x_train, y_train)


# In[55]:


y_pred = model.predict(x_test)


# In[59]:


mse = mean_squared_error(y_test, y_pred)


# In[62]:


r2 = r2_score(y_test, y_pred)


# In[63]:


print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[66]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Prices")


# In[68]:


new_data = pd.DataFrame({'GrLivArea':[2000], 'BedroomAbvGr':[3], 'FullBath': [2]})
predicted_price = model.predict(new_data)
print(f'Predicted Price: {predicted_price[0]}')


# In[ ]:




