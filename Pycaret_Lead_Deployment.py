#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pycaret


# In[ ]:





# In[10]:


import pandas as pd
import numpy as np

data =pd.read_csv(r'C:\Users\himanshu.s\Desktop\Leads.csv')
data.head()


# In[11]:


data.info()


# In[12]:


# init setup**
from pycaret.classification import *
s = setup(data, target = 'Converted', ignore_features = ['Prospect ID', 'Lead Number'])


# In[14]:


best_model = compare_models(sort='AUC')


# In[15]:


# Print specific parameters of the best model
print(f"Learning rate: {best_model.learning_rate}")
print(f"Number of estimators: {best_model.n_estimators}")
print(f"Maximum depth: {best_model.max_depth}")


# In[16]:


print(best_model)


# In[17]:


plot_model(best_model, plot = 'auc')


# In[18]:


# Shapley Values**
interpret_model(best_model)


# In[19]:


# Feature Importance
plot_model(best_model, plot = 'feature')


# In[20]:


# Confusion Matrix 
plot_model(best_model, plot = 'confusion_matrix')


# ## How to use the model to generate a lead score?

# In[21]:


# create copy of data
data_new = data.copy()
data_new.drop('Converted', axis=1, inplace=True)


# In[22]:


# generate labels using predict_model
predict_model(best_model, data=data_new, raw_score=True)


# ## Save Model

# In[ ]:


# save_model(best_model, model_name='best_model')


# In[ ]:


loaded_bestmodel = load_model('best_model')
print(loaded_bestmodel)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




