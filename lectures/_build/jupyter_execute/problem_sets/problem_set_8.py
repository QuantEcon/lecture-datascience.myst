#!/usr/bin/env python
# coding: utf-8

# # Problem Set 8
# 
# In this problem set, your goal is to train a model to best predict
# log housing values. The criteria for best prediction is mean squared error.
# The file ahs.csv contains data from the [American Housing
# Survey](https://www.census.gov/programs-surveys/ahs.html) . Your predictive model will be
# graded based on another evaluation sample from the same survey.
# You should create a function that returns the predictions of your model
# when given an identically-formatted csv file with all the same variables.
# (Your function should not refit your model on the evaluation sample.)
# In addition, answer the questions below.
# 
# ## Additional Rules
# 
# You may not use additional data from the American Housing Survey to
# fit your model. You may use data from other sources (although this
# is not necessary to receive a good grade). You may use methods not
# covered in this course (although this is also not necessary to receive
# a good grade).

# In[1]:


import pandas as pd
import numpy as np
from sklearn import (
    linear_model, metrics, neural_network, pipeline,
    model_selection, tree
)
from sklearn.ensemble import RandomForestRegressor
# load data
ahs = pd.read_csv("ahs-train.csv")
ahs.info()


# In[2]:


# dataframe of variable descriptions
ahs_doc = pd.read_csv("ahs-doc.csv", encoding="latin1")
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'max_colwidth', -1):
    display(ahs_doc[["Variable","Question","Description","Associated.Response.Codes"]])


# ### Question 1
# 
# Create exploratory table(s) and/or visualization(s) to check the data
# and help make modelling choices. These need not be very polished.
# 
# ### Question 2
# 
# What model will you use for prediction and why you
# did you choose this model?
# 
# ### Question 3
# 
# Briefly describe how you chose any regularization and other parameters
# in your model.
# 
# ### Question 4
# 
# What have you done to avoid overfitting?
# 
# ### Question 5
# 
# Create a visualization to help evaluate of your model. This
# visualization can be part of your answer to questions 2-4 or it
# can simply summarize your model's predictive accuracy.
# 
# ### Question 6
# 
# Create a function that returns the predictions of
# your model when given an identically-formatted pandas DataFrame
# (created from an identically formatted csv file by pd.read_csv)
# with all the same variables. Your function should not refit your model on
# the evaluation sample.
