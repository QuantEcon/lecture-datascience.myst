#!/usr/bin/env python
# coding: utf-8

# # Problem Set 5
# 
# See {doc}`../scientific/optimization <../scientific/optimization>`, {doc}`../pandas/intro <../pandas/intro>`, and {doc}`../pandas/basics <../pandas/basics>`

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
# activate plot theme
import qeds
qeds.themes.mpl_style();


# ## Setup for Question 1-5
# 
# Load data from the {doc}`../pandas/basics <../pandas/basics>` lecture.

# In[2]:


url = "https://datascience.quantecon.org/assets/data/state_unemployment.csv"
unemp_raw = pd.read_csv(url, parse_dates=["Date"])


# And do the same manipulation as in the pandas basics lecture.

# In[3]:


states = [
    "Arizona", "California", "Florida", "Illinois",
    "Michigan", "New York", "Texas"
]

unemp = (
    unemp_raw
    .reset_index()
    .pivot_table(index="Date", columns="state", values="UnemploymentRate")
    [states]
)


# ## Question 1
# 
# At each date, what is the minimum unemployment rate across all states
# in our sample?

# In[4]:


# Your code here


# What was the median unemployment rate in each state?

# In[5]:


# Your code here


# What was the maximum unemployment rate across the states in our
# sample? In what state did it happen? In what month/year was this
# achieved?
# 
# - Hint 1: What Python type (not `dtype`) is returned by a reduction?
# - Hint 2: Read documentation for the method `idxmax`.

# In[6]:


# Your code here


# Classify each state as high or low volatility based on whether the
# variance of their unemployment is above or below 4.

# In[7]:


# Your code here


# ## Question 2
# 
# Imagine that we want to determine whether unemployment was high (> 6.5),
# medium (4.5 < x <= 6.5), or low (<= 4.5) for each state and each month.
# 
# Write a Python function that takes a single number as an input and
# outputs a single string which notes whether that number is high, medium, or low.

# In[8]:


# Your code here


# Pass your function to either `apply` or `applymap` and save the
# result in a new DataFrame called `unemp_bins`.

# In[9]:


# Your code here


# ## Question 3
# 
# This exercise has multiple parts:
# 
# Use another transformation on `unemp_bins` to count how many
# times each state had each of the three classifications.
# 
# - Hint 1: Will you need to use `apply` or `applymap` for transformation?
# - Hint 2: Try googling "pandas count unique value" or something similar to find the proper transformation.

# In[10]:


# Your code here


# Construct a horizontal bar chart to detail the occurrences of each level.
# Use one bar per state and classification for 21 total bars.

# In[11]:


# Your code here


# ## Question 4
# 
# Repeat Question 3, but count how many states had
# each classification in each month. Which month had the most states
# with high unemployment? What about medium and low?
# 
# Part 1: Write a Python function to classify unemployment levels

# In[12]:


# Your code here


# Part 2: Decide whether you should use `.apply` or `.applymap`.
# 
# Part 3: Pass your function from part 1 to the method you determined in Part 2.

# In[13]:


unemp_bins = unemp#replace this comment with your code!!


# Part 4: Count the number of times each state had each classification.

# In[14]:


## then make a horizontal bar chart here


# Part 5: Apply the same transformation from Part 4 to each date instead of to each state.

# In[15]:


# Your code here


# ## Question 5
# 
# For a single state of your choice, determine the mean
# unemployment during "Low", "Medium", and "High" unemployment times.
# (recall your `unemp_bins` DataFrame from the exercise above)

# In[16]:


# Your code here


# Which states in our sample performs the best during "bad times?" To
# determine this, compute each state's mean unemployment in
# months where the mean unemployment rate is greater than 7.

# In[17]:


# Your code here

