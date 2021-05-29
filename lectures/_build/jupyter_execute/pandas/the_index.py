#!/usr/bin/env python
# coding: utf-8

# # The Index
# 
# **Prerequisites**
# 
# - {doc}`Introduction to pandas <intro>`
# 
# **Outcomes**
# 
# - Understand how the index is used to align data
# - Know how to set and reset the index
# - Understand how to select subsets of data by slicing on index and columns
# - Understand that for DataFrames, the column names also align data
# 
# 
# ```{literalinclude} ../_static/colab_light.raw
# ```

# In[1]:


import pandas as pd
import numpy as np


# ## So What is this Index?
# 
# Every Series or DataFrame has an index.
# 
# We told you that the index was the "row labels" for the data.
# 
# This is true, but an index in pandas does much more than label the rows.
# 
# The purpose of this lecture is to understand the importance of the index.
# 
# The [pandas
# documentation](https://pandas.pydata.org/pandas-docs/stable/dsintro.html)
# says
# 
# > Data alignment is intrinsic. The link between labels and data will
# > not be broken unless done so explicitly by you.
# 
# In practice, the index and column names are used to make sure the data is
# properly aligned when operating on multiple DataFrames.
# 
# This is a somewhat abstract concept that is best understood by
# example...
# 
# Let's begin by loading some data on GDP components that we collected from
# the World Bank's World Development Indicators Dataset.

# In[2]:


url = "https://datascience.quantecon.org/assets/data/wdi_data.csv"
df = pd.read_csv(url)
df.info()

df.head()


# We'll also extract a couple smaller DataFrames we can use in examples.

# In[3]:


df_small = df.head(5)
df_small


# In[4]:


df_tiny = df.iloc[[0, 3, 2, 4], :]
df_tiny


# In[5]:


im_ex = df_small[["Imports", "Exports"]]
im_ex_copy = im_ex.copy()
im_ex_copy


# Observe what happens when we evaluate `im_ex + im_ex_copy`.

# In[6]:


im_ex + im_ex_copy


# Notice that this operated *elementwise*, meaning that the `+`
# operation was applied to each element of `im_ex` and the corresponding
# element of `im_ex_copy`.
# 
# Let's take a closer look at `df_tiny`:

# In[7]:


df_tiny


# Relative to `im_ex` notice a few things:
# 
# - The row labeled `1` appears in `im_ex` but not `df_tiny`.
# - For row labels that appear in both, they are not in the same position
#   within each DataFrame.
# - Certain columns appear only in `df_tiny`.
# - The `Imports` and `Exports` columns are the 6th and 5th columns of
#   `df_tiny` and the 1st and 2nd of `im_ex`, respectively.
# 
# Now, let's see what happens when we try `df_tiny + im_ex`.

# In[8]:


im_ex_tiny = df_tiny + im_ex
im_ex_tiny


# Whoa, a lot happened! Let's break it down.
# 
# ### Automatic Alignment
# 
# For all (row, column) combinations that appear in both DataFrames (e.g.
# rows `[1, 3]` and columns `[Imports, Exports]`), the value of `im_ex_tiny`
# is equal to `df_tiny.loc[row, col] + im_ex.loc[row, col]`.
# 
# This happened even though the rows and columns were not in the same
# order.
# 
# We refer to this as pandas *aligning* the data for us.
# 
# To see how awesome this is, think about how to do something similar in
# Excel:
# 
# - `df_tiny` and `im_ex` would be in different sheets.
# - The index and column names would be the first column and row in each
#   sheet.
# - We would have a third sheet to hold the sum.
# - For each label in the first row and column of *either* the `df_tiny`
#   sheet or the `im_ex` sheet we would have to do a `IFELSE` to check
#   if the label exists in the other sheet and then a `VLOOKUP` to
#   extract the value.
# 
# In pandas, this happens automatically, behind the scenes, and *very
# quickly*.
# 
# ### Handling Missing Data
# 
# For all elements in row `1` or columns
# `["country", "year", "GovExpend", "Consumption", "GDP"]`,
# the value in `im_ex_tiny` is `NaN`.
# 
# This is how pandas represents *missing data*.
# 
# So, when pandas was trying to look up the values in `df_tiny` and `im_ex`, it could
# only find a value in one DataFrame: the other value was missing.
# 
# When pandas tries to add a number to something that is missing, it says
# that the result is missing (spelled `NaN`).
# 
# ````{admonition} Exercise
# :name: pd-idx-dir1
# See exercise 1 in the {ref}`exercise list <pd-idx-ex>`.
# ````
# 
# ## Setting the Index
# 
# For a DataFrame `df`, the `df.set_index` method allows us to use one
# (or more) of the DataFrame's columns as the index.
# 
# Here's an example.

# In[9]:


# first, create the DataFrame
df_year = df.set_index(["year"])
df_year.head()


# Now that the year is on the index, we can use `.loc` to extract all the
# data for a specific year.

# In[10]:


df_year.loc[2010]


# This would be helpful, for example, if we wanted to compute the difference
# in the average of all our variables from one year to the next.

# In[11]:


df_year.loc[2009].mean() - df_year.loc[2008].mean()


# Notice that pandas did a few things for us.
# 
# - After computing `.mean()`, the row labels (index) were the former column names.
# - These column names were used to align data when we wanted asked pandas to
#   compute the difference.
# 
# Suppose that someone asked you, "What was the GDP in the US in 2010?"
# 
# To compute that using `df_year` you might do something like this:

# In[12]:


df_year.loc[df_year["country"] == "United States", "GDP"].loc[2010]


# That was a lot of work!
# 
# Now, suppose that after seeing you extract that data, your friend asks you
# "What about GDP in Germany and the UK in 2010?"
# 
# To answer that question, you might write.

# In[13]:


df_year.loc[df_year["country"].isin(["United Kingdom", "Germany"]), "GDP"].loc[2010]


# Notice that this code is similar to the code above, but now provides a result
# that is ambiguous.
# 
# The two elements in the series both have with label 2010.
# 
# How do we know which is which?
# 
# We might think that the first value corresponds to the United Kingdom because
# that is what we listed first in the call to `isin`, but we would be wrong!
# 
# Let's check.

# In[14]:


df_year.loc[2010]


# Setting just the year as index has one more potential issue: we will
# get data alignment only on the year, which may not be sufficient.
# 
# To demonstrate this point, suppose now you are asked to use our WDI dataset
# to compute an approximation for net exports and investment in 2009.
# 
# As a seasoned economist, you would remember the expenditure formula for GDP is
# written
# 
# $$
# GDP = Consumption + Investment + GovExpend + Net Exports
# $$
# 
# which we can rearrange to compute investment as a function of the variables in
# our DataFrame...
# 
# $$
# Investment = GDP - Consumption - GovExpend - Net Exports
# $$
# 
# Note that we can compute NetExports as `Exports - Imports`.

# In[15]:


nx = df_year["Exports"] - df_year["Imports"]
nx.head(19)


# Now, suppose that we accidentally had a bug in our code that swapped
# the data for Canada and Germany's net exports in 2017.
# 
# ```{note}
# This example is contrived, but if you were getting unclean data from
# some resource or doing more complicated operations, this type of mistake
# becomes increasingly likely.
# ```

# In[16]:


ca17 = nx.iloc[[0]]
g17 = nx.iloc[[18]]
nx.iloc[[0]] = g17
nx.iloc[[18]] = ca17

nx.head(19)


# Notice that if we now add `nx` to the DataFrame and compute investment
# pandas doesn't complain.

# In[17]:


df_year["NetExports"] = nx
df_year["Investment"] = df_year.eval("GDP - Consumption - GovExpend - NetExports")
df_year.head(19)


# Because we didn't also have data alignment on the country, we would have overstated Canada's investment by 281 billion USD and understated Germany's by the
# same amount.
# 
# To make these types operation easier, we need to include both the year
# and country in the index...
# 
# ### Setting a Hierarchical Index
# 
# Include multiple columns in the index is advantageous in some situations.
# 
# These situations might include:
# 
# - When we need more than one piece of information (column) to identify an
#   observation (as in the Germany and UK GDP example above)
# - When we need data-alignment by more than one column
# 
# To achieve multiple columns in the index, we pass a list of multiple column
# names to `set_index`.

# In[18]:


wdi = df.set_index(["country", "year"])
wdi.head(20)


# Notice that in the display above, the row labels seem to have two
# *levels* now.
# 
# The *outer* (or left-most) level is named `country` and the *inner* (or
# right-most) level is named `year`.
# 
# When a DataFrame's index has multiple levels, we (and the pandas documentation)
# refer to the DataFrame as having a hierarchical index.
# 
# ### Slicing a Hierarchical Index
# 
# Now, we can answer our friend's questions in a much more straightforward way.

# In[19]:


wdi.loc[("United States", 2010), "GDP"]


# In[20]:


wdi.loc[(["United Kingdom", "Germany"], 2010), "GDP"]


# As shown above, we can use `wdi.loc` to extract different slices of our
# national accounts data.
# 
# The rules for using `.loc` with a hierarchically-indexed DataFrame are
# similar to the ones we've learned for standard DataFrames, but they are a bit
# more elaborate as we now have more structure to our data.
# 
# We will summarize the main rules, and then work through an exercise that
# demonstrates each of them.
# 
# **Slicing rules**
# 
# pandas slicing reacts differently to `list`s and `tuple`s.
# 
# It does this to provide more flexibility to select the
# data you want.
# 
# `list` in row slicing will be an "or" operation, where it chooses rows
# based on whether the index value corresponds to any element of the list.
# 
# `tuple` in row slicing will be used to denote a single hierarchical
# index and must include a value for each level.
# 
# **Row slicing examples**
# 
# 1. `wdi.loc["United States"]`: all rows where the *outer* most index value is
#    equal to `United States`
# 1. `wdi.loc[("United States", 2010)]`: all rows where the *outer-most* index value
#    is equal to `"United States` and the second level is equal to `2010`
# 1. `wdi.loc[["United States", "Canada"]]`: all rows where the *outer-most* index is
#    either `"United States"` or `"Canada"`
# 1. `wdi.loc[(["United States", "Canada"], [2010, 2011]), :]`: all rows where the
#    *outer-most* index is either `"United States` or `"Canada"` AND where the
#    second level index is either `2010` or `2011`
# 1. `wdi.loc[[("United States", 2010), ("Canada", 2011)], :]`: all rows where the the
#    two hierarchical indices are either `("United States", 2010)` or
#    `("Canada", 2011)`
# 
# We can also restrict `.loc` to extract certain columns by doing:
# 
# 1. `wdi.loc[rows, GDP]`: return the rows specified by rows (see rules
#    above) and only column named `GDP` (returned object will be a
#    Series)
# 1. `df.loc[rows, ["GDP", "Consumption"]]`: return the rows specified by rows
#    (see rules above) and only columns `GDP` and `Consumption`
# 
# ````{admonition} Exercise
# :name: pd-idx-dir2
# See exercise 2 in the {ref}`exercise list <pd-idx-ex>`.
# ````
# 
# ### Alignment with `MultiIndex`
# 
# The data alignment features we talked about above also apply to a
# `MultiIndex` DataFrame.
# 
# The exercise below gives you a chance to experiment with this.
# 
# ````{admonition} Exercise
# :name: pd-idx-dir3
# See exercise 3 in the {ref}`exercise list <pd-idx-ex>`.
# ````
# 
# ### `pd.IndexSlice`
# 
# When we want to extract rows for a few values of the outer index and all
# values for an inner index level, we can use the convenient
# `df.loc[[id11, id22]]` shorthand.
# 
# We can use this notation to extract all the data for the United States and
# Canada.

# In[21]:


wdi.loc[["United States", "Canada"]]


# However, suppose we wanted to extract the data for all countries, but only the
# years 2005, 2007, and 2009.
# 
# We cannot do this using `wdi.loc` because the year is on the second level,
# not outer-most level of our index.
# 
# To get around this limitation, we can use the `pd.IndexSlice` helper.
# 
# Here's an example.

# In[22]:


wdi.loc[pd.IndexSlice[:, [2005, 2007, 2009]], :]


# Notice that the `:` in the first part of `[:, ["A", "D"]]`
# instructed pandas to give us rows for all values of the outer most index
# level and that the `:` just before `]` said grab all the columns.
# 
# ````{admonition} Exercise
# :name: pd-idx-dir4
# See exercise 4 in the {ref}`exercise list <pd-idx-ex>`.
# ````
# 
# ### Multi-index Columns
# 
# The functionality of `MultiIndex` also applies to the column names.
# 
# Let's see how it works.

# In[23]:


wdiT = wdi.T  # .T means "transpose" or "swap rows and columns"
wdiT


# Notice that `wdiT` seems to have two levels of names for the columns.
# 
# The same logic laid out in the above row slicing rules applies when we
# have a hierarchical index for column names.

# In[24]:


wdiT.loc[:, "United States"]


# In[25]:


wdiT.loc[:, ["United States", "Canada"]]


# In[26]:


wdiT.loc[:, (["United States", "Canada"], 2010)]


# ````{admonition} Exercise
# :name: pd-idx-dir5
# See exercise 5 in the {ref}`exercise list <pd-idx-ex>`.
# ````
# 
# ## Re-setting the Index
# 
# The `df.reset_index` method will move one or more level of the index
# back into the DataFrame as a normal column.
# 
# With no additional arguments, it moves all levels out of the index and
# sets the index of the returned DataFrame to the default of
# `range(df.shape[0])`.

# In[27]:


wdi.reset_index()


# ````{admonition} Exercise
# :name: pd-idx-dir6
# See exercise 6 in the {ref}`exercise list <pd-idx-ex>`.
# ````
# 
# ## Choose the Index Carefully
# 
# So, now that we know that we use index and column names for
# aligning data, "how should we pick the index?" is a natural question to ask.
# 
# To guide us to the right answer, we will list the first two components
# to [Hadley Wickham's](http://hadley.nz/) description of [tidy
# data](http://vita.had.co.nz/papers/tidy-data.html):
# 
# 1. Each column should each have one variable.
# 1. Each row should each have one observation.
# 
# If we strive to have our data in a tidy form (we should), then when
# choosing the index, we should set:
# 
# - the row labels (index) to be a unique identifier for an observation
#   of data
# - the column names to identify one variable
# 
# For example, suppose we are looking data on interest rates.
# 
# Each column might represent one bond or asset and each row might
# represent the date.
# 
# Using hierarchical row and column indices allows us to store higher
# dimensional data in our (inherently) two dimensional DataFrame.
# 
# ### Know Your Goal
# 
# The correct column(s) to choose for the index often depends on the context of
# your analysis.
# 
# For example, if I were studying how GDP and consumption evolved over time for
# various countries, I would want time (year) and country name on the index
# 
# On the other hand, if I were trying to look at the differences across countries
# and variables within a particular year, I may opt to put the country and
# variable on the index and have years be columns.
# 
# Following the tidy data rules above and thinking about how you intend to *use*
# the data -- and a little practice -- will enable you to consistently select the
# correct index.
# 
# (pd-idx-ex)=
# ## Exercises
# 
# ### Exercise 1
# 
# What happens when you apply the `mean` method to `im_ex_tiny`?
# 
# In particular, what happens to columns that have missing data? 
# ```{hint}
# Also looking at the output of the `sum` method might help.
# ```
# 
# ({ref}`back to text <pd-idx-dir1>`)
# 
# ### Exercise 2
# 
# For each of the examples below do the following:
# 
# - Determine which of the rules above applies.
# - Identify the `type` of the returned value.
# - Explain why the slicing operation returned the data it did.
# 
# Write your answers.

# In[28]:


wdi.loc[["United States", "Canada"]]


# In[29]:


wdi.loc[(["United States", "Canada"], [2010, 2011, 2012]), :]


# In[30]:


wdi.loc["United States"]


# In[31]:


wdi.loc[("United States", 2010), ["GDP", "Exports"]]


# In[32]:


wdi.loc[("United States", 2010)]


# In[33]:


wdi.loc[[("United States", 2010), ("Canada", 2015)]]


# In[34]:


wdi.loc[["United States", "Canada"], "GDP"]


# In[35]:


wdi.loc["United States", "GDP"]


# ({ref}`back to text <pd-idx-dir2>`)
# 
# ### Exercise 3
# 
# Try setting `my_df` to some subset of the rows in `wdi` (use one of the
# `.loc` variations above).
# 
# Then see what happens when you do `wdi / my_df` or `my_df ** wdi`.
# 
# Try changing the subset of rows in `my_df` and repeat until you
# understand what is happening.
# 
# ({ref}`back to text <pd-idx-dir3>`)
# 
# ### Exercise 4
# 
# Below, we create `wdi2`, which is the same as `df4` except that the
# levels of the index are swapped.
# 
# In the cells after `df6` is defined, we have commented out
# a few of the slicing examples from the previous exercise.
# 
# For each of these examples, use `pd.IndexSlice` to extract the same
# data from `df6`.
# 
# ```{hint}
# You will need to *swap* the order of the row slicing arguments
# within the `pd.IndexSlice`.
# ```

# In[36]:


wdi2 = df.set_index(["year", "country"])


# In[37]:


# wdi.loc["United States"]


# In[38]:


# wdi.loc[(["United States", "Canada"], [2010, 2011, 2012]), :]


# In[39]:


# wdi.loc[["United States", "Canada"], "GDP"]


# ({ref}`back to text <pd-idx-dir4>`)
# 
# ### Exercise 5
# 
# Use `pd.IndexSlice` to extract all data from `wdiT` where the `year`
# level of the column names (the second level) is one of 2010, 2012, and 2014
# 
# ({ref}`back to text <pd-idx-dir5>`)
# 
# ### Exercise 6
# 
# Look up the documentation for the `reset_index` method and study it to
# learn how to do the following:
# 
# - Move just the `year` level of the index back as a column.
# - Completely throw away all levels of the index.
# - Remove the `country` of the index and *do not* keep it as a column.

# In[40]:


# remove just year level and add as column


# In[41]:


# throw away all levels of index


# In[42]:


# Remove country from the index -- don't keep it as a column


# ({ref}`back to text <pd-idx-dir6>`)
