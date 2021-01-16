---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Basic Functionality

**Prerequisites**

- {doc}`pandas Intro <intro>`

**Outcomes**

- Be familiar with `datetime`
- Use built-in aggregation functions and be able to create your own and
  apply them using `agg`
- Use built-in Series transformation functions and be able to create your
  own and apply them using `apply`
- Use built-in scalar transformation functions and be able to create your
  own and apply them using `applymap`
- Be able to select subsets of the DataFrame using boolean selection
- Know what the "want operator" is and how to apply it

**Data**

- US state unemployment data from Bureau of Labor Statistics

```{contents} Outline
:depth: 2
```

```{literalinclude} _static/colab_light.raw
```

## State Unemployment Data

In this lecture, we will use unemployment data by state at a monthly
frequency.

```{code-cell} python
import pandas as pd

%matplotlib inline
# activate plot theme
import qeds
qeds.themes.mpl_style();

pd.__version__
```

First, we will download the data directly from a url and read it into a pandas DataFrame.

```{code-cell} python
## Load up the data -- this will take a couple seconds
url = "https://datascience.quantecon.org/assets/data/state_unemployment.csv"
unemp_raw = pd.read_csv(url, parse_dates=["Date"])
```

The pandas `read_csv` will determine most datatypes of the underlying columns.  The
exception here is that we need to give pandas a hint so it can load up the `Date` column as a Python datetime type: the `parse_dates=["Date"]`.

We can see the basic structure of the downloaded data by getting the first 5 rows, which directly matches
the underlying CSV file.

```{code-cell} python
unemp_raw.head()
```

Note that a row has a date, state, labor force size, and unemployment rate.

For our analysis, we want to look at the unemployment rate across different states over time, which
requires a transformation of the data similar to an Excel pivot-table.

```{code-cell} python
# Don't worry about the details here quite yet
unemp_all = (
    unemp_raw
    .reset_index()
    .pivot_table(index="Date", columns="state", values="UnemploymentRate")
)
unemp_all.head()
```

Finally, we can filter it to look at a subset of the columns (i.e. "state" in this case).

```{code-cell} python
states = [
    "Arizona", "California", "Florida", "Illinois",
    "Michigan", "New York", "Texas"
]
unemp = unemp_all[states]
unemp.head()
```

When plotting, a DataFrame knows the column and index names.

```{code-cell} python
unemp.plot(figsize=(8, 6))
```

````{exercise}
**Exercise 1**

Looking at the displayed DataFrame above, can you identify the index? The columns?

You can use the cell below to verify your visual intuition.

```{code-block} python
# your code here
```

````

## Dates in pandas

You might have noticed that our index now has a nice format for the
dates (`YYYY-MM-DD`) rather than just a year.

This is because the `dtype` of the index is a variant of `datetime`.

```{code-cell} python
unemp.index
```

We can index into a DataFrame with a `DatetimeIndex` using string
representations of dates.

For example

```{code-cell} python
# Data corresponding to a single date
unemp.loc["01/01/2000", :]
```

```{code-cell} python
# Data for all days between New Years Day and June first in the year 2000
unemp.loc["01/01/2000":"06/01/2000", :]
```

We will learn more about what pandas can do with dates and times in an
upcoming lecture on time series data.

## DataFrame Aggregations

Let's talk about *aggregations*.

Loosely speaking, an aggregation is an operation that combines multiple
values into a single value.

For example, computing the mean of three numbers (for example
`[0, 1, 2]`) returns a single number (1).

We will use aggregations extensively as we analyze and manipulate our data.

Thankfully, pandas makes this easy!

### Built-in Aggregations

pandas already has some of the most frequently used aggregations.

For example:

- Mean  (`mean`)
- Variance (`var`)
- Standard deviation (`std`)
- Minimum (`min`)
- Median (`median`)
- Maximum (`max`)
- etc...

```{note}
When looking for common operations, using "tab completion" goes a long way.
```

```{code-cell} python
unemp.mean()
```

As seen above, the aggregation's default is to aggregate each column.

However, by using the `axis` keyword argument, you can do aggregations by
row as well.

```{code-cell} python
unemp.var(axis=1).head()
```

### Writing Your Own Aggregation

The built-in aggregations will get us pretty far in our analysis, but
sometimes we need more flexibility.

We can have pandas perform custom aggregations by following these two
steps:

1. Write a Python function that takes a `Series` as an input and
   outputs a single value.
1. Call the `agg` method with our new function as an argument.

For example, below, we will classify states as "low unemployment" or
"high unemployment" based on whether their mean unemployment level is
above or below 6.5.

```{code-cell} python
#
# Step 1: We write the (aggregation) function that we'd like to use
#
def high_or_low(s):
    """
    This function takes a pandas Series object and returns high
    if the mean is above 6.5 and low if the mean is below 6.5
    """
    if s.mean() < 6.5:
        out = "Low"
    else:
        out = "High"

    return out
```

```{code-cell} python
#
# Step 2: Apply it via the agg method.
#
unemp.agg(high_or_low)
```

```{code-cell} python
# How does this differ from unemp.agg(high_or_low)?
unemp.agg(high_or_low, axis=1).head()
```

Notice that `agg` can also accept multiple functions at once.

```{code-cell} python
unemp.agg([min, max, high_or_low])
```

````{exercise}
**Exercise 2**

Do the following exercises in separate code cells below:

- At each date, what is the minimum unemployment rate across all states
  in our sample?
- What was the median unemployment rate in each state?
- What was the maximum unemployment rate across the states in our
  sample? What state did it happen in? In what month/year was this
  achieved?
    - Hint 1: What Python type (not `dtype`) is returned by the
      aggregation?
    - Hint 2: Read documentation for the method `idxmax`
- Classify each state as high or low volatility based on whether the
  variance of their unemployment is above or below 4.

```{code-block} python
# min unemployment rate by state
```

```{code-block} python
# median unemployment rate by state
```

```{code-block} python
# max unemployment rate across all states and Year
```

```{code-block} python
# low or high volatility
```

````

## Transforms

Many analytical operations do not necessarily involve an aggregation.

The output of a function applied to a Series might need to be a new
Series.

Some examples:

- Compute the percentage change in unemployment from month to month.
- Calculate the cumulative sum of elements in each column.

### Built-in Transforms

pandas comes with many transform functions including:

- Cumulative sum/max/min/product (`cum(sum|min|max|prod)`)
- Difference  (`diff`)
- Elementwise addition/subtraction/multiplication/division (`+`, `-`, `*`, `/`)
- Percent change (`pct_change`)
- Number of occurrences of each distinct value (`value_counts`)
- Absolute value (`abs`)

Again, tab completion is helpful when trying to find these functions.

```{code-cell} python
unemp.head()
```

```{code-cell} python
unemp.pct_change(fill_method = None).head() # Skip calculation for missing data
```

```{code-cell} python
unemp.diff().head()
```

Transforms can be split into to several main categories:

1. *Series transforms*: functions that take in one Series and produce another Series. The index of the input and output does not need to be the same.
1. *Scalar transforms*: functions that take a single value and produce a single value. An example is the `abs` method, or adding a constant to each value of a Series.

### Custom Series Transforms

pandas also simplifies applying custom Series transforms to a Series or the
columns of a DataFrame. The steps are:

1. Write a Python function that takes a Series and outputs a new Series.
1. Pass our new function as an argument to the `apply` method (alternatively, the `transform` method).

As an example, we will standardize our unemployment data to have mean 0
and standard deviation 1.

After doing this, we can use an aggregation to determine at which date the
unemployment rate is most different from "normal times" for each state.

```{code-cell} python
#
# Step 1: We write the Series transform function that we'd like to use
#
def standardize_data(x):
    """
    Changes the data in a Series to become mean 0 with standard deviation 1
    """
    mu = x.mean()
    std = x.std()

    return (x - mu)/std
```

```{code-cell} python
#
# Step 2: Apply our function via the apply method.
#
std_unemp = unemp.apply(standardize_data)
std_unemp.head()
```

```{code-cell} python
# Takes the absolute value of all elements of a function
abs_std_unemp = std_unemp.abs()

abs_std_unemp.head()
```

```{code-cell} python
# find the date when unemployment was "most different from normal" for each State
def idxmax(x):
    # idxmax of Series will return index of maximal value
    return x.idxmax()

abs_std_unemp.agg(idxmax)
```

### Custom Scalar Transforms

As you may have predicted, we can also apply custom scalar transforms to our
pandas data.

To do this, we use the following pattern:

1. Define a Python function that takes in a scalar and produces a scalar.
1. Pass this function as an argument to the `applymap` Series or DataFrame method.

Complete the exercise below to practice writing and using your own scalar
transforms.

````{exercise}
**Exercise 3**

Imagine that we want to determine whether unemployment was high (> 6.5),
medium (4.5 < x <= 6.5), or low (<= 4.5) for each state and each month.

1. Write a Python function that takes a single number as an input and
   outputs a single string noting if that number is high, medium, or low.
1. Pass your function to `applymap` (quiz: why `applymap` and not
   `agg` or `apply`?) and save the result in a new DataFrame called
   `unemp_bins`.
1. (Challenging) This exercise has multiple parts:
    1. Use another transform on `unemp_bins` to count how many
       times each state had each of the three classifications.
        - Hint 1: Will this value counting function be a Series or scalar
          transform?
        - Hint 2: Try googling "pandas count unique value" or something
          similar to find the right transform.
    1. Construct a horizontal bar chart of the number of occurrences of
       each level with one bar per state and classification (21 total
       bars).
1. (Challenging) Repeat the previous step, but count how many states had
   each classification in each month. Which month had the most states
   with high unemployment? What about medium and low?

```{code-block} python
# Part 1: Write a Python function to classify unemployment levels.
```

```{code-block} python
# Part 2: Pass your function from part 1 to applymap
unemp_bins = unemp.applymap#replace this comment with your code!!
```

```{code-block} python
# Part 3: Count the number of times each state had each classification.

## then make a horizontal bar chart here
```

```{code-block} python
# Part 4: Apply the same transform from part 4, but to each date instead of to each state.
```

````

## Boolean Selection

We have seen how we can select subsets of data by referring to the index
or column names.

However, we often want to select based on conditions met by
the data itself.

Some examples are:

- Restrict analysis to all individuals older than 18.
- Look at data that corresponds to particular time periods.
- Analyze only data that corresponds to a recession.
- Obtain data for a specific product or customer ID.

We will be able to do this by using a Series or list of boolean values
to index into a Series or DataFrame.

Let's look at some examples.

```{code-cell} python
unemp_small = unemp.head()  # Create smaller data so we can see what's happening
unemp_small
```

```{code-cell} python
# list of booleans selects rows
unemp_small.loc[[True, True, True, False, False]]
```

```{code-cell} python
# second argument selects columns, the  ``:``  means "all".
# here we use it to select all columns
unemp_small.loc[[True, False, True, False, True], :]
```

```{code-cell} python
# can use booleans to select both rows and columns
unemp_small.loc[[True, True, True, False, False], [True, False, False, False, False, True, True]]
```

### Creating Boolean DataFrames/Series

We can use {doc}`conditional statements <../python_fundamentals/control_flow>` to
construct Series of booleans from our data.

```{code-cell} python
unemp_small["Texas"] < 4.5
```

Once we have our Series of bools, we can use it to extract subsets of
rows from our DataFrame.

```{code-cell} python
unemp_small.loc[unemp_small["Texas"] < 4.5]
```

```{code-cell} python
unemp_small["New York"] > unemp_small["Texas"]
```

```{code-cell} python
big_NY = unemp_small["New York"] > unemp_small["Texas"]
unemp_small.loc[big_NY]
```

#### Multiple Conditions

In the boolean section of the {doc}`basics lecture <../python_fundamentals/basics>`, we saw
that we can use the words `and` and `or` to combine multiple booleans into
a single bool.

Recall:

- `True and False -> False`
- `True and True -> True`
- `False and False -> False`
- `True or False -> True`
- `True or True -> True`
- `False or False -> False`

We can do something similar in pandas, but instead of
`bool1 and bool2` we write:

```{code-block} python
(bool_series1) & (bool_series2)
```

Likewise, instead of `bool1 or bool2` we write:

```{code-block} python
(bool_series1) | (bool_series2)
```

```{code-cell} python
small_NYTX = (unemp_small["Texas"] < 4.7) & (unemp_small["New York"] < 4.7)
small_NYTX
```

```{code-cell} python
unemp_small[small_NYTX]
```

#### `isin`

Sometimes, we will want to check whether a data point takes on one of a
several fixed values.

We could do this by writing `(df["x"] == val_1) | (df["x"] == val_2)`
(like we did above), but there is a better way: the `.isin` method

```{code-cell} python
unemp_small["Michigan"].isin([3.3, 3.2])
```

```{code-cell} python
# now select full rows where this Series is True
unemp_small.loc[unemp_small["Michigan"].isin([3.3, 3.2])]
```

#### `.any` and `.all`

Recall from the boolean section of the {doc}`basics lecture <../python_fundamentals/basics>`
that the Python functions `any` and `all` are aggregation functions that
take a collection of booleans and return a single boolean.

`any` returns True whenever at least one of the inputs are True while
`all` is True only when all the inputs are `True`.

Series and DataFrames with `dtype` bool have `.any` and `.all`
methods that apply this logic to pandas objects.

Let's use these methods to count how many months all the states in our
sample had high unemployment.

As we work through this example, consider the ["want
operator"](http://albertjmenkveld.com/2014/07/07/endogeneous-price-dispersion/), a helpful
concept from Nobel Laureate [Tom
Sargent](http://www.tomsargent.com) for clearly stating the goal of our analysis and
determining the steps necessary to reach the goal.

We always begin by writing `Want:` followed by what we want to
accomplish.

In this case, we would write:

> Want: Count the number of months in which all states in our sample
> had unemployment above 6.5%

After identifying the **want**, we work *backwards* to identify the
steps necessary to accomplish our goal.

So, starting from the result, we have:

1. Sum the number of `True` values in a Series indicating dates for
   which all states had high unemployment.
1. Build the Series used in the last step by using the `.all` method
   on a DataFrame containing booleans indicating whether each state had
   high unemployment at each date.
1. Build the DataFrame used in the previous step using a `>`
   comparison.

Now that we have a clear plan, let's follow through and *apply* the want
operator:

```{code-cell} python
# Step 3: construct the DataFrame of bools
high = unemp > 6.5
high.head()
```

```{code-cell} python
# Step 2: use the .all method on axis=1 to get the dates where all states have a True
all_high = high.all(axis=1)
all_high.head()
```

```{code-cell} python
# Step 1: Call .sum to add up the number of True values in `all_high`
#         (note that True == 1 and False == 0 in Python, so .sum will count Trues)
msg = "Out of {} months, {} had high unemployment across all states"
print(msg.format(len(all_high), all_high.sum()))
```

````{exercise}
**Exercise 4**

- For a single state of your choice, determine what the mean
  unemployment is during "Low", "Medium", and "High" unemployment times
  (recall your `unemp_bins` DataFrame from the exercise above).
    - Think about how you would do this for all the
      states in our sample and write your thoughts... We will soon
      learn tools that will *greatly* simplify operations like
      this that operate on distinct *groups* of data at a time.
- Which states in our sample performs the best during "bad times?" To
  determine this, compute the mean unemployment for each state only for
  months in which the mean unemployment rate in our sample is greater
  than 7.

````

## Exercises

````{exerciselist}
````

