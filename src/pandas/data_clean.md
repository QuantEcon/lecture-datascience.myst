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

# Cleaning Data

**Prerequisites**

- {doc}`Intro <intro>`
- {doc}`Boolean selection <basics>`
- {doc}`Indexing <the_index>`

**Outcomes**

- Be able to use string methods to clean data that comes as a string
- Be able to drop missing data
- Use cleaning methods to prepare and analyze a real dataset

**Data**

- Item information from about 3,000 Chipotle meals from about 1,800
  Grubhub orders

```{literalinclude} _static/colab_light.raw
```

```{code-cell} python
import pandas as pd
import numpy as np
import qeds
```


## Cleaning Data

For many data projects, a [significant proportion of
time](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/#74d447456f63)
is spent collecting and cleaning the data — not performing the analysis.

This non-analysis work is often called "data cleaning".

pandas provides very powerful data cleaning tools, which we
will demonstrate using the following dataset.

```{code-cell} python
df = pd.DataFrame({"numbers": ["#23", "#24", "#18", "#14", "#12", "#10", "#35"],
                   "nums": ["23", "24", "18", "14", np.nan, "XYZ", "35"],
                   "colors": ["green", "red", "yellow", "orange", "purple", "blue", "pink"],
                   "other_column": [0, 1, 0, 2, 1, 0, 2]})
df
```

What would happen if we wanted to try and compute the mean of
`numbers`?

```{code-cell} python
---
tags: [raises-exception]
---
df["numbers"].mean()
```

It throws an error!

Can you figure out why?

```{hint}
When looking at error messages, start at the very bottom.
```

The final error says, `TypeError: Could not convert #23#24... to numeric`.

````{exercise} 1
:nonumber:

Convert the string below into a number.

```{code-block} python
c2n = "#39"
```

````

## String Methods

Our solution to the previous exercise was to remove the `#` by using
the `replace` string method: `int(c2n.replace("#", ""))`.

One way to make this change to every element of a column would be to
loop through all elements of the column and apply the desired string
methods...

```{code-cell} python
%%time

# Iterate over all rows
for row in df.iterrows():

    # `iterrows` method produces a tuple with two elements...
    # The first element is an index and the second is a Series with the data from that row
    index_value, column_values = row

    # Apply string method
    clean_number = int(column_values["numbers"].replace("#", ""))

    # The `at` method is very similar to the `loc` method, but it is specialized
    # for accessing single elements at a time... We wanted to use it here to give
    # the loop the best chance to beat a faster method which we show you next.
    df.at[index_value, "numbers_loop"] = clean_number
```

While this is fast for a small dataset like this, this method slows for larger datasets.

One *significantly* faster (and easier) method is to apply a string
method to an entire column of data.

Most methods that are available to a Python string (we learned a
few of them in the {doc}`strings lecture <../python_fundamentals/basics>`) are
also available to a pandas Series that has `dtype` object.

We access them by doing `s.str.method_name` where `method_name` is
the name of the method.

When we apply the method to a Series, it is applied to all rows in the
Series in one shot!

Let's redo our previous example using a pandas `.str` method.

```{code-cell} python
%%time

# ~2x faster than loop... However, speed gain increases with size of DataFrame. The
# speedup can be in the ballpark of ~100-500x faster for big DataFrames.
# See appendix at the end of the lecture for an application on a larger DataFrame
df["numbers_str"] = df["numbers"].str.replace("#", "")
```

We can use `.str` to access almost any string method that works on
normal strings. (See the [official
documentation](https://pandas.pydata.org/pandas-docs/stable/text.html)
for more information.)

```{code-cell} python
df["colors"].str.contains("p")
```

```{code-cell} python
df["colors"].str.capitalize()
```

````{exercise} 2
:nonumber:

Make a new column called `colors_upper` that contains the elements of
`colors` with all uppercase letters.

````

## Type Conversions

In our example above, the `dtype` of the `numbers_str` column shows that pandas still treats
it as a string even after we have removed the `"#"`.

We need to convert this column to numbers.

The best way to do this is using the `pd.to_numeric` function.

This method attempts to convert whatever is stored in a Series into
numeric values

For example, after the `"#"` removed, the numbers of column
`"numbers"` are ready to be converted to actual numbers.

```{code-cell} python
df["numbers_numeric"] = pd.to_numeric(df["numbers_str"])
```

```{code-cell} python
df.dtypes
```

```{code-cell} python
df.head()
```

We can convert to other types well.

Using the `astype` method, we can convert to any of the supported
pandas `dtypes` (recall the {doc}`intro lecture <intro>`).

Below are some examples. (Pay attention to the reported `dtype`)

```{code-cell} python
df["numbers_numeric"].astype(str)
```

```{code-cell} python
df["numbers_numeric"].astype(float)
```

````{exercise} 3
:nonumber:

Convert the column `"nums"` to a numeric type using `pd.to_numeric` and
save it to the DataFrame as `"nums_tonumeric"`.

Notice that there is a missing value, and a value that is not a number.

Look at the documentation for `pd.to_numeric` and think about how to
overcome this.

Think about why this could be a bad idea of used without
knowing what your data looks like. (Think about what happens when you
apply it to the `"numbers"` column before replacing the `"#"`.)

````

## Missing Data

Many datasets have missing data.

In our example, we are missing an element from the `"nums"` column.

```{code-cell} python
df
```

We can find missing data by using the `isnull` method.

```{code-cell} python
df.isnull()
```

We might want to know whether particular rows or columns have any
missing data.

To do this we can use the `.any` method on the boolean DataFrame
`df.isnull()`.

```{code-cell} python
df.isnull().any(axis=0)
```

```{code-cell} python
df.isnull().any(axis=1)
```

Many approaches have been developed to deal with missing data, but the two most commonly used (and the corresponding DataFrame method) are:

- Exclusion: Ignore any data that is missing (`.dropna`).
- Imputation: Compute "predicted" values for the data that is missing
  (`.fillna`).

For the advantages and disadvantages of these (and other) approaches,
consider reading the [Wikipedia
article](https://en.wikipedia.org/wiki/Missing_data).

For now, let's see some examples.

```{code-cell} python
# drop all rows containing a missing observation
df.dropna()
```

```{code-cell} python
# fill the missing values with a specific value
df.fillna(value=100)
```

```{code-cell} python
# use the _next_ valid observation to fill the missing data
df.fillna(method="bfill")
```

```{code-cell} python
# use the _previous_ valid observation to fill missing data
df.fillna(method="ffill")
```

We will see more examples of dealing with missing data in future
chapters.

## Case Study

We will now use data from an
[article](https://www.nytimes.com/interactive/2015/02/17/upshot/what-do-people-actually-order-at-chipotle.html)
written by The Upshot at the NYTimes.

This data has order information from almost 2,000 Chipotle orders and
includes information on what was ordered and how much it cost.

```{code-cell} python
chipotle = qeds.data.load("chipotle_raw")
chipotle.head()
```

````{exercise} 4
:nonumber:

We'd like you to use this data to answer the following questions.

- What is the average price of an item with chicken?
- What is the average price of an item with steak?
- Did chicken or steak produce more revenue (total)?
- How many missing items are there in this dataset? How many missing
  items in each column?

```{hint}
Before you will be able to do any of these things you will need to
make sure the `item_price` column has a numeric `dtype` (probably
float).
```

````

## Appendix: Performance of `.str` Methods

Let's repeat the "remove the `#`" example from above, but this time on
a much larger dataset.

```{code-cell} python
import numpy as np
test = pd.DataFrame({"floats": np.round(100*np.random.rand(100000), 2)})
test["strings"] = test["floats"].astype(str) + "%"
test.head()
```

```{code-cell} python
%%time

for row in test.iterrows():
    index_value, column_values = row
    clean_number = column_values["strings"].replace("%", "")
    test.at[index_value, "numbers_loop"] = clean_number
```

```{code-cell} python
%%time
test["numbers_str_method"] = test["strings"].str.replace("%", "")
```

```{code-cell} python
test["numbers_str_method"].equals(test["numbers_loop"])
```

We got the exact same result in a fraction of the time!

## Exercises

````{exerciselist}
````

