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

# Time series

**Prerequisites**

- {doc}`Python functions <../python_fundamentals/functions>`
- {doc}`GroupBy <./groupby>`

**Outcomes**

- Know how pandas handles dates
- Understand how to parse strings into `datetime` objects
- Know how to write dates as custom formatted strings
- Be able to access day, month, year, etc. for a `DateTimeIndex` and
  a column with `dtype` `datetime`
- Understand both rolling and re-sampling operations and the difference
  between the two

**Data**

- Bitcoin to USD exchange rates from March 2014 to the present


```{literalinclude} ../_static/colab_light.raw
```

```{code-cell} python
import os
import pandas as pd
import matplotlib.pyplot as plt
import quandl

# see section on API keys at end of lecture!
quandl.ApiConfig.api_key = os.environ.get("QUANDL_AUTH", "Dn6BtVoBhzuKTuyo6hbp")
start_date = "2014-05-01"

%matplotlib inline
```

## Intro

pandas has extensive support for handling dates and times.

We will loosely refer to data with date or time information as time
series data.

In this lecture, we will cover the most useful parts of pandas' time
series functionality.

Among these topics are:

- Parsing strings as dates
- Writing `datetime` objects as (inverse operation of previous point)
- Extracting data from a DataFrame or Series with date information in
  the index
- Shifting data through time (taking leads or lags)
- Re-sampling data to a different frequency and rolling operations

However, even more than with previous topics, we will skip a lot of the
functionality pandas offers, and we urge you to refer to the [official
documentation](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)
for more information.

## Parsing Strings as Dates

When working with time series data, we almost always receive the data
with dates encoded as strings.

Hopefully, the date strings follow a structured format or pattern.

One common pattern is `YYYY-MM-DD`: 4 numbers for the year, 2 for the
month, and 2 for the day with each section separated by a `-`.

For example, we write Christmas day 2017 in this format as

```{code-cell} python
christmas_str = "2017-12-25"
```

To convert a string into a time-aware object, we use the
`pd.to_datetime` function.

```{code-cell} python
christmas = pd.to_datetime(christmas_str)
print("The type of christmas is", type(christmas))
christmas
```

The `pd.to_datetime` function is pretty smart at guessing the format
of the date...

```{code-cell} python
for date in ["December 25, 2017", "Dec. 25, 2017",
             "Monday, Dec. 25, 2017", "25 Dec. 2017", "25th Dec. 2017"]:
    print("pandas interprets {} as {}".format(date, pd.to_datetime(date)))
```

However, sometimes we will need to give pandas a hint.

For example, that same time (midnight on Christmas) would be reported on
an Amazon transaction report as

```{code-cell} python
christmas_amzn = "2017-12-25T00:00:00+ 00 :00"
```

If we try to pass this to `pd.to_datetime`, it will fail.

```{code-cell} python
---
tags: [raises-exception]
---
pd.to_datetime(christmas_amzn)
```

To parse a date with this format, we need to specify the `format`
argument for `pd.to_datetime`.

```{code-cell} python
amzn_strftime = "%Y-%m-%dT%H:%M:%S+ 00 :00"
pd.to_datetime(christmas_amzn, format=amzn_strftime)
```

Can you guess what `amzn_strftime` represents?

Let's take a closer look at `amzn_strftime` and `christmas_amzn`.

```{code-cell} python
print(amzn_strftime)
print(christmas_amzn)
```

Notice that both of the strings have a similar form, but that instead of actual numerical values, `amzn_strftime` has *placeholders*.

Specifically, anywhere the `%` shows up is a signal to the `pd.to_datetime`
function that it is where relevant information is stored.

For example, the `%Y` is a stand-in for a four digit year, `%m` is
for 2 a digit month, and so on...

The official [Python
documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) contains a complete list of possible `%`something patterns that are accepted
in the `format` argument.

````{admonition} Exercise
:name: pd-tim-dir1
See exercise 1 in the {ref}`exercise list <pd-tim-ex>`.
````

### Multiple Dates

If we have dates in a Series (e.g. column of DataFrame) or a list, we
can pass the entire collection to `pd.to_datetime` and get a
collection of dates back.

We'll just show an example of that here as the mechanics are the same as
a single date.

```{code-cell} python
pd.to_datetime(["2017-12-25", "2017-12-31"])
```

## Date Formatting

We can use the `%`pattern format to have pandas write `datetime`
objects as specially formatted strings using the `strftime` (string
format time) method.

For example,

```{code-cell} python
christmas.strftime("We love %A %B %d (also written %c)")
```

````{admonition} Exercise
:name: pd-tim-dir2
See exercise 2 in the {ref}`exercise list <pd-tim-ex>`.
````

## Extracting Data

When the index of a DataFrame has date information and pandas
recognizes the values as `datetime` values, we can leverage some
convenient indexing features for extracting data.

The flexibility of these features is best understood through example,
so let's load up some data and take a look.

```{code-cell} python
btc_usd = quandl.get("BCHARTS/BITSTAMPUSD", start_date=start_date)
btc_usd.info()
btc_usd.head()
```

Here, we have the Bitcoin (BTC) to US dollar (USD) exchange rate from
March 2014 until today.

Notice that the type of index is `DateTimeIndex`.

This is the key that enables things like...

Extracting all data for the year 2015 by passing `"2015"` to `.loc`.

```{code-cell} python
btc_usd.loc["2015"]
```

We can also narrow down to specific months.

```{code-cell} python
# By month's name
btc_usd.loc["August 2017"]
```

```{code-cell} python
# By month's number
btc_usd.loc["08/2017"]
```

Or even a day...

```{code-cell} python
# By date name
btc_usd.loc["August 1, 2017"]
```

```{code-cell} python
# By date number
btc_usd.loc["08-01-2017"]
```

What can we pass as the `.loc` argument when we have a
`DateTimeIndex`?

Anything that can be converted to a `datetime` using
`pd.to_datetime`, *without* having to specify the format argument.

When that condition holds, pandas will return *all* rows whose date in
the index "belong" to that date or period.

We can also use the range shorthand notation to give a start and end
date for selection.

```{code-cell} python
btc_usd.loc["April 1, 2015":"April 10, 2015"]
```

````{admonition} Exercise
:name: pd-tim-dir3
See exercise 3 in the {ref}`exercise list <pd-tim-ex>`.
````

## Accessing Date Properties

Sometimes, we would like to directly access a part of the date/time.

If our date/time information is in the index, we can to `df.index.XX`
where `XX` is replaced by `year`, `month`, or whatever we would
like to access.

```{code-cell} python
btc_usd.index.year
```

```{code-cell} python
btc_usd.index.day
```

We can also do the same if the date/time information is stored in a
column, but we have to use a slightly different syntax.

```{code-block} python
df["column_name"].dt.XX
```

```{code-cell} python
btc_date_column = btc_usd.reset_index()
btc_date_column.head()
```

```{code-cell} python
btc_date_column["Date"].dt.year.head()
```

```{code-cell} python
btc_date_column["Date"].dt.month.head()
```

## Leads and Lags: `df.shift`

When doing time series analysis, we often want to compare data at one
date against data at another date.

pandas can help us with this if we leverage the `shift` method.

Without any additional arguments, `shift()` will move all data
*forward* one period, filling the first row with missing data.

```{code-cell} python
# so we can see the result of shift clearly
btc_usd.head()
```

```{code-cell} python
btc_usd.shift().head()
```

We can use this to compute the percent change from one day to the next.
(Quiz: Why does that work? Remember how pandas uses the index to *align*
data.)

```{code-cell} python
((btc_usd - btc_usd.shift()) / btc_usd.shift()).head()
```

Setting the first argument to `n` tells pandas to shift the data down
`n` rows (apply an `n` period lag).

```{code-cell} python
btc_usd.shift(3).head()
```

A negative value will shift the data *up* or apply a lead.

```{code-cell} python
btc_usd.shift(-2).head()
```

```{code-cell} python
btc_usd.shift(-2).tail()
```

````{admonition} Exercise
:name: pd-tim-dir4
See exercise 4 in the {ref}`exercise list <pd-tim-ex>`.
````

## Rolling Computations: `.rolling`

pandas has facilities that enable easy computation of *rolling
statistics*.

These are best understood by example, so we will dive right in.

```{code-cell} python
# first take only the first 6 rows so we can easily see what is going on
btc_small = btc_usd.head(6)
btc_small
```

Below, we compute the 2 day moving average (for all columns).

```{code-cell} python
btc_small.rolling("2d").mean()
```

To do this operation, pandas starts at each row (date) then looks
*backwards* the specified number of periods (here 2 days) and then
applies some aggregation function (`mean`) on all the data in that
window.

If pandas cannot look back the full length of the window (e.g. when
working on the first row), it fills as much of the window as possible
and then does the operation. Notice that the value at 2014-05-01 is
the same in both DataFrames.

Below, we see a visual depiction of the rolling maximum on a 21 day
window for the whole dataset.

```{code-cell} python
fig, ax = plt.subplots(figsize=(10, 4))
btc_usd["Open"].plot(ax=ax, linestyle="--", alpha=0.8)
btc_usd.rolling("21d").max()["Open"].plot(ax=ax, alpha=0.8, linewidth=3)
ax.legend(["Original", "21 day max"])
```

We can also ask pandas to `apply` custom functions, similar to what we
saw when studying GroupBy.

```{code-cell} python
def is_volatile(x):
    "Returns a 1 if the variance is greater than 1, otherwise returns 0"
    if x.var() > 1.0:
        return 1.0
    else:
        return 0.0
```

```{code-cell} python
btc_small.rolling("2d").apply(is_volatile)
```

````{admonition} Exercise
:name: pd-tim-dir5
See exercise 5 in the {ref}`exercise list <pd-tim-ex>`.
````

To make the optimal decision, we need to know the maximum
difference between the close price at the end of the window and the open
price at the start of the window.

````{admonition} Exercise
:name: pd-tim-dir6
See exercise 6 in the {ref}`exercise list <pd-tim-ex>`.
````

## Changing Frequencies: `.resample`

In addition to computing rolling statistics, we can also change the
frequency of the data.

For example, instead of a monthly moving average, suppose that we wanted
to compute the average *within* each calendar month.

We will use the `resample` method to do this.

Below are some examples.

```{code-cell} python
# business quarter
btc_usd.resample("BQ").mean()
```

Note that unlike with `rolling`, a single number is returned for
each column for each quarter.

The `resample` method will alter the frequency of the data and the
number of rows in the result will be different from the number of rows
in the input.

On the other hand, with `rolling`, the size and frequency of the result
are the same as the input.

We can sample at other frequencies and aggregate with multiple aggregations
function at once.

```{code-cell} python
# multiple functions at 2 start-of-quarter frequency
btc_usd.resample("2BQS").agg(["min", "max"])
```

As with `groupby` and `rolling`, you can also provide custom
functions to `.resample(...).agg` and `.resample(...).apply`

````{admonition} Exercise
:name: pd-tim-dir7
See exercise 7 in the {ref}`exercise list <pd-tim-ex>`.
````

To make the optimal decision we need to, for each month,
compute the maximum value of the close price on any day minus the open
price on the first day of the month.

````{admonition} Exercise
:name: pd-tim-dir8
See exercise 8 in the {ref}`exercise list <pd-tim-ex>`.
````

## Optional: API keys

Recall above that we had the line of code:

```{code-block} python
quandl.ApiConfig.api_key = "Dn6BtVoBhzuKTuyo6hbp"
```

This line told the `quandl` library that when obtaining making requests for data, it should use the *API key* `Dn6BtVoBhzuKTuyo6hbp`.

An API key is a sort of password that web services (like the Quandl API) require you to provide when you make requests.

Using this password, we were able to make a request to Quandl to obtain data directly from them.

The API key used here is one that we requested on behalf of this course.

If you plan to use Quandl more extensively, you should obtain your own personal API key from [their website](https://docs.quandl.com/docs#section-authentication) and re-run the `quandl.ApiConfig.api_key...` line of code with your new API key on the right-hand side.

(pd-tim-ex)=
## Exercises

### Exercise 1

By referring to table found at the link above, figure out the correct argument to
pass as `format` in order to parse the dates in the next three cells below.

Test your work by passing your format string to `pd.to_datetime`.

```{code-cell} python
christmas_str2 = "2017:12:25"
```

```{code-cell} python
dbacks_win = "M:11 D:4 Y:2001 9:15 PM"
```

```{code-cell} python
america_bday = "America was born on July 4, 1776"
```

({ref}`back to text <pd-tim-dir1>`)

### Exercise 2

Use `pd.to_datetime` to express the birthday of one of your friends
or family members as a `datetime` object.

Then use the `strftime` method to write a message of the format:

```{code-block} python
NAME's birthday is June 10, 1989 (a Saturday)
```

(where the name and date are replaced by the appropriate values)

({ref}`back to text <pd-tim-dir2>`)

### Exercise 3

For each item in the list, extract the specified data from `btc_usd`:

- July 2017 through August 2017 (inclusive)
- April 25, 2015 to June 10, 2016
- October 31, 2017

({ref}`back to text <pd-tim-dir3>`)

### Exercise 4

Using the `shift` function, determine the week with the largest percent change
in the volume of trades (the `"Volume (BTC)"` column).

Repeat the analysis at the bi-weekly and monthly frequencies.

```{hint}
We have data at a *daily* frequency and one week is `7` days.
```
```{hint}
Approximate a month by 30 days.
```

```{code-cell} python
# your code here
```

({ref}`back to text <pd-tim-dir4>`)

### Exercise 5

Imagine that you have access to the [DeLorean time machine](https://en.wikipedia.org/wiki/DeLorean_time_machine)
from "Back to the Future".

You are allowed to use the DeLorean only once, subject to the following
conditions:

- You may travel back to any day in the past.
- On that day, you may purchase one bitcoin *at market open*.
- You can then take the time machine 30 days into the future and sell your bitcoin *at market close*.
- Then you return to the present, pocketing the profits.

How would you pick the day?

Think carefully about what you would need to compute to make the
optimal choice. Try writing it out in the markdown cell below so you
have a clear description of the *want* operator that we will apply after
the exercise.

(Note: **Don't** look too far below, because in the next non-empty cell
we have written out our answer.)

To make this decision, we want to know ...

**Your answer here**

({ref}`back to text <pd-tim-dir5>`)

### Exercise 6

Do the following:

1. Write a pandas function that implements your strategy.
1. Pass it to the `agg` method of `rolling_btc`.
1. Extract the `"Open"` column from the result.
1. Find the date associated with the maximum value in that column.

How much money did you make? Compare with your neighbor.

```{code-cell} python
def daily_value(df):
    # DELETE `pass` below and replace it with your code
    pass

rolling_btc = btc_usd.rolling("30d")

# do steps 2-4 here
```

({ref}`back to text <pd-tim-dir6>`)

### Exercise 7

Now suppose you still have access to the DeLorean, but the conditions are
slightly different.

You may now:

- Travel back to the *first day* of any month in the past.
- On that day, you may purchase one bitcoin *at market open*.
- You can then travel to any day *in that month* and sell the bitcoin *at market close*.
- Then return to the present, pocketing the profits.

To which month would you travel? On which day of that month would you return
to sell the bitcoin?

Discuss with your neighbor what you would need to compute to make the
optimal choice. Try writing it out in the markdown cell below so you
have a clear description of the *want* operator that we will apply after
the exercise.

(Note: **Don't** look too many cells below, because we have written out
our answer.)

To make the optimal decision we need ...

**Your answer here**

({ref}`back to text <pd-tim-dir7>`)

### Exercise 8

Do the following:

1. Write a pandas function that implements your strategy.
1. Pass it to the `agg` method of `resampled_btc`.
1. Extract the `"Open"` column from the result.
1. Find the date associated with the maximum value in that column.

How much money did you make? Compare with your neighbor.

Was this strategy more profitable than the previous one? By how much?

```{code-cell} python
def monthly_value(df):
    # DELETE `pass` below and replace it with your code
    pass

resampled_btc = btc_usd.resample("MS")

# Do steps 2-4 here
```

({ref}`back to text <pd-tim-dir8>`)

