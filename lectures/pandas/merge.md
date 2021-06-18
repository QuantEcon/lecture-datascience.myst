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

# Merge

**Prerequisites**

- {doc}`Reshape <reshape>`

**Outcomes**

- Know the different pandas routines for combining datasets
- Know when to use `pd.concat` vs `pd.merge` vs `pd.join`
- Be able to apply the three main combining routines

**Data**

- WDI data on GDP components, population, and square miles of countries
- Book ratings: 6,000,000 ratings for the 10,000 most rated books on
  [Goodreads](https://www.goodreads.com/)
- Details for all delayed US domestic flights in November 2016,
  obtained from the [Bureau of Transportation
  Statistics](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)


```{literalinclude} ../_static/colab_light.raw
```

## Combining Datasets

Often, we will want perform joint analysis on data from different sources.

For example, when analyzing the regional sales for a company, we might
want to include industry aggregates or demographic information for each
region.

Or perhaps we are working with product-level data, have a list of
product groups in a separate dataset, and want to compute aggregate
statistics for each group.

```{code-cell} python
import pandas as pd
import qeds
%matplotlib inline
# activate plot theme
import qeds
qeds.themes.mpl_style();

from IPython.display import display
```

```{code-cell} python
# from WDI. Units trillions of 2010 USD
url = "https://raw.githubusercontent.com/QuantEcon/lecture-datascience.myst/main/lectures/pandas/wdi_data.csv"
wdi = pd.read_csv(url).set_index(["country", "year"])
wdi.info()

wdi2017 = wdi.xs(2017, level="year")
wdi2017
```

```{code-cell} python
wdi2016_17 = wdi.loc[pd.IndexSlice[:, [2016, 2017]],: ]
wdi2016_17
```

```{code-cell} python
# Data from https://www.nationmaster.com/country-info/stats/Geography/Land-area/Square-miles
# units -- millions of square miles
sq_miles = pd.Series({
   "United States": 3.8,
   "Canada": 3.8,
   "Germany": 0.137,
   "United Kingdom": 0.0936,
   "Russia": 6.6,
}, name="sq_miles").to_frame()
sq_miles.index.name = "country"
sq_miles
```

```{code-cell} python
# from WDI. Units millions of people
pop_url = "https://raw.githubusercontent.com/QuantEcon/lecture-datascience.myst/main/lectures/pandas/wdi_population.csv"
pop = pd.read_csv(pop_url).set_index(["country", "year"])
pop.info()
pop.head(10)
```

Suppose that we were asked to compute a number of statistics with the data above:

- As a measure of land usage or productivity, what is Consumption per square mile?
- What is GDP per capita (per person) for each country in each year? How about
  Consumption per person?
- What is the population density of each country? How much does it change over time?

Notice that to answer any of the questions from above, we will have to use data
from more than one of our DataFrames.

In this lecture, we will learn many techniques for combining datasets that
originate from different sources, careful to ensure that data is properly
aligned.

In pandas three main methods can combine datasets:

1. `pd.concat([dfs...])`
1. `pd.merge(df1, df2)`
1. `df1.join(df2)`

We'll look at each one.

## `pd.concat`

The `pd.concat` function is used to stack two or more DataFrames
together.

An example of when you might want to do this is if you have monthly data
in separate files on your computer and would like to have 1 year of data
in a single DataFrame.

The first argument to `pd.concat` is a list of DataFrames to be
stitched together.

The other commonly used argument is named `axis`.

As we have seen before, many pandas functions have an `axis` argument
that specifies whether a particular operation should happen down rows
(`axis=0`) or along columns (`axis=1`).

In the context of `pd.concat`, setting `axis=0` (the default case)
will stack DataFrames on top of one another while `axis=1` stacks them
side by side.

We'll look at each case separately.

### `axis=0`

When we call `pd.concat` and set `axis=0`, the list of DataFrames
passed in the first argument will be stacked on top of one another.

Let's try it out here.

```{code-cell} python
# equivalent to pd.concat([wdi2017, sq_miles]) -- axis=0 is default
pd.concat([wdi2017, sq_miles], axis=0)
```

Notice a few things:

The number of rows in the output is the total number
: of rows in all inputs. The labels are all from the original
  DataFrames.

- The column labels are all the distinct column labels from all the inputs.
- For columns that appeared only in one input, the value for all row labels
  originating from a different input is equal to `NaN` (marked as missing).

### `axis=1`

In this example, concatenating by stacking
side-by-side makes more sense.

We accomplish this by passing `axis=1` to `pd.concat`:

```{code-cell} python
pd.concat([wdi2017, sq_miles], axis=1)
```

Notice here that

- The index entries are all unique index entries that appeared in any DataFrame.
- The column labels are all column labels from the inputs.
- As `wdi2017` didn't have a `Russia` row, the value for all of its columns
  is `NaN`.

Now we can answer one of our questions from above: What is
Consumption per square mile?

```{code-cell} python
temp = pd.concat([wdi2017, sq_miles], axis=1)
temp["Consumption"] / temp["sq_miles"]
```

## `pd.merge`

`pd.merge` operates on two DataFrames at a time and is primarily used
to bring columns from one DataFrame into another, *aligning* data based
on one or more "key" columns.

This is a somewhat difficult concept to grasp by reading, so let's look at some
examples.

```{code-cell} python
pd.merge(wdi2017, sq_miles, on="country")
```

The output here looks very similar to what we saw with `concat` and
`axis=1`, except that the row for `Russia` does not appear.

We will talk more about why this happened soon.

For now, let's look at a slightly more intriguing example:

```{code-cell} python
pd.merge(wdi2016_17, sq_miles, on="country")
```

Here's how we think about what happened:

- The data in `wdi2016_17` is copied over exactly as is.
- Because `country` was on the index for both DataFrames, it is on the
  index of the output.
- We lost the year on the index -- we'll work on getting it back below.
- The additional column in `sq_miles` was added to column labels for the
  output.
- The data from the `sq_miles` column was added to the output by looking up
  rows where the `country` in the two DataFrames lined up.
  -  Note that all the countries appeared twice, and the data in `sq_miles` was repeated. This is because `wdi2016_17` had two rows for each country.
  -  Also note that because `Russia` did not appear in `wdi2016_17`, the value `sq_miles.loc["Russia"]` (i.e. `6.6`) is not used the output.

How do we get the year back?

We must first call `reset_index` on `wdi2016_17` so
that in the first step when all columns are copied over, `year` is included.

```{code-cell} python
pd.merge(wdi2016_17.reset_index(), sq_miles, on="country")
```

### Multiple Columns

Sometimes, we need to merge multiple columns.

For example our `pop` and `wdi2016_17` DataFrames both have observations
organized by country and year.

To properly merge these datasets, we would need to align the data by
both country and year.

We pass a list to the `on` argument to accomplish this:

```{code-cell} python
pd.merge(wdi2016_17, pop, on=["country", "year"])
```

Now, we can answer more of our questions from above: What is GDP per capita (per
person) for each country in each year? How about Consumption per person?

```{code-cell} python
wdi_pop = pd.merge(wdi2016_17, pop, on=["country", "year"])
wdi_pop["GDP"] / wdi_pop["Population"]
```

```{code-cell} python
wdi_pop["Consumption"] / wdi_pop["Population"]
```

````{admonition} Exercise
:name: pd-mrg-dir1
See exercise 1 in the {ref}`exercise list <pd-mrg-ex>`.
````

## Arguments to `merge`

The `pd.merge` function can take many optional arguments.

We'll talk about a few of the most commonly-used ones here and refer you
to the
[documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html#pandas.merge)
for more details.

We'll follow the pandas convention and refer to the first argument to
`pd.merge` as `left` and call the second `right`.

### `on`

We have already seen this one used before, but we want to point out that on
is optional.

If nothing is given for this argument, pandas will use **all** columns
in `left` and `right` with the same name.

In our example, `country` is the only column that appears in both
DataFrames, so it is used for `on` if we don't pass anything.

The following two are equivalent.

```{code-cell} python
pd.merge(wdi2017, sq_miles, on="country")
```

```{code-cell} python
# if we move index back to columns, the `on` is un-necessary
pd.merge(wdi2017.reset_index(), sq_miles.reset_index())
```

### `left_on`, `right_on`

Above, we used the `on` argument to identify a column in both `left`
and `right` that was used to align data.

Sometimes, both DataFrames don't have the same name for this column.

In that case, we use the `left_on` and `right_on` arguments, passing
the proper column name(s) to align the data.

We'll show you an example below, but it is somewhat silly as our
DataFrames do both have the `country` column.

```{code-cell} python
pd.merge(wdi2017, sq_miles, left_on="country", right_on="country")
```

### `left_index`, `right_index`

Sometimes, as in our example, the key used to align data is actually in the
index instead of one of the columns.

In this case, we can use the `left_index` or `right_index` arguments.

We should only set these values to a boolean (`True` or `False`).

Let's practice with this.

```{code-cell} python
pd.merge(wdi2017, sq_miles, left_on="country", right_index=True)
```

### `how`

The `how` is perhaps the most powerful, but most conceptually
difficult of the arguments we will cover.

This argument controls which values from the key column(s) appear in the
output.

The 4 possible options for this argument are summarized in
the image below.

```{figure} ../_static/merge_venns.png
:alt: merge\_venns.png
```

In words, we have:

- `left`: Default and what we described above. It uses
  the keys from the `left` DataFrame.
- `right`: Output will contain all keys from `right`.
- `inner`: The output will only contain keys that appear in *both*
  `left` and `right`.
- `outer`: The output will contain any key found in either `left`
  or `right`.

In addition to the above, we will use the following two DataFrames to
illustrate the `how` option.

```{code-cell} python
wdi2017_no_US = wdi2017.drop("United States")
wdi2017_no_US
```

```{code-cell} python
sq_miles_no_germany = sq_miles.drop("Germany")
sq_miles_no_germany
```

Now, let's see all the possible `how` options.

```{code-cell} python
# default
pd.merge(wdi2017, sq_miles, on="country", how="left")
```

```{code-cell} python
# notice ``Russia`` is included
pd.merge(wdi2017, sq_miles, on="country", how="right")
```

```{code-cell} python
# notice no United States or Russia
pd.merge(wdi2017_no_US, sq_miles, on="country", how="inner")
```

```{code-cell} python
# includes all 5, even though they don't all appear in either DataFrame
pd.merge(wdi2017_no_US, sq_miles_no_germany, on="country", how="outer")
```

````{admonition} Exercise
:name: pd-mrg-dir2
See exercise 2 in the {ref}`exercise list <pd-mrg-ex>`.
````
````{admonition} Exercise
:name: pd-mrg-dir3
See exercise 3 in the {ref}`exercise list <pd-mrg-ex>`.
````

### `df.merge(df2)`

Note that the DataFrame type has a `merge` *method*.

It is the same as the function we have been working with, but passes the
DataFrame before the period as `left`.

Thus `df.merge(other)` is equivalent to `pd.merge(df, other)`.

```{code-cell} python
wdi2017.merge(sq_miles, on="country", how="right")
```

## `df.join`

The `join` method for a DataFrame is very similar to the `merge`
method described above, but only allows you to use the index of the
`right` DataFrame as the join key.

Thus, `left.join(right, on="country")` is equivalent to calling
`pd.merge(left, right, left_on="country", right_index=True)`.

The implementation of the `join` method calls `merge` internally,
but sets the `left_on` and `right_index` arguments for you.

You can do anything with `df.join` that you can do with
`df.merge`, but df.join` is more convenient to use if the keys of `right`
are in the index.

```{code-cell} python
wdi2017.join(sq_miles, on="country")
```

```{code-cell} python
wdi2017.merge(sq_miles, left_on="country", right_index=True)
```

## Case Study

Let's put these tools to practice by loading some real datasets and
seeing how these functions can be applied.

We'll analyze ratings of books from the website [Goodreads](https://www.goodreads.com/).

We accessed the data [here](https://github.com/zygmuntz/goodbooks-10k).

Let's load it up.

```{code-cell} python
ratings = qeds.data.load("goodreads_ratings")
display(ratings.head())
ratings.info()
```

We can already do some interesting things with just the ratings data.

Let's see how many ratings of each number are in our dataset.

```{code-cell} python
ratings["rating"].value_counts().sort_index().plot(kind="bar");
```

Let's also see how many users have rated `N` books, for all `N`
possible.

To do this, we will use `value_counts` twice (can you think of why?).

We will see a more flexible way of performing similar grouped operations in
a future lecture.

```{code-cell} python
users_by_n = (
    ratings["user_id"]
    .value_counts()  # Series. Index: user_id, value: n ratings by user
    .value_counts()  # Series. Index: n_ratings by user, value: N_users with this many ratings
    .sort_index()    # Sort our Series by the index (number of ratings)
    .reset_index()   # Dataframe with columns `index` (from above) and `user_id`
    .rename(columns={"index": "N_ratings", "user_id": "N_users"})
)
users_by_n.head(10)
```

Let's look at some statistics on that dataset.

```{code-cell} python
users_by_n.describe()
```

We can see the same data visually in a box plot.

```{code-cell} python
users_by_n.plot(kind="box", subplots=True)
```

Let's practice applying the want operator...

**Want**: Determine whether a relationship between the number of
ratings a user has written and the distribution of the ratings exists. (Maybe we
are an author hoping to inflate our ratings and wonder if we should
target "more experienced" Goodreads users, or focus on newcomers.)

Let's start from the result and work our way backwards:

1. We can answer our question if we have two similar DataFrames:
    - All ratings by the `N` (e.g. 5) users with the most ratings
    - All ratings by the `N` users with the least number of
      ratings
1. To get that, we will need to extract rows of `ratings` with
   `user_id` associated with the `N` most and least prolific raters
1. For that, we need the most and least active `user_id`s
1. To get that info, we need a count of how many ratings each user left.
    - We can get that with `df["user_id"].value_counts()`, so let's
      start there.

```{code-cell} python
# step 4
n_ratings = ratings["user_id"].value_counts()
n_ratings.head()
```

```{code-cell} python
# step 3
N = 5
most_prolific_users = n_ratings.nlargest(5).index.tolist()
least_prolific_users = n_ratings.nsmallest(5).index.tolist()
```

```{code-cell} python
# step 2
active_ratings = ratings.loc[ratings["user_id"].isin(most_prolific_users), :]
inactive_ratings = ratings.loc[ratings["user_id"].isin(least_prolific_users), :]
```

```{code-cell} python
# step 1 -- get the answer!
active_ratings["rating"].value_counts().sort_index().plot(
    kind="bar", title="Distribution of ratings by most active users"
)
```

```{code-cell} python
inactive_ratings["rating"].value_counts().sort_index().plot(
    kind="bar", title="Distribution of ratings by least active users"
)
```

Nice! From the picture above, the new users look much more
likely to leave 5 star ratings than more experienced users.

### Book Data

We know what you are probably thinking: "Isn't this a lecture on merging?
Why are we only using one dataset?"

We hear you.

Let's also load a dataset containing information on the actual books.

```{code-cell} python
books = qeds.data.load("goodreads_books")
# we only need a few of the columns
books = books[["book_id", "authors", "title"]]
print("shape: ", books.shape)
print("dtypes:\n", books.dtypes, sep="")
books.head()
```

We could do similar interesting things with just the books dataset,
but we will skip it for now and merge them together.

```{code-cell} python
rated_books = pd.merge(ratings, books)
```

Now, let's see which books have been most often rated.

```{code-cell} python
most_rated_books_id = rated_books["book_id"].value_counts().nlargest(10).index
most_rated_books = rated_books.loc[rated_books["book_id"].isin(most_rated_books_id), :]
list(most_rated_books["title"].unique())
```

Let's use our `pivot_table` knowledge to compute the average rating
for each of these books.

```{code-cell} python
most_rated_books.pivot_table(values="rating", index="title")
```

These ratings seem surprisingly low, given that they are the most often
rated books on Goodreads.

I wonder what the bottom of the distribution looks like...

````{admonition} Exercise
:name: pd-mrg-dir4
See exercise 4 in the {ref}`exercise list <pd-mrg-ex>`.
````

Let's compute the average number of ratings for each book in our sample.

```{code-cell} python
average_ratings = (
    rated_books
    .pivot_table(values="rating", index="title")
    .sort_values(by="rating", ascending=False)
)
average_ratings.head(10)
```

What does the overall distribution of average ratings look like?

```{code-cell} python
# plot a kernel density estimate of average ratings
average_ratings.plot.density(xlim=(1, 5))

# or a histogram
average_ratings.plot.hist(bins=30, xlim=(1, 5))
```

It looks like most books have an average rating of just below 4.

## Extra Example: Airline Delays

Let's look at one more example.

This time, we will use a dataset from the [Bureau of Transportation
Statistics](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)
that describes the cause of all US domestic flight delays
in November 2016:

```{code-cell} python
air_perf = qeds.load("airline_performance_dec16")[["CRSDepTime", "Carrier", "CarrierDelay", "ArrDelay"]]
air_perf.info()
air_perf.head()
```

The `Carrier` column identifies the airline and the `CarrierDelay`
reports the number of minutes of the total delay assigned as the
"carrier's fault".

**Want**: Determine which airlines, on average, contribute most to
delays.

We can do this using `pivot_table`:

```{code-cell} python
avg_delays = (
    air_perf
    .pivot_table(index="Carrier", values="CarrierDelay", aggfunc="mean")
    .sort_values("CarrierDelay")
    .nlargest(10, "CarrierDelay")
)
avg_delays
```

The one issue with this dataset is that we don't know what all those two
letter carrier codes are!

Thankfully, we have a second dataset that maps the two letter code
into the full airline name.

```{code-cell} python
carrier_code = qeds.load("airline_carrier_codes")
carrier_code.tail()
```

Let's merge these names so we know which airlines we should avoid
flying...

```{code-cell} python
avg_delays_w_code = avg_delays.join(carrier_code)
avg_delays_w_code.sort_values("CarrierDelay", ascending=False)
```

Based on that information, which airlines would you avoid near
the holidays?

## Visualizing Merge Operations

As we did in the {doc}`reshape lecture <reshape>`, we will visualize the
various merge operations using artificial DataFrames.

First, we create some dummy DataFrames.

```{code-cell} python
dfL = pd.DataFrame(
    {"Key": ["A", "B", "A", "C"], "C1":[1, 2, 3, 4], "C2": [10, 20, 30, 40]},
    index=["L1", "L2", "L3", "L4"]
)[["Key", "C1", "C2"]]

print("This is dfL: ")
display(dfL)

dfR = pd.DataFrame(
    {"Key": ["A", "B", "C", "D"], "C3": [100, 200, 300, 400]},
    index=["R1", "R2", "R3", "R4"]
)[["Key", "C3"]]

print("This is dfR:")
display(dfR)
```

### `pd.concat`

Recall that calling `pd.concat(..., axis=0)` will stack DataFrames on top of
one another:

```{code-cell} python
pd.concat([dfL, dfR], axis=0)
```

Here's how we might visualize that.

```{figure} ../_static/concat_axis0.gif
:alt: concat\_axis0.gif
```

We can also set `axis=1` to stack side by side.

```{code-cell} python
pd.concat([dfL, dfR], axis=1)
```

Here's how we might visualize that.

```{figure} ../_static/concat_axis1.gif
:alt: concat\_axis1.gif
```

### `pd.merge`

The animation below shows a visualization of what happens when we call

```{code-cell} python
pd.merge(dfL, dfR, on="Key")
```

```{figure} ../_static/left_merge.gif
:alt: left\_merge.gif
```

Now, let's focus on what happens when we set `how="right"`.

Pay special attention to what happens when filling the output value for
the key `A`.

```{code-cell} python
pd.merge(dfL, dfR, on="Key", how="right")
```

```{figure} ../_static/right_merge.gif
:alt: right\_merge.gif
```

### Exercises With Artificial Data
````{admonition} Exercise
:name: pd-mrg-dir5
See exercise 5 in the {ref}`exercise list <pd-mrg-ex>`.
````
````{admonition} Exercise
:name: pd-mrg-dir6
See exercise 6 in the {ref}`exercise list <pd-mrg-ex>`.
````
````{admonition} Exercise
:name: pd-mrg-dir7
See exercise 7 in the {ref}`exercise list <pd-mrg-ex>`.
````

(pd-mrg-ex)=
## Exercises

### Exercise 1

Use your new `merge` skills to answer the final question from above: What
is the population density of each country? How much does it change over
time?

```{code-cell} python
# your code here
```

({ref}`back to text <pd-mrg-dir1>`)

### Exercise 2

Compare the `how="left"` with `how="inner"` options using the
DataFrames `wdi2017_no_US` and `sq_miles_no_germany`.

Are the different? How?

Will this happen for all pairs of DataFrames, or are `wdi2017_no_US` and
`sq_miles_no_germany` special in some way?

Also compare `how="right"` and `how="outer"` and answer the same
questions.

```{code-cell} python
# your code here
```

({ref}`back to text <pd-mrg-dir2>`)

### Exercise 3

Can you pick the correct argument for `how` such that `pd.merge(wdi2017, sq_miles,
how="left")` is equal to `pd.merge(sq_miles, wdi2017, how=XXX)`?

```{code-cell} python
# your code here
```

({ref}`back to text <pd-mrg-dir3>`)

### Exercise 4

Repeat the analysis above to determine the average rating for the books with the
*least* number ratings.

Is there a distinguishable difference in the average rating compared to
the most rated books?

Did you recognize any of the books?

```{code-cell} python
# your code here
```

({ref}`back to text <pd-mrg-dir4>`)

### Exercise 5

In writing, describe what the output looks like when you do
`pd.concat([dfL, dfR], axis=1)` (see above and/or run the cell below).

Be sure to describe things like:

- What are the columns? What about columns with the same name?
- What is the index?
- Do any `NaN`s get introduced? If so, where? Why?

```{code-cell} python
:tags: ["remove-output"]
pd.concat([dfL, dfR], axis=1)
```

({ref}`back to text <pd-mrg-dir5>`)

### Exercise 6

Determine what happens when you run each of the two cells below.

For each cell, answer the list of questions from the previous exercise.

```{code-cell} python
:tags: ["remove-output"]
# First code cell for above exercise
pd.concat([dfL, dfL], axis=0)
```

```{code-cell} python
:tags: ["remove-output"]
# Second code cell for above exercise
pd.concat([dfR, dfR], axis=1)
```

({ref}`back to text <pd-mrg-dir6>`)

### Exercise 7

Describe in words why the output of `pd.merge(dfL, dfR,
how="right")` has more rows than either `dfL` or `dfR`.

Run the cell below to see the output of that operation.

```{code-cell} python
:tags: ["remove-output"]
pd.merge(dfL, dfR, how="right")
```

({ref}`back to text <pd-mrg-dir7>`)
