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

# Reshape

**Prerequisites**

- {doc}`pandas intro <intro>`
- {doc}`pandas basics <basics>`
- {doc}`Importance of index <the_index>`

**Outcomes**

- Understand and be able to apply the `melt`/`stack`/`unstack`/`pivot` methods
- Practice transformations of indices
- Understand tidy data

```{literalinclude} ../_static/colab_light.raw
```

```{code-cell} python
import numpy as np
import pandas as pd

%matplotlib inline
# activate plot theme
import qeds
qeds.themes.mpl_style();
```


## Tidy Data

While pushed more generally in the `R` language, the concept of "[tidy data](https://en.wikipedia.org/wiki/Tidy_data)" is helpful in understanding the
objectives for reshaping data, which in turn makes advanced features like
{doc}`groupby <groupby>` more seamless.

Hadley Wickham gives a terminology slightly better-adapted for the experimental
sciences, but nevertheless useful for the social sciences.

> A dataset is a collection of values, usually either numbers (if quantitative) or strings (if qualitative). Values are organized in two ways. Every value belongs to a variable and an observation. A variable contains all values that measure the same underlying attribute (like height, temperature, duration) across units. An observation contains all values measured on the same unit (like a person, or a day, or a race) across attributes. -- [Tidy Data (Journal of Statistical Software 2013)](https://www.jstatsoft.org/index.php/jss/article/view/v059i10/v59i10.pdf)

With this framing,

> A dataset is messy or tidy depending on how rows, columns and tables are
> matched with observations, variables, and types. In tidy data:
>
> 1.  Each variable forms a column.
>
> 2.  Each observation forms a row.
>
> 3.  Each type of observational unit forms a table.

The "column" and "row" terms map directly to pandas columns and rows, while the
"table" maps to a pandas DataFrame.

With this thinking and interpretation, it becomes essential to think through
what uniquely identifies an "observation" in your data.

Is it a country? A year? A combination of country and year?

These will become the indices of your DataFrame.

For those with more of a database background, the "tidy" format matches the
[3rd normal form](https://en.wikipedia.org/wiki/Third_normal_form) in
database theory, where the referential integrity of the database is maintained
by the uniqueness of the index.

When considering how to map this to the social sciences, note that
reshaping data can change what we consider to be the variable and
observation in a way that doesn't occur within the natural sciences.

For example, if the "observation" uniquely identified by a country and year and
the "variable" is GDP, you may wish to reshape it so that the "observable" is a
country, and the variables are a GDP for each year.

A word of caution: The tidy approach, where there is no redundancy and each
type of observational unit forms a table, is a good approach for storing data,
but you will frequently reshape/merge/etc. in order to make graphing or
analysis easier.  This doesn't break the tidy format since those examples are
ephemeral states used in analysis.

## Reshaping your Data

The data you receive is not always in a "shape" that makes it easy to analyze.

What do we mean by shape? The number of rows and columns in a
DataFrame and how information is stored in the index and column names.

This lecture will teach you the basic concepts of reshaping data.

As with other topics, we recommend reviewing the [pandas
documentation](https://pandas.pydata.org/pandas-docs/stable/reshaping.html)
on this subject for additional information.

We will keep our discussion here as brief and simple as possible because
these tools will reappear in subsequent lectures.

```{code-cell} python
url = "https://datascience.quantecon.org/assets/data/bball.csv"
bball = pd.read_csv(url)
bball.info()

bball
```

## Long vs Wide

Many of these operations change between long and wide DataFrames.

What does it mean for a DataFrame to be long or wide?

Here is long possible long-form representation of our basketball data.

```{code-cell} python
# Don't worry about what this command does -- We'll see it soon
bball_long = bball.melt(id_vars=["Year", "Player", "Team", "TeamName"])

bball_long
```

And here is a wide-form version.

```{code-cell} python
# Again, don't worry about this command... We'll see it soon too
bball_wide = bball_long.pivot_table(
    index="Year",
    columns=["Player", "variable", "Team"],
    values="value"
)
bball_wide
```

## `set_index`, `reset_index`, and Transpose

We have already seen a few basic methods for reshaping a
DataFrame.

- `set_index`: Move one or more columns into the index.
- `reset_index`: Move one or more index levels out of the index and make
  them either columns or drop from DataFrame.
- `T`: Swap row and column labels.

Sometimes, the simplest approach is the right approach.

Let's review them briefly.

```{code-cell} python
bball2 = bball.set_index(["Player", "Year"])
bball2.head()
```

```{code-cell} python
bball3 = bball2.T
bball3.head()
```

## `stack` and `unstack`

The `stack` and `unstack` methods operate directly on the index
and/or column labels.

### `stack`

`stack` is used to move certain levels of the column labels into the
index (i.e. moving from wide to long)

Let's take `ball_wide` as an example.

```{code-cell} python
bball_wide
```

Suppose that we want to be able to use the `mean` method to compute the
average value of each stat for each player, regardless of year or team.

To do that, we need two column levels: one for the player and one for the variable.

We can achieve this using the `stack` method.

```{code-cell} python
bball_wide.stack()
```

Now, we can compute the statistic we are after.

```{code-cell} python
player_stats = bball_wide.stack().mean()
player_stats
```

Now suppose instead of that we wanted to compute the average for each team and
stat, averaging over years and players.

We'd need to move the `Player` level down into the index so we are
left with column levels for Team and variable.

We can ask pandas do this using the `level` keyword argument.

```{code-cell} python
bball_wide.stack(level="Player")
```

Now we can compute the mean.

```{code-cell} python
bball_wide.stack(level="Player").mean()
```

Notice a few features of the `stack` method:

- Without any arguments, the `stack` arguments move the level of column
  labels closest to the data (also called inner-most or bottom level of labels)
  to become the index level closest to the data (also called the inner-most or
  right-most level of the index). In our example, this moved `Team` down from
  columns to the index.
- When we do pass a level, that level of column labels is moved down to the
  right-most level of the index and all other column labels stay in their
  relative position.

Note that we can also move multiple levels at a time in one call to `stack`.

```{code-cell} python
bball_wide.stack(level=["Player", "Team"])
```

In the example above, we started with one level on the index (just the year) and
stacked two levels to end up with a three-level index.

Notice that the two new index levels went closer to the data than the existing
level and that their order matched the order we passed in our list argument to
`level`.

### `unstack`

Now suppose that we wanted to see a bar chart of each player's stats.

This chart should have one "section" for each player and a different colored
bar for each variable.

As we'll learn in more detail in a later lecture,  we will
need to have the player's name on the index and the variables as columns to do this.

```{note}
In general, for a DataFrame, calling the `plot` method will put the index
on the horizontal (x) axis and make a new line/bar/etc. for each column.
```

Notice that we are close to that with the `player_stats` variable.

```{code-cell} python
player_stats
```

We now need to rotate the variable level of the index up to be column layers.

We use the `unstack` method for this.

```{code-cell} python
player_stats.unstack()
```

And we can make our plot!

```{code-cell} python
player_stats.unstack().plot.bar()
```

This particular visualization would be helpful if we wanted to see which stats
for which each player is strongest.

For example, we can see that Steph Curry scores far more points than he does
rebound, but Serge Ibaka is a bit more balanced.

What if we wanted to be able to compare all players for each statistic?

This would be easier to do if the bars were grouped by variable, with a
different bar for each player.

To plot this, we need to have the variables on the index and the player
name as column names.

We can get this DataFrame by setting `level="Player"` when calling `unstack`.

```{code-cell} python
player_stats.unstack(level="Player")
```

```{code-cell} python
player_stats.unstack(level="Player").plot.bar()
```

Now we can use the chart to make a number of statements about players:

- Ibaka does not get many assists, compared to Curry and Durant.
- Steph and Kevin Durant are both high scorers.

Based on the examples above, notice a few things about `unstack`:

- It is the *inverse* of `stack`; `stack` will move labels down
  from columns to index, while `unstack` moves them up from index to columns.
- By default, `unstack` will move the level of the index closest to the data
  and place it in the column labels closest to the data.

```{note}
Just as we can pass multiple levels to `stack`, we can also pass multiple
levels to `unstack`.

We needed to use this in our solution to the exercise below.
```

````{admonition} Exercise
:name: pd-shp-dir1
See exercise 1 in the {ref}`exercise list <pd-shp-ex>`.
````

### Summary

In some ways `set_index`, `reset_index`, `stack`, and `unstack`
are the "most fundamental" reshaping operations...

The other operations we discuss can be formulated with these
four operations (and, in fact, some of them are exactly written as these
operations in `pandas`'s code base).

*Pro tip*: We remember stack vs unstack with a mnemonic: **U**nstack moves index
levels **U**p

## `melt`

The `melt` method is used to move from wide to long form.

It can be used to move all of the "values" stored in your DataFrame to a
single column with all other columns being used to contain identifying
information.

**Warning**: When you use `melt`, any index that you currently have
will be deleted.

We saw used `melt` above when we constructed `bball_long`:

```{code-cell} python
bball
```

```{code-cell} python
# this is how we made ``bball_long``
bball.melt(id_vars=["Year", "Player", "Team", "TeamName"])
```

Notice that the columns we specified as `id_vars` remained columns, but all
other columns were put into two new columns:

1. `variable`: This has dtype string and contains the former column names.
   as values
1. `value`: This has the former values.

Using this method is an effective way to get our data in *tidy* form as noted
above.

````{admonition} Exercise
:name: pd-shp-dir2
See exercise 2 in the {ref}`exercise list <pd-shp-ex>`.
````

## `pivot` and `pivot_table`

The next two reshaping methods that we will use are closely related.

Some of you might even already be familiar with these ideas because you
have previously used *pivot tables* in Excel.

- If so, good news. We think this is even more powerful than Excel
  and easier to use!
- If not, good news. You are about to learn a very powerful and user-friendly tool.

We will begin with `pivot`.

The `pivot` method:

- Takes the unique values of one column and places them along the index.
- Takes the unique values of another column and places them along the
  columns.
- Takes the values that correspond to a third column and fills in the
  DataFrame values that correspond to that index/column pair.

We'll illustrate with an example.

```{code-cell} python
# .head 8 excludes Ibaka -- will discuss why later
bball.head(6).pivot(index="Year", columns="Player", values="Pts")
```

We can replicate `pivot` using three of the fundamental operations
from above:

1. Call `set_index` with the `index` and `columns` arguments
1. Extract the `values` column
1. `unstack` the columns level of the new index

```{code-cell} python
#  1---------------------------------------  2---  3----------------------
bball.head(6).set_index(["Year", "Player"])["Pts"].unstack(level="Player")
```

One important thing to be aware of is that in order for `pivot` to
work, the index/column pairs must be *unique*!

Below, we demonstrate the error that occurs when they are not unique.

```{code-cell} python
---
tags: [raises-exception]
---
# Ibaka shows up twice in 2016 because he was traded mid-season from
# the Orlando Magic to the Toronto Raptors
bball.pivot(index="Year", columns="Player", values="Pts")
```

### `pivot_table`

The `pivot_table` method is a generalization of `pivot`.

It overcomes two limitations of `pivot`:

1. It allows you to choose multiple columns for the index/columns/values
   arguments.
1. It allows you to deal with duplicate entries by
   having you choose how to combine them.

```{code-cell} python
bball
```

Notice that we can replicate the functionality of `pivot` if we pass
the same arguments.

```{code-cell} python
bball.head(6).pivot(index="Year", columns="Player", values="Pts")
```

But we can also choose multiple columns to be used in
index/columns/values.

```{code-cell} python
bball.pivot_table(index=["Year", "Team"], columns="Player", values="Pts")
```

```{code-cell} python
bball.pivot_table(index="Year", columns=["Player", "Team"], values="Pts")
```

AND we can deal with duplicated index/column pairs.

```{code-cell} python
# This produced an error
# bball.pivot(index="Year", columns="Player", values="Pts")

# This doesn't!
bball_pivoted = bball.pivot_table(index="Year", columns="Player", values="Pts")
bball_pivoted
```

`pivot_table` handles duplicate index/column pairs using an aggregation.

By default, the aggregation is the mean.

For example, our duplicated index/column pair is `("x", 1)` and had
associated values of 2 and 5.

Notice that `bball_pivoted.loc[2016, "Ibaka"]` is `(15.1 + 14.2)/2 = 14.65`.

We can choose how `pandas` aggregates all of the values.

For example, here's how we would keep the max.

```{code-cell} python
bball.pivot_table(index="Year", columns="Player", values="Pts", aggfunc=max)
```

Maybe we wanted to count how many values there were.

```{code-cell} python
bball.pivot_table(index="Year", columns="Player", values="Pts", aggfunc=len)
```

We can even pass multiple aggregation functions!

```{code-cell} python
bball.pivot_table(index="Year", columns="Player", values="Pts", aggfunc=[max, len])
```

````{admonition} Exercise
:name: pd-shp-dir3
See exercise 3 in the {ref}`exercise list <pd-shp-ex>`.
````

## Visualizing Reshaping

Now that you have learned the basics and had a chance to experiment,
we will use some generic data to provide a visualization of what the above
reshape operations do.

The data we will use is:

```{code-cell} python
# made up
# columns A and B are "identifiers" while C, D, and E are variables.
df = pd.DataFrame({
    "A": [0, 0, 1, 1],
    "B": "x y x z".split(),
    "C": [1, 2, 1, 4],
    "D": [10, 20, 30, 20,],
    "E": [2, 1, 5, 4,]
})

df.info()
df
```

```{code-cell} python
df2 = df.set_index(["A", "B"])
df2.head()
```

```{code-cell} python
df3 = df2.T
df3.head()
```

### `stack` and `unstack`

Below is an animation that shows how stacking works.

```{figure} https://datascience.quantecon.org/assets/_static/reshape_files/stack.gif
:alt: stack.gif
```

```{code-cell} python
df2
```

```{code-cell} python
df2_stack = df2.stack()
df2_stack
```

And here is an animation that shows how unstacking works.

```{figure} https://datascience.quantecon.org/assets/_static/reshape_files/unstack_level0.gif
:alt: unstack\_level0.gif
```

```{code-cell} python
df2
```

```{code-cell} python
df2.unstack()
```

### `melt`

As noted above, the `melt` method transforms data from wide to long in form.

Here's a visualization of that operation.

```{figure} https://datascience.quantecon.org/assets/_static/reshape_files/melt.gif
:alt: melt.gif
```

```{code-cell} python
df
```

```{code-cell} python
df_melted = df.melt(id_vars=["A", "B"])
df_melted
```

(pd-shp-ex)=
## Exercises

### Exercise 1

(*Warning*: This one is challenging):

Recall the `bball_wide` DataFrame from above (repeated below to jog
your memory).

In this task, you will start from `ball` and re-recreate `bball_wide`
by combining the operations we just learned about.

There are many ways to do this, so be creative.

Our solution used `set_index`, `T`, `stack`, and `unstack` in
that order.

Here are a few hints:

- ```{hint}
  Think about what columns you will need to call `set_index` on so
  that their data ends up as labels (either in index or columns).
  ```
- ```{hint}
  Leave other columns (e.g. the actual game stats) as actual columns so
  their data can stay data during your reshaping.
  ```

Don't spend too much time on this... if you get stuck, open up **this**
markdown cell, and you will see our answer hidden.

```{hint}
You might need to add `.sort_index(axis=1)` after you are
finished to get the columns in the same order.
```

```{hint}
You may not end up with a `variable` header on the second
level of column labels. This is ok.
```

```{raw} html
<div style="display: none;">
`bball.drop("TeamName", axis=1).set_index(["Year", "Player", "Team"]).stack().unstack(level=[1, 3, 2]).sort_index(axis=1)`
</div>
```

```{code-cell} python
:tags: ["remove-output"]
bball_wide
```

({ref}`back to text <pd-shp-dir1>`)

### Exercise 2

- What do you think would happen if we wrote `bball.melt(id_vars=["Year", "Player"])`
  rather than `bball.melt(id_vars=["Year", "Player", "Team", "TeamName"])`?
  Were you right? Write your thoughts.
- Read the documentation and focus on the argument `value_vars`. How
  does `bball.melt(id_vars=["Year", "Player"], value_vars=["Pts", "Rebound"])`
  differ from `bball.melt(id_vars=["Year", "Player"])`?
- Consider the differences between `bball.stack` and `bball.melt`.
  Is there a way to make them generate the same output?
  Write your thoughts.
  - ```{hint}
    You might need to use both `stack` and another method from
    above
    ```

({ref}`back to text <pd-shp-dir2>`)

### Exercise 3

- First, take a breath... That was a lot to take in.
- Can you think of a reason to ever use `pivot` rather than
  `pivot_table`? Write your thoughts.
- Create a pivot table with column `Player` as the index, `TeamName` as the
  columns, and `[Rebound, Assist]` as the values. What happens when you use
  `aggfunc=[np.max, np.min, len]`? Describe how Python produced
  each of the values in the resultant pivot table.

({ref}`back to text <pd-shp-dir3>`)
