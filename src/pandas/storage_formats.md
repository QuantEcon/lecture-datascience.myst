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

# Storage Formats

**Prerequisites**

- {doc}`Intro to DataFrames and Series <intro>`

**Outcomes**

- Understand that data can be saved in various formats
- Know where to get help on file input and output
- Know when to use csv, xlsx, feather, and sql formats

**Data**

- Results for all NFL games between September 1920 to February 2017


```{literalinclude} _static/colab_light.raw
```

```{code-cell} python
import pandas as pd
import numpy as np
```

## File Formats

Data can be saved in a variety of formats.

pandas understands how to write and read DataFrames to and from many of
these formats.

We defer to the [official
documentation](https://pandas.pydata.org/pandas-docs/stable/io.html)
for a full description of how to interact with all the file formats, but
will briefly discuss a few of them here.

### CSV

**What is it?** CSVs store data as plain text (strings) where each row is a
line and columns are separated by `,`.

**Pros**

- Widely used (you should be familiar with it)
- Plain text file (can open on any computer, "future proof")
- Can be read from and written to by most data software

**Cons**

- Not the most efficient way to store or access
- No formal standard, so there is room for user interpretation on how to
  handle edge cases (e.g. what to do about a data field that itself includes
  a comma)

**When to use**:

- A great default option for most use cases

### xlsx

**What is it?** xlsx is a binary file format used as Excel's default.

**Pros**:

- Standard format in many industries
- Easy to share with colleagues that use Excel

**Cons**:

- Quite slow to read/write large amounts of data
- Stores both data and *metadata* like styling and display information
  and even plots. This metadata is not always portable to other file formats
  or programs.

**When to use**:

- When sharing data with Excel
- When you would like special formatting to be applied to the
  spreadsheet when viewed in Excel

### Parquet

**What is it?** Parquet is a custom binary format designed for efficient reading and
writing of data stored in columns.

**Pros**:

- *Very* fast
- Naturally understands all `dtypes` used by pandas, including
  multi-index DataFrames
- Very common in "big data" systems like Hadoop or Spark
- Supports various compression algorithms

**Cons**:

- Binary storage format that is not human-readable

**When to use**:

- If you have "not small" amounts (> 100 MB) of unchanging data that
  you want to read quickly
- If you want to store data in an size-and-time-efficient way that may
  be accessed by external systems

### Feather

**What is it?** Feather is a custom binary format designed for efficient reading and
writing of data stored in columns.

**Pros**:

- *Very* fast -- even faster than parquet
- Naturally understands all `dtypes` used by pandas

**Cons**:

- Can only read and write from Python and a handful of other
  programming languages
- New file format (introduced in March '16), so most files don't come
  in this format
- Only supports standard pandas index, so you need to `reset_index`
  before saving and then `set_index` after loading

**When to use**:

- Use as an alternative to Parquet if you need the absolute best read and write
  speeds for unchanging datasets
- Only use when you will not need to access the data in a programming language
  or software outside of Python, R, and Julia

### SQL

**What is it?** SQL is a language used to interact with relational
databases... [more info](https://en.wikipedia.org/wiki/SQL)

**Pros**:

- Well established industry standard for handling data
- Much of the world's data is in a SQL database somewhere

**Cons**:

- Complicated: to have full control you need to learn another language
  (SQL)

**When to use**:

- When reading from or writing to existing SQL databases

**NOTE**: We can cover interacting with SQL databases in a dedicated
lecture -- contact us for more information.

## Writing DataFrames

Let's now talk about saving a DataFrame to a file.

As a general rule of thumb, if we have a DataFrame `df` and we would
like to save to save it as a file of type `FOO`, then we would call
the method named `df.to_FOO(...)`.

We will show you how this can be done and try to highlight some of the
items mentioned above.

But, we will not cover all possible options and features — we feel
it is best to learn these as you need them by consulting the appropriate
documentation.

First, we need some DataFrames to save. Let's make them now.

Note that by default `df2` will be approximately 10 MB.

If you need to change this number, adjust the value of
the `wanted_mb` variable below.

```{code-cell} python
np.random.seed(42)  # makes sure we get the same random numbers each time
df1 = pd.DataFrame(
    np.random.randint(0, 100, size=(10, 4)),
    columns=["a", "b", "c", "d"]
)

wanted_mb = 10  # CHANGE THIS LINE
nrow = 100000
ncol = int(((wanted_mb * 1024**2) / 8) / nrow)
df2 = pd.DataFrame(
    np.random.rand(nrow, ncol),
    columns=["x{}".format(i) for i in range(ncol)]
)

print("df2.shape = ", df2.shape)
print("df2 is approximately {} MB".format(df2.memory_usage().sum() / (1024**2)))
```

### [df.to_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html)

Let's start with `df.to_csv`.

Without any additional arguments, the `df.to_csv` function will return
a string containing the csv form of the DataFrame:

```{code-cell} python
# notice the plain text format -- one row per line, columns separated by `'`
print(df1.to_csv())
```

If we do pass an argument, the first argument will be used as the file name.

```{code-cell} python
df1.to_csv("df1.csv")
```

Run the cell below to verify that the file was created.

```{code-cell} python
import os
os.path.isfile("df1.csv")
```

Let's see how long it takes to save `df2` to a file. (Because of the `%%time` at
the top, Jupyter will report the total time to run all code in
the cell)

```{code-cell} python
%%time
df2.to_csv("df2.csv")
```

As we will see below, this isn't as fastest file format we could choose.

### [df.to_excel](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_excel.html)

When saving a DataFrame to an Excel workbook, we can
choose both the name of the workbook (file) and the name of the sheet
within the file where the DataFrame should be written.

We do this by passing the workbook name as the first argument and the
sheet name as the second argument as follows.

```{code-cell} python
df1.to_excel("df1.xlsx", "df1")
```

pandas also gives us the option to write more than one DataFrame to a
workbook.

To do this, we need to first construct an instance of `pd.ExcelWriter`
and then pass that as the first argument to `df.to_excel`.

Let's see how this works.

```{code-cell} python
with pd.ExcelWriter("df1.xlsx") as writer:
    df1.to_excel(writer, "df1")
    (df1 + 10).to_excel(writer, "df1 plus 10")
```

The

```{code-block} python
with ... as ... :
```

syntax used above is an example of a *context manager*.

We don't need to understand all the details behind what this means
(google it if you are curious).

For now, just recognize that particular syntax as the way to write
multiple sheets to an Excel workbook.

```{raw} html
<p style="color:red;">
```

WARNING:

```{raw} html
</p>
```

Saving `df2` to an excel file takes a very long time.

For that reason, we will just show the code and hard-code the output
we saw when we ran the code.

```{code-block} python
%%time
df2.to_excel("df2.xlsx")
```

```{code-block} python
 Wall time: 25.7 s
```

### [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather)

As noted above, the feather file format was developed for very efficient
reading and writing between Python and your computer.

Support for this format is provided by a separate Python package called `pyarrow`.

This package is not installed by default. To install it, copy/paste the code
below into a code cell and execute.

```{code-block}
!pip install pyarrow
```

The parameters for `pyarrow.feather.write_feather` are the DataFrame and file name.

Let's try it out.

```{code-cell} python
import pyarrow.feather
pyarrow.feather.write_feather(df1, "df1.feather")
```

```{code-cell} python
%%time
pyarrow.feather.write_feather(df2, "df2.feather")
```

An example timing result:

















|format|time|
|:---------:|:----------------------:|
|csv|2.66 seconds|
|xlsx|25.7 seconds|
|feather|43 milliseconds|

As you can see, saving this DataFrame in the feather format was far
faster than either CSV or Excel.

## Reading Files into DataFrames

As with the `df.to_FOO` family of methods, there are similar
`pd.read_FOO` functions. (Note: they are in defined pandas, not as
methods on a DataFrame.)

These methods have many more options because data storage can be messy or wrong.

We will explore these in more detail in a separate lecture.

For now, we just want to highlight the differences in how to read data
from each of the file formats.

Let's start by reading the files we just created to verify that they
match the data we began with.

```{code-cell} python
# notice that index was specified in the first (0th -- why?) column of the file
df1_csv = pd.read_csv("df1.csv", index_col=0)
df1_csv.head()
```

```{code-cell} python
df1_xlsx = pd.read_excel("df1.xlsx", "df1", index_col=0)
df1_xlsx.head()
```

```{code-cell} python
# notice feather already knows what the index is
df1_feather = pyarrow.feather.read_feather("df1.feather")
df1_feather.head()
```

With the `pd.read_FOO` family of functions, we can also read files
from places on the internet.

We saved our `df1` DataFrame to a file
and posted it online.

Below, we show an example of using `pd.read_csv` to read this file.

```{code-cell} python
df1_url = "https://storage.googleapis.com/workshop_materials/df1.csv"
df1_web = pd.read_csv(df1_url, index_col=0)
df1_web.head()
```

## Practice

Now it's your turn...

In the cell below, the variable `url` contains a web address to a csv
file containing the result of all NFL games from September 1920 to
February 2017.

Your task is to do the following:

- Use `pd.read_csv` to read this file into a DataFrame named `nfl`
- Print the shape and column names of `nfl`
- Save the DataFrame to a file named `nfl.xlsx`
- Open the spreadsheet using Excel on your computer

If you finish quickly, do some basic analysis of the data. Try to do
something interesting. If you get stuck, here are some suggestions for
what to try:

- Compute the average total points in each game (note, you will need to
  sum two of the columns to get total points).
- Repeat the above calculation, but only for playoff games.
- Compute the average score for your favorite team (you'll need to
  consider when they were team1 vs team2).
- Compute the ratio of "upsets" to total games played. An upset is
  defined as a team with a lower ELO winning the game.

```{code-cell} python
url = "https://raw.githubusercontent.com/fivethirtyeight/nfl-elo-game/"
url = url + "3488b7d0b46c5f6583679bc40fb3a42d729abd39/data/nfl_games.csv"

# your code here --- create more cells if necessary
```

### Cleanup

If you want to remove the files we just created, run the following cell.

```{code-cell} python
def try_remove(file):
    if os.path.isfile(file):
        os.remove(file)

for df in ["df1", "df2"]:
    for extension in ["csv", "feather", "xlsx"]:
        filename = df + "." + extension
        try_remove(filename)
```

