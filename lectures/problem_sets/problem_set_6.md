---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
orphan: true
---

# Problem Set 6

See {doc}`Merge <../pandas/merge>`, {doc}`Reshape <../pandas/reshape>`, and {doc}`GroupBy <../pandas/groupby>`

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline
```

## Questions 1-7

Lets start with a relatively straightforward exercise before we get to the really fun stuff.

The following code loads a cleaned piece of census data from Statistics Canada.

```{code-cell} python
df = pd.read_csv("https://datascience.quantecon.org/assets/data/canada_census.csv", header=0, index_col=False)
df.head()
```

A *census division* is a geographical area, smaller than a Canadian province, that is used to
organize information at a slightly more granular level than by province or by city. The census
divisions are shown below.

```{figure} ../_static/canada_censusdivisions_map.png
:alt: canada_censusdivision_map.png
```

The data above contains information on the population, percent of population with a college
degree, percent of population who own their house/apartment, and the median after-tax income at the
*census division* level.

Hint: The `groupby` is the key here.  You will need to practice different split, apply, and combine options.

### Question 1

Assume that you have a separate data source with province codes and names.

```{code-cell} python
df_provincecodes = pd.DataFrame({
    "Pname" : [ 'Newfoundland and Labrador', 'Prince Edward Island', 'Nova Scotia',
                'New Brunswick', 'Quebec', 'Ontario', 'Manitoba', 'Saskatchewan',
                'Alberta', 'British Columbia', 'Yukon', 'Northwest Territories','Nunavut'],
    "Code" : ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
            })
df_provincecodes
```

With this,

1. Either merge or join these province codes into the census dataframe to provide province codes for each province
   name. Hint: You need to figure out which "key" matches in the merge, and don't be afraid to rename columns for convenience.
1. Drop the province names from the resulting dataframe.
1. Rename the column with the province codes to "Province".  Hint: `.rename(columns = <YOURDICTIONARY>)`

```{code-cell} python
# Your code here
```

For this particular example, you could have renamed the column using `replace`. This is a good check.

```{code-cell} python
(pd.read_csv("https://datascience.quantecon.org/assets/data/canada_census.csv", header=0, index_col=False)
.replace({
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB", "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL", "Northwest Territories": "NT", "Nova Scotia": "NS",
    "Nunavut": "NU", "Ontario": "ON", "Prince Edward Island": "PE", "Quebec": "QC",
    "Saskatchewan": "SK", "Yukon": "YT"})
.rename(columns={"Pname" : "Province"})
.head()
)
```

### Question 2

Which province has the highest population? Which has the lowest?

```{code-cell} python
# Your code here
```

### Question 3

Show a bar plot and a pie plot of the province populations.  Hint: After the split-apply-combine, you can use `.plot.bar()` or `.plot.pie()`.

```{code-cell} python
# Your code here
```

### Question 3

Which province has the highest percent of individuals with a college education? Which has the
lowest?

Hint: Remember to weight this calculation by population!

```{code-cell} python
# Your code here
```

### Question 4

What is the census division with the highest median income in each province?

```{code-cell} python
# Your code here
```

### Question 5

By province, what is the total population of census divisions where more than 80 percent of the population own houses ?

```{code-cell} python
# Your code here
```

### Question 6

By province, what is the median income and average proportion of college-educated individuals in census divisions
where more than 80 percent of the population own houses?

```{code-cell} python
# Your code here
```

### Question 7

Classify the census divisions as low, medium, and highly-educated by using the college-educated proportions,
where "low" indicates that less than 10 percent of the area is college-educated, "medium" indicates between 10 and 20 percent is college-educated, and "high" indicates more than 20 percent.

Based on that classification, find the average income. Weight this average income by population for each of the low, medium, high education groups.

```{code-cell} python
# Your code here
```

## Questions 8-9

Let's look at another example.

This time, we will use a dataset from the [Bureau of Transportation
Statistics](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)
that describes the cause for all US domestic flight delays in November 2016:

Loading this dataset the first time will take a minute or two because it is quite hefty... We recommend taking a break to view this [xkcd comic](https://xkcd.com/303/).

```{code-cell} python
url = "https://datascience.quantecon.org/assets/data/airline_performance_dec16.csv.zip"
air_perf = pd.read_csv(url)[["CRSDepTime", "Carrier", "CarrierDelay", "ArrDelay"]]
air_perf.info()
air_perf.head()
```

The `Carrier` column identifies the airline while the `CarrierDelay`
reports the total delay, in minutes, that was the "carrier's fault".

### Question 8

Determine the 10 airlines which, on average, contribute most to delays.

```{code-cell} python
# Your code here
# avg_delays =
```

### Question 9

One issue with this dataset is that we might not know what all those two letter carrier codes are!

Thankfully, we have a second dataset that maps two-letter codes
to full airline names:

```{code-cell} python
url = "https://datascience.quantecon.org/assets/data/airline_carrier_codes.csv.zip"
carrier_code = pd.read_csv(url)
carrier_code.tail()
```

In this question, you should merge the carrier codes and the previously computed dataframe from Question 9 (the 10 airlines that contribute most to delays).

```{code-cell} python
# Your code here
# avg_delays_w_name
```

## Question 10

In this question, we will load data from the World Bank. World Bank data is often stored in formats containing vestigial columns because of their data format standardization.

This particular data contains the world's age dependency ratios (old) across countries. The ratio is the number of people who are
above 65 divided by the number of people between 16 and 65 and measures how many working
individuals exist relative to the number of dependent (retired) individuals.

```{code-cell} python
adr = pd.read_csv("https://datascience.quantecon.org/assets/data/WorldBank_AgeDependencyRatio.csv")
adr.head()
```

This data only has a single variable, so you can eliminate the `Series Name` and `Series Code`
columns. You can also eliminate the `Country Code` or  `Country Name` column (but not both),
since they contain repetitive information.

We can organize this data in a couple of ways.

The first (and the one we'd usually choose) is to place the years and country names on the index and
have a single column. (If we had more variables, each variable could have its own column.)

Another reasonable organization is to have one country per column and place the years on the index.

Your goal is to reshape the data both ways. Which is easier? Which do you
think a better organization method?

```{code-cell} python
# Reshape to have years and countries on index
```

```{code-cell} python
# Reshape to have years on index and country identifiers as columns
```

