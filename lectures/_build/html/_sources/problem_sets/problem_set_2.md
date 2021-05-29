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

# Problem Set 2

See "Check Your Understanding" from {doc}`collections <../python_fundamentals/collections>` and {doc}`control flow <../python_fundamentals/control_flow>`

Note:  unless stated otherwise, the timing of streams of payoffs is immediately at
time `0` where appropriate.  For example, dividends $\{d_1, d_2, \ldots d_{\infty}\}$
should be valued as $d_1 + \beta d_2 + \beta^2 d_3 \ldots = \sum_{j=0}^{\infty} \beta^j d_j$.

This timing is consistent with the lectures and most economics, but is different from the timing assumptions
of some finance models.

## Question 1-4

Consider a bond that pays a $500 dividend once a quarter.

It pays in the months of March, June, September, and December.

It promises to do so for 10 years after you purchase it (start in January 2019).

You discount the future at rate $r = 0.005$ per _month_.

### Question 1

How much do you value the asset in January 2019?

```{code-cell} python
# Your code goes here
```

### Question 2

Consider a different asset that pays a lump sum at its expiration date rather than a quarterly dividend of $500 dollars, how much would this asset need to pay in December 2028 (the final payment date of the quarterly asset) for the two assets to be equally valued?

```{code-cell} python
# Your code goes here
```

### Question 3

How much should you be willing to pay if your friend bought the quarterly asset (from the main text) in January
2019 but wanted to sell it to you in October 2019?

```{code-cell} python
# Your code goes here
```

### Question 4

If you already knew that your discount rate would change annually according to the
table below, at what price would you value the quarterly asset (from the main text) in January 2019?

*Hint*: There are various ways to do this... One way might include a zipped loop for years and a
second loop for months.

*More Challenging*: Can you create the list of interest rates without calculating each year individually?













































|Year|Discount Rate|
|:----:|:-------------:|
|2019|0.005|
|2020|0.00475|
|2021|0.0045|
|2022|0.00425|
|2023|0.004|
|2024|0.00375|
|2025|0.0035|
|2026|0.00325|
|2027|0.003|
|2028|0.00275|

Hint: create appropriate collections typing from the data directly in the code.  You cannot parse the
text table directly.

```{code-cell} python
# Your code goes here
```

## Questions 5-6

Companies often invest in training their employees to raise their
productivity. Economists sometimes wonder why companies
spend money on training employees when this incentivizes other companies to poach
their employees with higher salaries since the employees gain human capital from training.

Imagine it costs a company 25,000 dollars to teach their employees Python, but
it also raises their output by 2,500 dollars per month. The company discounts the future
at rate of $r = 0.01$ per month.

### Question 5

For how many full months does an employee need to stay at a company for that company to make a profit for
paying for their employees' Python training?

```{code-cell} python
# Your code goes here
```

### Question 6

Imagine that 3/4 of the employees stay for 8 months and 1/4 of the employees stay
for 24 months. Is it worth it for the company to invest in employee Python training?

```{code-cell} python
# Your code goes here
```

## Question 7 and 8

Take the following stock market data, including a stock ticker, price, and company name:

























|Ticker|Price|Name|
|:------:|:-------:|:----------------:|
|AAPL|175.96|Apple Inc.|
|GOOGL|0.00475|Alphabet Inc.|
|TVIX|0.0045|Credit Suisse AG|

Hint: create appropriate collections typing from the data directly in the code.  You cannot parse the table directly.

### Question 7

- Create a new dict which associates ticker with its price.  i.e. the dict key should be a string, the dict value should be a number, and you can ignore the name.
- Display a list of the underlying stock tickers using the dictionary.  Hint:
  use `.<TAB>` on your dictionary to look for methods to get the list

```{code-cell} python
# Your code goes here
```

### Question 8 (More Challenging)

Using the same data,

- Create a new dict, whose values are dictionaries that have a price and name key. These keys should associate the stock tickers with both its stock price and the company name.
- Display a list of the underlying stock names (i.e. not the ticker symbol) using the dictionary. Hint: use a comprehension.

```{code-cell} python
# Your code goes here
```

## Question 9 (More Challenging)

Imagine that we'd like to invest in a small startup company. We have secret
information (given to us from a legal source, like statistics, so that we
aren't insider trading or anything like that...) that the startup will have
4,000 dollars of profits for its first 5 years,
then 20,000 dollars of profits for the next 10 years, and
then 50,000 dollars of profits for the 10 years after that.

After year 25, the company will go under and pay 0 profits.

The company would like you to buy 50% of its shares, which means that you
will receive 50% of all of the future profits.

If you discount the future at $r = 0.05$, how much would you be willing to pay?

Hint: Think of this in terms of NPV; you should use some conditional
statements.

*More Challenging*: Can you think of a way to use the summation equations from the lectures
to check your work!?

```{code-cell} python
profits_0_5 = 4000
profits_5_15 = 20_000
profits_15_25 = 50_000

willingness_to_pay = 0.0
for year in range(25):
    print("replace with your code!")
```

## Question 10 (More Challenging)

For the tuple `foo` below, use a combination of `zip`, `range`, and `len` to mimic `enumerate(foo)`. Verify that your proposed solution is correct by converting each to a list and checking equality with == HINT: You can see what the answer should look like by starting with `list(enumerate(foo))`.

```{code-cell} python
foo = ("good", "luck!")
# Your code goes here
```

## Question 11

In economics, when an individual has knowledge, skills, or education that provides them with a
source of future income, we call it [human capital](https://en.wikipedia.org/wiki/Human_capital).
When a student graduating from high school is considering whether to continue with post-secondary
education, they may consider that it gives them higher-paying jobs in the future, but requires that
they commence work only after graduation.

Consider the simplified example where a student has perfectly forecastable employment and is given two choices:

1. Begin working immediately and make 40,000 dollars a year until they retire 40 years later.
1. Pay 5,000 dollars a year for the next 4 years to attend university and then get a job paying
   50,000 dollars a year until they retire 40 years after making the college attendance decision.

Should the student enroll in school if the discount rate is $r = 0.05$?

```{code-cell} python
# Discount rate
r = 0.05

# High school wage
w_hs = 40_000

# College wage and cost of college
c_college = 5_000
w_college = 50_000

# Compute npv of being a hs worker

# Compute npv of attending college

# Compute npv of being a college worker

# Is npv_collegeworker - npv_collegecost > npv_hsworker
```

