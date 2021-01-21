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

# Control Flow

**Prerequisites**

- Booleans section in {doc}`Basics <basics>`
- {doc}`Collections <collections>`

**Outcomes**

- Asset pricing and NPV
    - Understand basic principles of pricing assets with deterministic payoffs
    - Apply programming with iteration and conditionals to asset pricing examples
- Conditionals
    - Understand what a conditional is
    - Be able to construct `if`/`elif`/`else` conditional blocks
    - Understand how conditionals can be used to selectively execute blocks of code
- Iteration
    - Understand what an iterable is
    - Be able to write `for` and `while` loops
    - Understand the keywords `break` and `continue`


(deterministic_asset_pricing)=
## Net Present Values and Asset Pricing

In this lecture, we'll introduce two related topics from economics:

* Net present valuations
* Asset pricing

These topics will motivate some of the programming we do in this course.

In economics and finance, "assets" provide a stream of
payoffs.

These "assets" can be concrete or abstract: a stock pays dividends over time, a
bond pays interest, an apple tree provides apples, a job pays wages, and an
education provides possible jobs (which, in turn, pay wages).

When deciding the price to pay for an asset or how to choose between
different alternatives, we need to take into account that most people would
prefer to receive 1 today vs. 1 next year.

This reflection on consumer preferences leads to the notion of a discount rate.
If you are indifferent between receiving 1.00 today and 1.10 next year, then
the discount rate over the next year is $r = 0.10$.

If we assume that an individuals preferences are consistent over time, then we
can apply that same discount rate to valuing assets further into the future.

For example, we would expect that the consumer would be indifferent between
consuming 1.00 today and $(1+r)(1+r) = 1.21$ dollars two years from now
(i.e. discount twice).

Inverting this formula, 1 delivered two years from now is equivalent to
$\frac{1}{(1+r)^2}$ today.

````{exercise}
:nonumber:
:label: dir2-3-1

See {ref}`exercise 1 <ex2-3-1>` in the exercise list.
````

(npv)=
### Net Present Value

If an asset pays a stream of payoffs over multiple time periods, then we
can use a discount rate to calculate the value to the consumer of a entire
sequence of payoffs.

Most generally, we enumerate each discrete time period (e.g. year, month, day) by
the index $t$ where today is $t=0$ and the asset lives for $T$
periods.

List the payoff at each time period as $y_t$,  which we will assume, for
now, is known in advance.

Then if the discount factor is $r \geq 0$, the consumer "values" the
payoff $y_t$ delivered at time $t$ as $\frac{1}{(1+r)^t}y_t$
where we note that if $t=0$, the value is just the current payoff
$y_0$.

Using this logic, we can write an expression for the value of the entire
sequence of payoffs with a sum.

```{math}
:label: eq_NPV

P_0 = \sum_{t=0}^T \left(\frac{1}{1 + r}\right)^t y_t
```

If $y_t$ is a constant, then we can compute this sum with a simple formula!

Below, we present some useful formulas that come from infinite series that we
will use to get our net present value formula.

For any constant $0 < \beta < 1$ and integer value $\tau > 0$,

```{math}
:label: eq_infinite_sums

\begin{aligned}
\sum_{t=0}^{\infty} \beta^t & = \frac{1}{1-\beta}\\
\sum_{t=0}^{\tau} \beta^t &= \frac{1- \beta^{\tau+1}}{1-\beta}\\
\sum_{t=\tau}^{\infty} \beta^t &=  \frac{\beta^{\tau}}{1-\beta}
\end{aligned}
```

In the case of an asset which pays one dollar until time $T$, we can use
these formulas, taking $\beta = \frac{1}{1+r}$ and $T = \tau$, to
find

$$
\begin{aligned}
P_0 &= \sum_{t=0}^T \left(\frac{1}{1 + r}\right)^t = \frac{1- (\frac{1}{1+r})^{\tau+1}}{1-\frac{1}{1+r}}\\
&= \frac{1 + r}{r} - \frac{1}{r}\left(\frac{1}{1+r} \right)^\tau
\end{aligned}
$$

Note that we can also consider an asset that lives and pays forever if
$T= \infty$, and from {eq}`eq_infinite_sums`, the value of an asset which
pays 1 forever is $\frac{1+r}{r}$.

## Conditional Statements and Blocks

Sometimes, we will only want to execute some piece of code if a certain condition
is met.

These conditions can be anything.

For example, we might add to total sales if the transaction value is positive,
but add to total returns if the value is negative.

Or, we might want to add up all incurred costs, only if the transaction happened
before a certain date.

We use *conditionals* to run particular pieces of code when certain criterion
are met.

Conditionals are closely tied to booleans, so if you don't remember what those
are, go back to the {doc}`basics <basics>` lecture for a refresher.

The basic syntax for conditionals is

```{code-block} python
if condition:
    # code to run when condition is True
else:
    # code to run if no conditions above are True
```

Note that immediately following the condition, there is a colon *and*
that the next line begins with blank spaces.

Using 4 spaces is a *very strong* convention, so that is what
we do — we recommend that you do the same.

Also note that the `else` clause is optional.

Let's see some simple examples.

```{code-cell} python
if True:
    print("This is where `True` code is run")
```

Alternatively, you could have a test which returns a booleans

```{code-cell} python
if 1 < 2:
     print("This is where `True` code is run")
```

This example is equivalent to just typing the print statement, but the
example below isn't...

```{code-cell} python
if False:
    print("This is where `True` code is run")
```

Or

```{code-cell} python
if 1 > 2:
     print("This is where `True` code is run")
```

Notice that when you run the cells above nothing is printed.

That is because the condition for the `if` statement was not true, so the code
inside the indented block was never run.

This also allows us to demonstrate the role of indentation
in determining the "block" of code.

```{code-cell} python
val = False

if val is True: # check an expression
    print("This is where `True` code is run")
    print("More code in the if block")
print("Code runs after 'if' block, regardless of val")
```

````{exercise}
:nonumber:
:label: dir2-3-2

See {ref}`exercise 2 <ex2-3-2>` in the exercise list.
````

The next example shows us how `else` works.

```{code-cell} python
val = (2 == 4)  # returns False
if val is True:
    print("This is where `True` code is run")
else:
    print("This is where `False` code is run")
    print("More else code")
print("Code runs after 'if' block, regardless of val")
```

The `if False: ...` part of this example is the same as the example
before, but now, we added an `else:` clause.

In this case, because the conditional for the `if` statement was not
`True`, the if code block was not executed, but the `else` block was.

Finally, the `Condition is True` is assumed in the `if` statement, and is often left out.  For example, the following are identical

```{code-cell} python
if (1 < 2) is True:
    print("1 < 2")

if 1 < 2:
    print("1 < 2")
```

````{exercise}
:nonumber:
:label: dir2-3-3

See {ref}`exercise 3 <ex2-3-3>` in the exercise list.
````


````{exercise}
:nonumber:
:label: dir2-3-4

See {ref}`exercise 4 <ex2-3-4>` in the exercise list.
````


### `elif` clauses

Sometimes, you have more than one condition you want to check.

For example, you might want to run a different set of code based on which
quarter a particular transaction took place in.

In this case you could check whether the date is in Q1, or in Q2, or in Q3, or if not
any of these it must be in Q4.

The way to express this type of conditional is to use one or more `elif`
clause in addition to the `if` and the `else`.

The syntax is

```{code-block} python
if condition1:
    # code to run when condition1 is True
elif condition2:
    # code to run when condition2 is True
elif condition3:
    # code to run when condition3 is True
else:
    # code to run when none of the above are true
```

You can include as many `elif` clauses as you want.

As before, the `else` part is optional.

Here's how we might express the quarter example referred to above.

```{code-cell} python
import datetime
halloween = datetime.date(2017, 10, 31)

if halloween.month > 9:
    print("Halloween is in Q4")
elif halloween.month > 6:
    print("Halloween is in Q3")
elif halloween.month > 3:
    print("Halloween is in Q2")
else:
    print("Halloween is in Q1")
```

Note that when there are multiple `if` or `elif` conditions, only the code
corresponding to the **first** true clause is run.

We saw this in action above.

We know that when `halloween.month > 9` is true, then `halloween.month > 6`
and `halloween.month > 3` must also be true, but only the code block
associated with `halloween.month > 9` was printed.

## Iteration

When doing computations or analyzing data, we often need to repeat certain
operations a finite number of times or until some condition is met.

Examples include processing all data files in a directory (folder), aggregating
revenues and costs for every period in a year, or computing the net present
value of certain assets. (In fact, later in this section, we will verify the equations
that we wrote down above.)

These are all examples of a programming concept called iteration.

We feel the concept is best understood through example, so we will present a
contrived example and then discuss the details behind doing iteration in Python.

### A Contrived Example

Suppose we wanted to print out the first 10 integers and their squares.

We *could* do something like this.

```{code-cell} python
print(f"1**2 = {1**2}")
print(f"2**2 = {2**2}")
print(f"3**2 = {3**2}")
print(f"4**2 = {4**2}")
# .. and so on until 10
```

As you can see, the code above is repetitive.

For each integer, the code is exactly the same except for the two places where
the "current" integer appears.

Suppose that I asked you to write the same print statement for an int stored in
a variable named `i`.

You might write the following code:

```{code-block} python
print(f"{i}**2 = {i**2}")
```

This more general version of the operation suggests a strategy for achieving our
goal with less repetition: have a variable `i` take on the values 1 through 10
(Quiz: How can we use `range` to create the numbers 1 to 10?) and run the line
of code above for each new value of `i`.

This can be accomplished with a `for` loop!

```{code-cell} python
for i in range(1, 11):
     print(f"{i}**2 = {i**2}")
```

Whoa, what just happened?

The integer `i` took on the values in `range(1, 11)` one by one and
for each new value it did the operations in the indented block (here
just one line that called the `print` function).

### `for` Loops

The general structure of a standard `for` loop is as follows.

```{code-block} python
for item in iterable:
   # operation 1 with item
   # operation 2 with item
   # ...
   # operation N with item
```

where `iterable` is anything capable of producing one item at a time (see
[here](https://docs.python.org/3/glossary.html#term-iterable) for official
definition from the Python team).

We've actually already seen some of the most common iterables!

Lists, tuples,
dicts, and range/zip/enumerate objects are all iterables.

Note that we can have as many operations as we want inside the indented block.

We will refer to the indented block as the "body" of the loop.

When the for loop is executed, `item` will take on one value from `iterable`
at a time and execute the loop body for each value.

(human_capital_application)=
````{exercise}
:nonumber:
:label: dir2-3-5

See {ref}`exercise 5 <ex2-3-5>` in the exercise list.
````


When iterating, each `item` in `iterable` might actually contain more than
one value.

Recall that tuples (and lists) can be unpacked directly into variables.

```{code-cell} python
tup = (4, "test")
i, x = tup
print(f"i = {i}, x = {x}, tup = {tup}")
```

Also, recall that the value of a `enumerate(iterable)` is a tuple of the
form `(i, x)` where `iterable[i] == x`.

When we use `enumerate` in a for loop, we can "unpack" both values at the same
time as follows:

```{code-cell} python
# revenue by quarter
company_revenue = [5.12, 5.20, 5.50, 6.50]

for index, value in enumerate(company_revenue):
    print(f"quarter {index} revenue is ${value} million")
```

Similarly, the index can be used to access another vector.

```{code-cell} python
cities = ["Phoenix", "Austin", "San Diego", "New York"]
states = ["Arizona", "Texas", "California", "New York"]
for index, city in enumerate(cities):
    state = states[index]
    print(f"{city} is in {state}")
```

````{exercise}
:nonumber:
:label: dir2-3-6

See {ref}`exercise 6 <ex2-3-6>` in the exercise list.
````


### `while` Loops

A related but slightly different form of iteration is to repeat something
until some condition is met.

This is typically achieved using a `while` loop.

The structure of a while loop is

```{code-block} python
while True_condition:
    # repeat these steps
```

where `True_condition` is some conditional statement that should evaluate to
`True` when iterations should continue and `False` when Python should stop
iterating.

For example, suppose we wanted to know the smallest `N` such that
$\sum_{i=0}^N i > 1000$.

We figure this out using a while loop as follows.

```{code-cell} python
total = 0
i = 0
while total <= 1000:
    i = i + 1
    total = total + i

print("The answer is", i)
```

Let's check our work.

```{code-cell} python
# Should be just less than 1000 because range(45) goes from 0 to 44
sum(range(45))
```

```{code-cell} python
# should be between 990 + 45 = 1035
sum(range(46))
```

A warning: one common programming error with while loops is to forget to
set the variable you use in the condition prior to executing.  For example,
take the following code which correctly sets a counter

```{code-cell} python
i = 0
```

And then executes a while loop

```{code-cell} python
while i < 3:
    print(i)
    i = i + 1
print("done")
```

No problems.  But if you were to execute the above cell again, or another cell, the `i=3` remains, and code is never executed (since `i < 3` begins as False).

```{code-cell} python
while i < 3:
    print(i)
    i = i + 1
print("done")
```

````{exercise}
:nonumber:
:label: dir2-3-7

See {ref}`exercise 7 <ex2-3-7>` in the exercise list.
````


### `break` and `continue`

#### `break` Out of a Loop

Sometimes we want to stop a loop early if some condition is met.

Let's revisit the example of finding the smallest `N` such that
$\sum_{i=0}^N i > 1000$.

Clearly `N` must be less than 1000, so we know we will find the answer
if we start with a `for` loop over all items in `range(1001)`.

Then, we can keep a running total as we proceed and tell Python to stop
iterating through our range once total goes above 1000.

```{code-cell} python
total = 0
for i in range(1001):
    total = total + i
    if total > 1000:
        break

print("The answer is", i)
```

````{exercise}
:nonumber:
:label: dir2-3-8

See {ref}`exercise 8 <ex2-3-8>` in the exercise list.
````


#### `continue` to the Next Iteration

Sometimes we might want to stop the *body of a loop* early if a condition is met.

To do this we can use the `continue` keyword.

The basic syntax for doing this is:

```{code-block} python
for item in iterable:
    # always do these operations
    if condition:
        continue

    # only do these operations if condition is False
```

Inside the loop body, Python will stop that loop iteration of the loop and continue directly to the next iteration when it encounters the `continue` statement.

For example, suppose I ask you to loop over the numbers 1 to 10 and print out
the message "{i} An odd number!" whenever the number `i` is odd, and do
nothing otherwise.

You can use continue to do this as follows:

```{code-cell} python
for i in range(1, 11):
    if i % 2 == 0:  # an even number... This is modulus division
        continue

    print(i, "is an odd number!")
```

````{exercise}
:nonumber:
:label: dir2-3-9

See {ref}`exercise 9 <ex2-3-9>` in the exercise list.
````


## Comprehension

Often, we will want to perform a very simple operation for every element of some iterable and
create a new iterable with these values.

This could be done by writing a for loop and saving each
value, but often using what is called a *comprehension* is more readable.

Like many Python concepts, a comprehension is easiest to understand through example.

Imagine that we have a list `x` with a list of numbers. We would like to create a list `x2` which has
the squared values of x.

```{code-cell} python
x = list(range(4))

# Create squared values with a loop
x2_loop = []
for x_val in x:
    x2_loop.append(x_val**2)

# Create squared values with a comprehension
x2_comp = [x_val**2 for x_val in x]

print(x2_loop)
print(x2_comp)
```

Notice that much of the same text appears when we do the operation in the loop and when we do the
operation with the comprehension.

- We need to specify what we are iterating over -- in both cases, this is `for x_val in x`.
- We need to square each element `x_val**2`.
- It needs to be stored somewhere -- in `x2_loop`, this is done by appending each element to a list,
  and in `x2_comp`, this is done automatically because the operation is enclosed in a list.

We can do comprehension with many different types of iterables, so we demonstrate a few more below.

```{code-cell} python
# Create a dictionary from lists
tickers = ["AAPL", "GOOGL", "TVIX"]
prices = [175.96, 1047.43, 8.38]
d = {key: value for key, value in zip(tickers, prices)}
d
```

```{code-cell} python
# Create a list from a dictionary
d = {"AMZN": "Seattle", "TVIX": "Zurich", "AAPL": "Cupertino"}

hq_cities = [d[ticker] for ticker in d.keys()]
hq_cities
```

```{code-cell} python
import math

# List from list
x = range(10)

sin_x = [math.sin(x_val) for x_val in x]
sin_x
```

````{exercise}
:nonumber:
:label: dir2-3-10

See {ref}`exercise 10 <ex2-3-10>` in the exercise list.
````

Finally, we can use this approach to build complicated nested dictionaries.

```{code-block} python
gdp_data = [9.607, 10.48, 11.06]
years = [2013, 2014, 2015]
exports = [ {"manufacturing": 2.4, "agriculture": 1.5, "services": 0.5},
            {"manufacturing": 2.5, "agriculture": 1.4, "services": 0.9},
            {"manufacturing": 2.7, "agriculture": 1.4, "services": 1.5}]
data = zip(years, gdp_data,exports)
data_dict = {year : {"gdp" : gdp, "exports": exports} for year, gdp, exports in data}
print(data_dict)

# total exports by year
[data_dict[year]["exports"]["services"] for year in data_dict.keys()]
```

## Exercises

````{exercise} 1
:nonumber:
:label: ex2-3-1

Government bonds are often issued as *zero-coupon bonds* meaning that they
make no payments throughout the entire time that they are held, but, rather
make a single payment at the time of maturity.

How much should you be willing to pay for a zero-coupon bond that paid
100 in 10 years with an interest rate of 5%?

```{code-block} python
# your code here
```

({ref}`back to text <dir2-3-1>`)
````

````{exercise} 2
:nonumber:
:label: ex2-3-2

Run the following two variations on the code with only a single change in the indentation.

After, modify the `x` to print `3` and then `2, 3` instead.

```{code-block} python
x = 1

if x > 0:
    print("1")
    print("2")
print("3")
```

```{code-block} python
x = 1

if x > 0:
    print("1")
print("2") # changed the indentation
print("3")
```

({ref}`back to text <dir2-3-2>`)
````

````{exercise} 3
:nonumber:
:label: ex2-3-3

Using the code cell below as a start, print `"Good afternoon"` if the
`current_time` is past noon.

Otherwise, do nothing.

```{hint}
Write some conditional based on `current_time.hour`.
```

```{code-block} python
import datetime
current_time = datetime.datetime.now()

## your code here
```

more text after

({ref}`back to text <dir2-3-3>`)
````

````{exercise} 4
:nonumber:
:label: ex2-3-4

In this example, you will generate a random number between 0 and 1
and then display "x > 0.5" or "x < 0.5" depending on the value of the
number.

This also introduces a new package `numpy.random` for
drawing random numbers (more in the [randomness](../scientific/randomness) lecture).

```{code-block} python
import numpy as np
x = np.random.random()
print(f"x = {x}")

## your code here
```

({ref}`back to text <dir2-3-4>`)
````

````{exercise} 5
:nonumber:
:label: ex2-3-5

In economics, when an individual has some knowledge, skills, or education
which provides them with a source of future income, we call it [human
capital](https://en.wikipedia.org/wiki/Human_capital).

When a student graduating from high school is considering whether to
continue with post-secondary education, they may consider that it gives
them higher paying jobs in the future, but requires that they don't begin
working until after graduation.

Consider the simplified example where a student has perfectly forecastable
employment and is given two choices:

1. Begin working immediately and make 40,000 a year until they retire 40
years later.
2. Pay 5,000 a year for the next 4 years to attend university, then
get a job paying 50,000 a year until they retire 40 years after making
the college attendance decision.

Should the student enroll in school if the discount rate is r = 0.05?

```{code-block} python

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

({ref}`back to text <dir2-3-5>`)
````

````{exercise} 6
:nonumber:
:label: ex2-3-6

Instead of the above, write a for loop that uses the lists of cities
and states below to print the same "{city} is in {state}" using
a `zip` instead of an `enumerate`.

```{hint}
Try using `zip`
```

```{code-block} python
cities = ["Phoenix", "Austin", "San Diego", "New York"]
states = ["Arizona", "Texas", "California", "New York"]

# Your code here
```

({ref}`back to text <dir2-3-6>`)
````

````{exercise} 7
:nonumber:
:label: ex2-3-7

Companies often invest in training their employees to raise their
productivity. Economists sometimes wonder why companies
spend this money when this incentivizes other companies to hire
their employees away with higher salaries since employees gain human capital from training?

Let's say that it costs a company 25,000 dollars to teach their
employees Python, but it raises their output by 2,500 per month. How
many months would an employee need to stay for the company to find it
profitable to pay for their employees to learn Python if their discount
rate is r = 0.01?

```{code-block} python
# Define cost of teaching python
cost = 25_000
r = 0.01

# Per month value
added_value = 2500

n_months = 0
total_npv = 0.0

# Put condition below here
while False: # (replace False with your condition here)
    n_months = n_months + 1  # Increment how many months they've worked

    # Increase total_npv
```

({ref}`back to text <dir2-3-7>`)
````

````{exercise} 8
:nonumber:
:label: ex2-3-8

Try to find the index of the first value in `x`
that is greater than 0.999 using a for loop and `break`.

```{hint}
try iterating over `range(len(x))`.
```

```{code-block} python
x = np.random.rand(10_000)
# Your code here
```

({ref}`back to text <dir2-3-8>`)
````

````{exercise} 9
:nonumber:
:label: ex2-3-9

Write a for loop that adds up all values in `x` that are greater than
or equal to 0.5.

Use the `continue` word to end the body of the loop early for all values
of `x` that are less than 0.5.

```{hint}
Try starting your loop with `for value in x:` instead of
iterating over the indices of `x`.
```

```{code-block} python
x = np.random.rand(10_000)
# Your code here
```

({ref}`back to text <dir2-3-9>`)
````

````{exercise} 10
:nonumber:
:label: ex2-3-10

Returning to our previous example: print "{city} is in {state}" for each combination
using a `zip` and a comprehension.

```{hint}
Try using `zip`
```

```{code-block} python
cities = ["Phoenix", "Austin", "San Diego", "New York"]
states = ["Arizona", "Texas", "California", "New York"]

# your code here
```

({ref}`back to text <dir2-3-10>`)
````