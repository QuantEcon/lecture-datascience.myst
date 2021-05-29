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

# Problem Set 4

See "Check Your Understanding" from {doc}`../scientific/applied_linalg <../scientific/applied_linalg>` and {doc}`../scientific/randomness <../scientific/randomness>`

## Question 1 and 2

Alice is a stock broker who owns two types of assets: A and B. She owns 100
units of asset A and 50 units of asset B. The current interest rate is 5%.
Each of the A assets have a remaining duration of 6 years and pay
$1500$ dollars each year while each of the B assets have a remaining duration
of 4 years and pay $500$ dollars each year (assume the first payment starts at the beginning of the
first year and hence, should not be discounted).

### Question 1

Alice would like to retire if she
can sell her assets for more than $1 million. Use vector addition, scalar
multiplication, and dot products to determine whether she can retire.

```{code-cell} python
r = 0.05

# your code here
```

### Question 2

Consider if Alice had several alternate portfolios and decide which is the most valuable.
Use a matrix product to simultaneously calculate the values of each portfolio,
as we did in the lecture notes.

- Portfolio 1: 100 units of A, 40 units of B
- Portfolio 2: 50 units of A, 150 units of B
- Portfolio 3: 120 units of A, 0 units of B

```{code-cell} python
r = 0.05

# your code here
```

## Question 3

As in {doc}`../scientific/applied_linalg <../scientific/applied_linalg>`:

Consider an economy where in any given year, $\alpha = 3\%$ of workers lose their jobs and
$\phi = 12\%$ of unemployed workers find jobs.

Define the vector $x_0 = \begin{bmatrix} 600,000 & 200,000 \end{bmatrix}$ as the number of
employed and unemployed workers (respectively) at time $0$ in the economy.

Adapting the code from the lecture notes, plot the unemployment rate over time in this economy for $t = 0, \ldots 20$ (i.e. number of employed over total number of workers).

```{code-cell} python
# your code here
```

Continue the simulation for 1000 periods to find a long-run unemployment rate.

```{code-cell} python
# your code here
```

Adapt the lecture notes code to use the matrix you set up for the evolution equation, and
find the (left) eigenvector associated with the unit eigenvalue.  Rescale this as required (i.e.
it is only unique up to a scaling parameter) to find the stationary unemployment rate. Compare to the simulated
one.

```{code-cell} python
# your code here
```

## Question 4

Adapt our unemployment example to add in an additional category: a probationary period where a firm is deciding if they want to make
an employee a permanent offer.  Now, in any given year
-  $\alpha = 3\%$ of workers with full job offers lose their jobs.
-  $\phi = 12\%$ of unemployed workers find a "probationary" job (i.e., they cannot get the permanent job directly).
-  Probation lasts for exactly an entire period, at which point $\lambda = 60\%$ of the employees get a permanent job offer, while the remainder become unemployed.

Define the vector $x_0 = \begin{bmatrix} 600,000 & 200,000 & 100,000\end{bmatrix}$ as the number of
fully employed, unemployed, and probationary workers (respectively) at time $0$ in the economy.

Adapting the code from the lecture notes, plot the mass of all three types of employment situations in this economy for $t = 0, \ldots 20$.

```{code-cell} python
# your code here
```

## Question 5

Wikipedia and other credible statistics sources tell us that the mean and
variance of the Uniform(0, 1) distribution are (1/2, 1/12) respectively.

How could we check whether the numpy random numbers approximate these
values? (*hint*: some functions in {doc}`../scientific/numpy_arrays <../scientific/numpy_arrays>` and {doc}`../scientific/randomness <../scientific/randomness>` might be useful)

```{code-cell} python
# your code here
```

## Question 6

Assume you have been given the opportunity to choose between one of three financial assets.

You will be given the asset for free, allowed to hold it indefinitely, and will keep all payoffs.

Also assume the assets' payoffs are distributed as follows (the notations are the same as in "Continuous Distributions" subsection of {doc}`../scientific/randomness <../scientific/randomness>`):

1. Normal with $\mu = 10, \sigma = 5$
1. Gamma with $k = 5.3, \theta = 2$
1. Gamma with $k = 5, \theta = 2$

Use `scipy.stats` to answer the following questions:

- Which asset has the highest average returns?
- Which asset has the highest median returns?
- Which asset has the lowest coefficient of variation, i.e., standard deviation divided by mean? (This measure is similar to "Sharpe Ratio")
- Which asset would you choose? Why? (There is not a single right answer here. Just be creative and express your preferences.)

You can find the official documentation of `scipy.stats` [here](https://docs.scipy.org/doc/scipy/reference/stats.html)

```{code-cell} python
# your code here
```

## Question 7

Let's revisit the unemployment example from the {doc}`../scientific/applied_linalg <../scientific/applied_linalg>`.

We'll repeat necessary details here.

Consider an economy where in any given year, $\alpha = 5\%$ of workers lose their jobs, and
$\phi = 10\%$ of unemployed workers find jobs.

Initially, 90% of the 1,000,000 workers are employed.

Suppose that the average employed worker earns 10 dollars while an unemployed worker
earns 1 dollar per period.

With this, do the following:

* Represent this problem as a Markov chain by defining the three components defined above

```{code-cell} python
# define components here
```

* Construct an instance of the QuantEcon MarkovChain using the objects defined in part 1.

```{code-cell} python
# construct the Markov chain
```

* Simulate the Markov chain 5 times for 50 time periods starting from an employment state and plot the chains over time (see helper code below).

```{code-cell} python
n = 50
M = 5

# uncomment the lines below and fill in the blanks
# sim = XXXXX.simulate(n, init = XXXXX, num_reps = M)
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(range(n), sim.T, alpha=0.4)
```

* Using the approach above, simulate the Markov chain `M=20` times for 50 time periods. Instead of starting from an employment state, start off the `M` in proportion to the initial condition above (i.e. 90% of them in an employment state and 10% in an unemployment state). (Hint: you can pass a list to the `init` parameter in the `simulate` function.)

With this simulation, plot the average proportion of `M` agents in the employment state (i.e. it should start at 0.90 from the initial condition).

```{code-cell} python
# define components here
```

* Calculate the steady-state of the Markov chain and compare results from this simulation to the steady-state unemployment rate for the Markov chain (on a similar graph).

```{code-cell} python
# define components here
```

* Determine the average long-run payment for a worker in this setting. (Hint: Think about the stationary distribution)

```{code-cell} python
# define components here

# construct Markov chain


# Long-run average payment
```

