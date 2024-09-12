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

```{raw-cell}
# ASSIGNMENT CONFIG
generate: true
export_cell: false
check_all_cell: false
seed:
    variable: rng_seed
    autograder_value: 42
    student_value: 90
```

# Problem Set 1

See "Check Your Understanding" from {doc}`Basics <../python_fundamentals/basics>` and {doc}`Collections <../python_fundamentals/collections>`

```{raw-cell}
# BEGIN QUESTION
name: q1
points: 2
manual: true
```

## Question 1

Below this cell, add

1. A Markdown cell with
   -  two levels of headings;
   -  a numbered list (We ask for a list in Markdown, not a Python list object);
   -  an unnumbered list (again not a Python list object);
   -  text with a `*` and a `-` sign (hint: look at this cell and [escape characters](https://www.markdownguide.org/basic-syntax/#characters-you-can-escape))
   -  backticked code (see [https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet))
1. A Markdown cell with
   - the [quadratic formula](https://en.wikipedia.org/wiki/Quadratic_formula) embedded in the cell using [LaTeX](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%20Equations.html)

```{raw-cell}
# END QUESTION
```

```{raw-cell}
# BEGIN QUESTION
name: q2
points: 2
```

## Question 2

Complete the following code, which sets up variables `a, b,` and `c`, to find the roots using the quadratic formula.

$$
x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}
$$

Note: because there are two roots, you will need to calculate two
values. Store them as `x1` and `x2`.

```{code-cell} python
a = 1.0
b = 2.0
c = 1.0
# Your code goes here
```

```{raw-cell}
# BEGIN SOLUTION
```

```{code-cell} python
# BEGIN SOLUTION
x1 = -(b + (b**2 - 4*a*c)**(1/2)) / (2*a)
x2 = -(b + (b**2 - 4*a*c)**(1/2)) / (2*a)
# END SOLUTION
```

```{raw-cell}
# END SOLUTION
```

```{raw-cell}
# BEGIN TESTS
```

```{code-cell} python
# HIDDEN
(abs(x1 - -1) < 1e-6)
```

```{code-cell} python
# HIDDEN
(abs(x2 - -1) < 1e-6)
```

```{raw-cell}
# END TESTS
```

```{raw-cell}
# END QUESTION
```


```{raw-cell}
# BEGIN QUESTION
name: q3
points: 1
```

## Question 3

In the cell below, use tab completion to find a function from the time
module that returns the **local** time.

Use `time.FUNC_NAME?` (where `FUNC_NAME` is replaced with the name
of the function you found) to see information about that function,
then call the function. (Hint: look for something involving the word
`local`).

```{raw-cell}
# BEGIN SOLUTION
```

```{code-cell} python
import time
# Your code goes here

# after time., hit <TAB> to see functions
currenttime = ( time.
    # BEGIN SOLUTION
    localtime() )
    # END SOLUTION
```

```{raw-cell}
# END SOLUTION
```

Hint: if you are using an online jupyter server, the time will be based on
the server settings.  If it doesn't match your location, don't worry about it.

```{raw-cell}
# BEGIN TESTS
```

```{code-cell} python
# HIDDEN
isinstance(currenttime, time.struct_time)
```

```{raw-cell}
# END TESTS
```

```{raw-cell}
# END QUESTION
```

```{raw-cell}
# BEGIN QUESTION
name: q4
points: 2
```
## Question 4

Create the following variables:

- `D`: A floating point number with the value 10,000
- `r`: A floating point number with the value 0.025
- `T`: An integer with the value 30

Compute the present discounted value of a payment (`D`) made
in `T` years, assuming an interest rate of 2.5%. Save this value to
a new variable called `PDV` and print your output.

Hint: The formula is

$$
\text{PDV} = \frac{D}{(1 + r)^T}
$$

```{raw-cell}
# BEGIN SOLUTION
```

```{code-cell} python
# Your code goes here
# BEGIN SOLUTION
D = 10000.0
r = 0.025
T = 30
PDV = D/(1+r)**T
# END SOLUTION
```

```{raw-cell}
# END SOLUTION
```

```{raw-cell}
# BEGIN TESTS
```

```{code-cell} python
# HIDDEN
isinstance(D, float) and isinstance(r, float) and isinstance(T, int)
```

```{code-cell} python
# HIDDEN
(abs(D - 10000) < 1e-5) and (abs(r -  0.025) < 1e-5) and (T == 30)
```

```{code-cell} python
# HIDDEN
abs(PDV - 10000/(1.025)**30) < 1e-5
```

```{raw-cell}
# END TESTS
```

```{raw-cell}
# END QUESTION
```

```{raw-cell}
# BEGIN QUESTION
name: q5
points: 1
```

## Question 5

How could you use the variables `x` and `y` to create the sentence
`Hello World` ?

Hint: Think about how to represent a space as a string.

```{raw-cell}
# BEGIN SOLUTION
```

```{code-cell} python
x = "Hello"
y = "World"
sentence = (
    # BEGIN SOLUTION
    x + " " + y )
    # END SOLUTION
```

```{raw-cell}
# END SOLUTION
```

```{raw-cell}
# BEGIN TESTS
```

```{code-cell} python
# HIDDEN
sentence == "Hello World"
```
```{raw-cell}
# END TESTS
```

```{raw-cell}
# END QUESTION
```

```{raw-cell}
# BEGIN QUESTION
name: q6
points: 1
```

## Question 6

Suppose you are working with price data and come across the value
`"€6.50"`.

When Python tries to interpret this value, it sees the value as the string
`"€6.50"` instead of the number `6.50`. (Quiz: why is this a
problem? Think about the examples above.)

In this exercise, your task is to convert the variable `price` below
into a number.

*Hint*: Once the string is in a suitable format, you can call
`float(clean_price)` to make it a number.

```{raw-cell}
# BEGIN SOLUTION
```

```{code-cell} python
price_string = "€6.50"
price_number = (
   # Your code goes here
   # BEGIN SOLUTION
   float(price_string[1:])
   # END SOLUTION
   )
```

```{raw-cell}
# END SOLUTION
```

```{raw-cell}
# BEGIN TESTS
```

```{code-cell} python
# HIDDEN
abs(price_number - 6.5) < 1e-5
```

```{raw-cell}
# END TESTS
```

```{raw-cell}
# END QUESTION
```


```{raw-cell}
# BEGIN QUESTION
name: q7
points: 2
manual: true
```

## Question 7

Use Python formatting (e.g. `print(f"text {somecode}")` where `somecode` is a valid expression or variable name) to produce the following
output.

```{code-block} none
The 1st quarter revenue was $110M
The 2nd quarter revenue was $95M
The 3rd quarter revenue was $100M
The 4th quarter revenue was $130M
```

```{code-cell} python
# Your code goes here
```

```{raw-cell}
# END QUESTION
```


```{raw-cell}
# BEGIN QUESTION
name: q8
points: 1
```

## Question 8

Define two lists y and z.

They can contain **anything you want**.

Check what happens when you do y + z.
When you have finished that, try 2 * x and x * 2 where x represents the object you created from y + z.

Briefly explain.

```{raw-cell}
# BEGIN SOLUTION
```

```{code-cell} python
y = [] # fill me in!
z = [] # fill me in!
# Your code goes here
# BEGIN SOLUTION
y = ["a",2]
z = ["zebra","lion","elephant"]
x = y + z
display(x)
display(2*x)
display(x*2)
# END SOLUTION
```

```{raw-cell}
# END SOLUTION
```

```{raw-cell}
# BEGIN TESTS
```

```{code-cell} python
# HIDDEN
isinstance(x, list) and isinstance(y,list)
```

```{raw-cell}
# END TESTS
```

```{raw-cell}
# END QUESTION
```
