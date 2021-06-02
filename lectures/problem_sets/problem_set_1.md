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

# Problem Set 1

See "Check Your Understanding" from {doc}`Basics <../python_fundamentals/basics>` and {doc}`Collections <../python_fundamentals/collections>`

## Question 1

Below this cell, add

1. A Markdown cell with
   -  two levels of headings;
   -  a numbered list (We ask for a list in Markdown, not a Python list object);
   -  an unnumbered list (again not a Python list object);
   -  text with a `%` and a `-` sign (hint: look at this cell and [escape characters](https://www.markdownguide.org/basic-syntax/#characters-you-can-escape))
   -  backticked code (see [https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet))
1. A Markdown cell with
   - the [quadratic formula](https://en.wikipedia.org/wiki/Quadratic_formula) embedded in the cell using [LaTeX](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%20Equations.html)

## Question 2

Complete the following code, which sets up variables `a, b,` and `c`, to find the roots using the quadratic formula.

$$
x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}
$$

Note: because there are two roots, you will need to calculate two values of `x`

```{code-cell} python
a = 1.0
b = 2.0
c = 1.0
# Your code goes here
```

## Question 3

In the cell below, use tab completion to find a function from the time
module that displays the **local** time.

Use `time.FUNC_NAME?` (where `FUNC_NAME` is replaced with the name
of the function you found) to see information about that function,
then call the function. (Hint: look for something involving the word
`local`).

```{code-cell} python
import time
# Your code goes here
# time. # uncomment and hit <TAB> to see functions
```

Hint: if you are using an online jupyter server, the time will be based on
the server settings.  If it doesn't match your location, don't worry about it.

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

```{code-cell} python
# Your code goes here
```

## Question 5

How could you use the variables `x` and `y` to create the sentence
`Hello World` ?

Hint: Think about how to represent a space as a string.

```{code-cell} python
x = "Hello"
y = "World"
# Your code goes here
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

```{code-cell} python
price = "€6.50"
# Your code goes here
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

## Question 8

Define two lists y and z.

They can contain **anything you want**.

Check what happens when you do y + z.
When you have finished that, try 2 * x and x * 2 where x represents the object you created from y + z.

Briefly explain.

```{code-cell} python
y = [] # fill me in!
z = [] # fill me in!
# Your code goes here
```

