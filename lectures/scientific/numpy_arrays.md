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

# Introduction to Numpy

**Prerequisites**

- {doc}`Python Fundamentals <../python_fundamentals/index>`

**Outcomes**

- Understand basics about numpy arrays
- Index into multi-dimensional arrays
- Use universal functions/broadcasting to do element-wise operations on arrays


## Numpy Arrays

Now that we have learned the fundamentals of programming in Python, we will learn how we can use Python
to perform the computations required in data science and economics. We call these the "scientific Python tools".

The foundational library that helps us perform these computations is known as `numpy` (numerical
Python).

Numpy's core contribution is a new data-type called an *array*.

An array is similar to a list, but numpy imposes some additional restrictions on how the data inside is organized.

These restrictions allow numpy to

1. Be more efficient in performing mathematical and scientific computations.
1. Expose functions that allow numpy to do the necessary linear algebra for machine learning and statistics.

Before we get started, please note that the convention for importing the numpy package is to use the
nickname `np`:

```{code-cell} python
import numpy as np
```

### What is an Array?

An array is a multi-dimensional grid of values.

What does this mean? It is easier to demonstrate than to explain.

In this block of code, we build a 1-dimensional array.

```{code-cell} python
# create an array from a list
x_1d = np.array([1, 2, 3])
print(x_1d)
```

You can think of a 1-dimensional array as a list of numbers.

```{code-cell} python
# We can index like we did with lists
print(x_1d[0])
print(x_1d[0:2])
```

Note that the range of indices does not include the end-point, that
is

```{code-cell} python
print(x_1d[0:3] == x_1d[:])
print(x_1d[0:2])
```

The differences emerge as we move into higher dimensions.

Next, we define a 2-dimensional array (a matrix)

```{code-cell} python
x_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x_2d)
```

Notice that the data is no longer represented as something flat, but rather,
as three rows and three columns of numbers.

The first question that you might ask yourself is: "how do I access the values in this array?"

You access each element by specifying a row first and then a column. For
example, if we wanted to access the `6`, we would ask for the (1, 2) element.

```{code-cell} python
print(x_2d[1, 2])  # Indexing into two dimensions!
```

Or to get the top left corner...

```{code-cell} python
print(x_2d[0, 0])  # Indexing into two dimensions!
```

To get the first, and then second rows...

```{code-cell} python
print(x_2d[0, :])
print(x_2d[1, :])
```

Or the columns...

```{code-cell} python
print(x_2d[:, 0])
print(x_2d[:, 1])
```

This continues to generalize, since numpy gives us as many dimensions as we want in an array.

For example, we build a 3-dimensional array below.

```{code-cell} python
x_3d_list = [[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]]
x_3d = np.array(x_3d_list)
print(x_3d)
```

#### Array Indexing

Now that there are multiple dimensions, indexing might feel somewhat non-obvious.

Do the rows or columns come first? In higher dimensions, what is the order of
the index?

Notice that the array is built using a list of lists (you could also use tuples!).

Indexing into the array will correspond to choosing elements from each list.

First, notice that the dimensions give two stacked matrices, which we can access with

```{code-cell} python
print(x_3d[0])
print(x_3d[1])
```

In the case of the first, it is synonymous with

```{code-cell} python
print(x_3d[0, :, :])
```

Let's work through another example to further clarify this concept with our
3-dimensional array.

Our goal will be to find the index that retrieves the `4` out of `x_3d`.

Recall that when we created `x_3d`, we used the list `[[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]]`.

Notice that the 0 element of that list is `[[1, 2, 3], [4, 5, 6]]`. This is the
list that contains the `4` so the first index we would use is a 0.

```{code-cell} python
print(f"The 0 element is {x_3d_list[0]}")
print(f"The 1 element is {x_3d_list[1]}")
```

We then move to the next lists which were the 0 element of the inner-most dimension. Notice that
the two lists at this level `[1, 2, 3]` and `[3, 4, 5]`.

The 4 is in the second 1 element (index `1`), so the second index we would choose is 1.

```{code-cell} python
print(f"The 0 element of the 0 element is {x_3d_list[0][0]}")
print(f"The 1 element of the 0 element is {x_3d_list[0][1]}")
```

Finally, we move to the outer-most dimension, which has a list of numbers
`[4, 5, 6]`.

The 4 is element 0 of this list, so the third, or outer-most index, would be `0`.

```{code-cell} python
print(f"The 0 element of the 1 element of the 0 element is {x_3d_list[0][1][0]}")
```

Now we can use these same indices to index into the array. With an array, we can index using a single operation rather than repeated indexing as we did with the list `x_3d_list[0][1][0]`.

Let's test it to see whether we did it correctly!

```{code-cell} python
print(x_3d[0, 1, 0])
```

Success!

````{admonition} Exercise
:name: dir3-1-1

See exercise 1 in the {ref}`exercise list <ex3-1>`.
````

````{admonition} Exercise
:name: dir3-1-2

See exercise 2 in the {ref}`exercise list <ex3-1>`.
````

We can also select multiple elements at a time -- this is called slicing.

If we wanted to have an array with just `[1, 2, 3]` then we would do

```{code-cell} python
print(x_3d[0, 0, :])
```

Notice that we put a `:` on the dimension where we want to select all of the elements. We can also
slice out subsets of the elements by doing `start:stop+1`.

Notice how the following arrays differ.

```{code-cell} python
print(x_3d[:, 0, :])
print(x_3d[:, 0, 0:2])
print(x_3d[:, 0, :2])  # the 0  in 0:2 is optional
```

````{admonition} Exercise
:name: dir3-1-3

See exercise 3 in the {ref}`exercise list <ex3-1>`.
````


### Array Functionality

#### Array Properties

All numpy arrays have various useful properties.

Properties are similar to methods in that they're accessed through
the "dot notation." However, they aren't a function so we don't need parentheses.

The two most frequently used properties are `shape` and `dtype`.

`shape` tells us how many elements are in each array dimension.

`dtype` tells us the types of an array's elements.

Let's do some examples to see these properties in action.

```{code-cell} python
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x.shape)
print(x.dtype)
```

We'll use this to practice unpacking a tuple, like `x.shape`, directly into variables.

```{code-cell} python
rows, columns = x.shape
print(f"rows = {rows}, columns = {columns}")
```

```{code-cell} python
x = np.array([True, False, True])
print(x.shape)
print(x.dtype)
```

Note that in the above, the `(3,)` represents a tuple of length 1, distinct from a scalar integer `3`.

```{code-cell} python
x = np.array([
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
])
print(x.shape)
print(x.dtype)
```

#### Creating Arrays

It's usually impractical to define arrays by hand as we have done so far.

We'll often need to create an array with default values and then fill it
with other values.

We can create arrays with the functions `np.zeros` and `np.ones`.

Both functions take a tuple that denotes the shape of an array and creates an
array filled with 0s or 1s respectively.

```{code-cell} python
sizes = (2, 3, 4)
x = np.zeros(sizes) # note, a tuple!
x
```

```{code-cell} python
y = np.ones((4))
y
```

#### Broadcasting Operations

Two types of operations that will be useful for arrays of any dimension are:

1. Operations between an array and a single number.
1. Operations between two arrays of the same shape.

When we perform operations on an array by using a single number, we simply apply that operation to every element of the array.

```{code-cell} python
# Using np.ones to create an array
x = np.ones((2, 2))
print("x = ", x)
print("2 + x = ", 2 + x)
print("2 - x = ", 2 - x)
print("2 * x = ", 2 * x)
print("x / 2 = ", x / 2)
```


````{admonition} Exercise
:name: dir3-1-4

See exercise 4 in the {ref}`exercise list <ex3-1>`.
````

Operations between two arrays of the same size, in this case `(2, 2)`, simply apply the operation
element-wise between the arrays.

```{code-cell} python
x = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.ones((2, 2))
print("x = ", x)
print("y = ", y)
print("x + y = ", x + y)
print("x - y", x - y)
print("(elementwise) x * y = ", x * y)
print("(elementwise) x / y = ", x / y)
```

### Universal Functions

We will often need to transform data by applying a function to every element of an array.

Numpy has good support for these operations, called *universal functions* or ufuncs for short.

The
[numpy documentation](https://docs.scipy.org/doc/numpy/reference/ufuncs.html?highlight=ufunc#available-ufuncs)
has a list of all available ufuncs.

```{note}
You should think of operations between a single number and an array, as we
just saw, as a ufunc.
```

Below, we will create an array that contains 10 points between 0 and 25.

```{code-cell} python
# This is similar to range -- but spits out 50 evenly spaced points from 0.5
# to 25.
x = np.linspace(0.5, 25, 10)
```

We will experiment with some ufuncs below:

```{code-cell} python
# Applies the sin function to each element of x
np.sin(x)
```

Of course, we could do the same thing with a comprehension, but
the code would be both less readable and less efficient.

```{code-cell} python
np.array([np.sin(xval) for xval in x])
```

You can use the inspector or the docstrings with `np.<TAB>` to see other available functions, such as

```{code-cell} python
# Takes log of each element of x
np.log(x)
```

A benefit of using the numpy arrays is that numpy has succinct code for combining vectorized operations.

```{code-cell} python
# Calculate log(z) * z elementwise
z = np.array([1,2,3])
np.log(z) * z
```

````{admonition} Exercise
:name: dir3-1-5

See exercise 5 in the {ref}`exercise list <ex3-1>`.
````

### Other Useful Array Operations

We have barely scratched the surface of what is possible using numpy arrays.

We hope you will experiment with other functions from numpy and see how they
work.

Below, we demonstrate a few more array operations that we find most useful -- just to give you an idea
of what else you might find.

When you're attempting to do an operation that you feel should be common, the numpy library probably has it.

Use Google and tab completion to check this.

```{code-cell} python
x = np.linspace(0, 25, 10)
```

```{code-cell} python
np.mean(x)
```

```{code-cell} python
np.std(x)
```

```{code-cell} python
# np.min, np.median, etc... are also defined
np.max(x)
```

```{code-cell} python
np.diff(x)
```

```{code-cell} python
np.reshape(x, (5, 2))
```

Note that many of these operations can be called as methods on `x`:

```{code-cell} python
print(x.mean())
print(x.std())
print(x.max())
# print(x.diff())  # this one is not a method...
print(x.reshape((5, 2)))
```

Finally, `np.vectorize` can be conveniently used with numpy broadcasting and any functions.

```{code-cell} python
np.random.seed(42)
x = np.random.rand(10)
print(x)

def f(val):
    if val < 0.3:
        return "low"
    else:
        return "high"

print(f(0.1)) # scalar, no problem
# f(x) # array, fails since f() is scalar
f_vec = np.vectorize(f)
print(f_vec(x))
```

Caution: `np.vectorize` is convenient for numpy broadcasting with any function
but is not intended to be high performance.

When speed matters, directly write a `f` function to work on arrays.

(ex3-1)=
## Exercises

### Exercise 1

Try indexing into another element of your choice from the
3-dimensional array.

Building an understanding of indexing means working through this
type of operation several times -- without skipping steps!

({ref}`back to text <dir3-1-1>`)

### Exercise 2

Look at the 2-dimensional array `x_2d`.

Does the inner-most index correspond to rows or columns? What does the
outer-most index correspond to?

Write your thoughts.

({ref}`back to text <dir3-1-2>`)

### Exercise 3

What would you do to extract the array `[[5, 6], [50, 60]]`?

({ref}`back to text <dir3-1-3>`)

### Exercise 4

Do you recall what multiplication by an integer did for lists?

How does this differ?

({ref}`back to text <dir3-1-4>`)

### Exercise 5

Let's revisit a bond pricing example we saw in {doc}`Control flow <../python_fundamentals/control_flow>`.

Recall that the equation for pricing a bond with coupon payment $C$,
face value $M$, yield to maturity $i$, and periods to maturity
$N$ is

$$
\begin{align*}
    P &= \left(\sum_{n=1}^N \frac{C}{(i+1)^n}\right) + \frac{M}{(1+i)^N} \\
      &= C \left(\frac{1 - (1+i)^{-N}}{i} \right) + M(1+i)^{-N}
\end{align*}
$$

In the code cell below, we have defined variables for `i`, `M` and `C`.

You have two tasks:

1. Define a numpy array `N` that contains all maturities between 1 and 10 

    ```{hint}
    look at the `np.arange` function.
    ```

1. Using the equation above, determine the bond prices of all maturity levels in your array.

```{code-cell} python
i = 0.03
M = 100
C = 5

# Define array here

# price bonds here
```

({ref}`back to text <dir3-1-5>`)