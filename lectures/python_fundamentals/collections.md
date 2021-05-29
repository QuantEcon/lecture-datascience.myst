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

# Collections

**Prerequisites**

- {doc}`Core data types <basics>`

**Outcomes**

- Ordered Collections
    - Know what a list is and a tuple is
    - Know how to tell a list from a tuple
    - Understand the `range`, `zip` and `enumerate` functions
    - Be able to use common list methods like `append`, `sort`,
      and `reverse`
- Associative Collections
    - Understand what a `dict` is
    - Know the distinction between a dicts keys and values
    - Understand when `dict`s are useful
    - Be familiar with common `dict` methods
- Sets  (optional)
    - Know what a set is
    - Understand how a set differs from a list and a tuple
    - Know when to use a set vs a list or a tuple



## Ordered Collections

### Lists

A Python list is an ordered collection of items.

We can create lists using the following syntax

```{code-block} python
[item1, item2, ...,  itemN]
```

where the `...` represents any number of additional items.

Each item can be of any type.

Let's create some lists.

```{code-cell} python
# created, but not assigned to a variable
[2.0, 9.1, 12.5]
```

```{code-cell} python
# stored as the variable `x`
x = [2.0, 9.1, 12.5]
print("x has type", type(x))
x
```

#### What Can We Do with Lists?

We can access items in a list called `mylist` using `mylist[N]`
where `N` is an integer.

Note: Anytime that we use the syntax `x[i]` we are doing what is
called indexing -- it means that we are selecting a particular element
of a *collection* `x`.

```{code-cell} python
x[1]
```

Wait? Why did `x[1]` return `9.1` when the first element in x is
actually `2.0`?

This happened because Python starts counting at zero!

Lets repeat that one more time for emphasis **Python starts counting at zero**!

To access the first element of x we must use `x[0]`:

```{code-cell} python
x[0]
```

We can also determine how many items are in a list using the `len` function.

```{code-cell} python
len(x)
```

What happens if we try to index with a number higher than the number of
items in a list?

```{code-cell} python
# uncomment the line below and run
# x[4]
```

We can check if a list contains an element using the `in` keyword.

```{code-cell} python
2.0 in x
```

```{code-cell} python
1.5 in x
```

For our list `x`, other common operations we might want to do are...

```{code-cell} python
x.reverse()
x
```

```{code-cell} python
number_list = [10, 25, 42, 1.0]
print(number_list)
number_list.sort()
print(number_list)
```

Note that in order to `sort`, we had to have all elements in our list
be numbers (`int` and `float`), more on this {ref}`below <inhomogenous_lists>`.

We could actually do the same with a list of strings. In this case, `sort`
will put the items in alphabetical order.

```{code-cell} python
str_list = ["NY", "AZ", "TX"]
print(str_list)
str_list.sort()
print(str_list)
```

The `append` method adds an element to the end of existing list.

```{code-cell} python
num_list = [10, 25, 42, 8]
print(num_list)
num_list.append(10)
print(num_list)
```

However, if you call `append` with a list, it adds a `list` to the end,
rather than the numbers in that list.

```{code-cell} python
num_list = [10, 25, 42, 8]
print(num_list)
num_list.append([20, 4])
print(num_list)
```

To combine the lists instead...

```{code-cell} python
num_list = [10, 25, 42, 8]
print(num_list)
num_list.extend([20, 4])
print(num_list)
```

````{admonition} Exercise
:name: dir2-2-1

See exercise 1 in the {ref}`exercise list <ex2-2>`.
````

(inhomogenous_lists)=
### Lists of Different Types

While most examples above have all used a `list` with
a single type of variable, this is not required.

Let's carefully make a small change to the first example: replace `2.0` with `2`

```{code-cell} python
x = [2, 9.1, 12.5]
```

This behavior is identical for many operations you might
apply to a list.

```{code-cell} python
import numpy as np
x = [2, 9.1, 12.5]
np.mean(x) == sum(x)/len(x)
```

Here we have also introduced a new {ref}`module <modules>`,
{doc}`Numpy <../scientific/index>`, which provides many functions
for working with numeric data.

Taking this further, we can put completely different types of elements
inside of a list.

```{code-cell} python
# stored as the variable `x`
x = [2, "hello", 3.0]
print("x has type", type(x))
x
```

To see the types of individual elements in the list:

```{code-cell} python
print(f"type(x[0]) = {type(x[0])}, type(x[0]) = {type(x[1])}, type(x[2]) = {type(x[2])}")
```

While no programming limitations prevent this, you should be careful if you write code
with different numeric and non-numeric types in the same list.

For example, if the types within the list cannot be compared, then how could you sort the elements of the list? (i.e. How do you determine whether the string "hello" is less than the integer 2, "hello" < 2?)

```{code-cell} python
x = [2, "hello", 3.0]
# uncomment the line below and see what happens!
# x.sort()
```

A few key exceptions to this general rule are:

- Lists with both integers and floating points are less error-prone
  (since mathematical code using the list would work with both types).
- When working with lists and data, you may want to represent missing
  values with a different type than the existing values.

### The `range` Function

One function you will see often in Python is the `range` function.

It has three versions:

1. `range(N)`: goes from 0 to N-1
1. `range(a, N)`: goes from a to N-1
1. `range(a, N, d)`: goes from a to N-1, counting by d

When we call the `range` function, we get back something that has type `range`:

```{code-cell} python
r = range(5)
print("type(r)", type(r))
```

To turn the `range` into a list:

```{code-cell} python
list(r)
```

````{admonition} Exercise
:name: dir2-2-2

See exercise 2 in the {ref}`exercise list <ex2-2>`.
````

### What are Tuples?

Tuples are very similar to lists and hold ordered collections of items.

However, tuples and lists have three main differences:

1. Tuples are created using parenthesis — `(` and `)` — instead of
   square brackets — `[` and `]`.
1. Tuples are *immutable*, which is a fancy computer science word
   meaning that they can't be changed or altered after they are created.
1. Tuples and multiple return values
   from functions are tightly connected, as we will see in {doc}`functions <functions>`.

```{code-cell} python
t = (1, "hello", 3.0)
print("t is a", type(t))
t
```

We can *convert* a list to a tuple by calling the `tuple` function on
a list.

```{code-cell} python
print("x is a", type(x))
print("tuple(x) is a", type(tuple(x)))
tuple(x)
```

We can also convert a tuple to a list using the list function.

```{code-cell} python
list(t)
```

As with a list, we access items in a tuple `t` using `t[N]` where
`N` is an int.

```{code-cell} python
t[0]  # still start counting at 0
```

```{code-cell} python
t[2]
```


````{admonition} Exercise
:name: dir2-2-3

See exercise 3 in the {ref}`exercise list <ex2-2>`.
````


Tuples (and lists) can be unpacked directly into variables.

```{code-cell} python
x, y = (1, "test")
print(f"x = {x}, y = {y}")
```

This will be a convenient way to work with functions returning
multiple values, as well as within {doc}`comprehensions and loops <control_flow>`.

### List vs Tuple: Which to Use?

Should you use a list or tuple?

This depends on what you are storing, whether you might need to reorder the elements,
or whether you'd add
new elements without a complete reinterpretation of the
underlying data.

For example, take data representing the GDP (in trillions) and population
(in billions) for China in 2015.

```{code-cell} python
china_data_2015 = ("China", 2015, 11.06, 1.371)

print(china_data_2015)
```

In this case, we have used a tuple since: (a) ordering would
be meaningless; and (b) adding more data would require a
reinterpretation of the whole data structure.

On the other hand, consider a list of GDP in China between
2013 and 2015.

```{code-cell} python
gdp_data = [9.607, 10.48, 11.06]
print(gdp_data)
```

In this case, we have used a list, since adding on a new
element to the end of the list for GDP in 2016 would make
complete sense.

Along these lines, collecting data on China for different
years may make sense as a list of tuples (e.g. year, GDP,
and population -- although we will see better ways to store this sort of data
in the {doc}`Pandas <../pandas/index>` section).

```{code-cell} python
china_data = [(2015, 11.06, 1.371), (2014, 10.48, 1.364), (2013, 9.607, 1.357)]
print(china_data)
```

In general, a rule of thumb is to use a list unless you *need* to use a tuple.

Key criteria for tuple use are when you want to:

- ensure the *order* of elements can't change
- ensure the actual values of the elements can't
  change
- use the collection as a key in a dict (we will learn what this
  means {ref}`soon <dictionaries>`)

### `zip` and `enumerate`

Two functions that can be extremely useful are `zip` and `enumerate`.

Both of these functions are best understood by example, so let's see
them in action and then talk about what they do.

```{code-cell} python
gdp_data = [9.607, 10.48, 11.06]
years = [2013, 2014, 2015]
z = zip(years, gdp_data)
print("type(z)", type(z))
```

To see what is inside `z`, let's convert it to a list.

```{code-cell} python
list(z)
```

Notice that we now have a list where each item is a tuple.

Within each tuple, we have one item from each of the collections we
passed to the zip function.

In particular, the first item in `z` contains the first item from
`[2013, 2014, 2015]` and the first item from `[9.607, 10.48, 11.06]`.

The second item in `z` contains the second item from each collection
and so on.

We can access an element in this and then unpack the resulting
tuple directly into variables.

```{code-cell} python
l = list(zip(years, gdp_data))
x, y = l[0]
print(f"year = {x}, GDP = {y}")
```

Now let's experiment with `enumerate`.

```{code-cell} python
e = enumerate(["a", "b", "c"])
print("type(e)", type(e))
e
```

Again, we call `list(e)` to see what is inside.

```{code-cell} python
list(e)
```

We again have a list of tuples, but this time, the first element in each
tuple is the *index* of the second tuple element in the initial
collection.

Notice that the third item is `(2, 'c')` because
`["a", "b", "c"][2]` is `'c'`


````{admonition} Exercise
:name: dir2-2-4

See exercise 4 in the {ref}`exercise list <ex2-2>`.
````

An important quirk of some iterable types that are not lists (such as the above `zip`) is that
you cannot convert the same type to a list twice.

This is because `zip`, `enumerate`, and `range` produce what is called a generator.

A generator will only produce each of its elements a single time, so if you call `list` on the same
generator a second time, it will not have any elements to iterate over anymore.

For more information, refer to the [Python documentation](https://docs.python.org/3/howto/functional.html#generators).

```{code-cell} python
gdp_data = [9.607, 10.48, 11.06]
years = [2013, 2014, 2015]
z = zip(years, gdp_data)
l = list(z)
print(l)
m = list(z)
print(m)
```

## Associative Collections
(dictionaries)=
### Dictionaries

A dictionary (or dict) associates `key`s with `value`s.

It will feel similar to a dictionary for words, where the keys are words and
the values are the associated definitions.

The most common way to create a `dict` is to use curly braces — `{`
and `}` — like this:

```{code-block} python
{"key1": value1, "key2": value2, ..., "keyN": valueN}
```

where the `...` indicates that we can have any number of additional
terms.

The crucial part of the syntax is that each key-value pair is written
`key: value` and that these pairs are separated by commas — `,`.

Let's see an example using our aggregate data on China in 2015.

```{code-cell} python
china_data = {"country": "China", "year": 2015, "GDP" : 11.06, "population": 1.371}
print(china_data)
```

Unlike our above example using a `tuple`, a `dict` allows us to
associate a name with each field, rather than having to remember the
order within the tuple.

Often, code that makes a dict is easier to read if we put each
`key: value` pair on its own line. (Recall our earlier comment on
using whitespace effectively to improve readability!)

The code below is equivalent to what we saw above.

```{code-cell} python
china_data = {
    "country": "China",
    "year": 2015,
    "GDP" : 11.06,
    "population": 1.371
}
```

Most often, the keys (e.g. "country", "year", "GDP", and "population")
will be strings, but we could also use numbers (`int`, or
`float`) or even tuples (or, rarely, a combination of types).

The values can be **any** type and different from each other.


````{admonition} Exercise
:name: dir2-2-5

See exercise 5 in the {ref}`exercise list <ex2-2>`.
````

This next example is meant to emphasize how values can be
*anything* -- including another dictionary.

```{code-cell} python
companies = {"AAPL": {"bid": 175.96, "ask": 175.98},
             "GE": {"bid": 1047.03, "ask": 1048.40},
             "TVIX": {"bid": 8.38, "ask": 8.40}}
print(companies)
```

#### Getting, Setting, and Updating dict Items

We can now ask Python to tell us the value for a particular key by using
the syntax `d[k]`,  where `d` is our `dict` and `k` is the key for which we want to
find the value.

For example,

```{code-cell} python
print(china_data["year"])
print(f"country = {china_data['country']}, population = {china_data['population']}")
```

Note: when inside of a formatting string, you can use `'` instead of `"` as above
to ensure the formatting still works with the embedded code.

If we ask for the value of a key that is not in the dict, we will get an error.

```{code-cell} python
# uncomment the line below to see the error
# china_data["inflation"]
```

We can also add new items to a dict using the syntax `d[new_key] = new_value`.

Let's see some examples.

```{code-cell} python
print(china_data)
china_data["unemployment"] = "4.05%"
print(china_data)
```

To update the value, we use assignment in the same way (which will
create the key and value as required).

```{code-cell} python
print(china_data)
china_data["unemployment"] = "4.051%"
print(china_data)
```

Or we could change the type.

```{code-cell} python
china_data["unemployment"] = 4.051
print(china_data)
```


````{admonition} Exercise
:name: dir2-2-6

See exercise 6 in the {ref}`exercise list <ex2-2>`.
````

#### Common `dict` Functionality

We can do some common things with dicts.

We will demonstrate them with examples below.

```{code-cell} python
# number of key-value pairs in a dict
len(china_data)
```

```{code-cell} python
# get a list of all the keys
list(china_data.keys())
```

```{code-cell} python
# get a list of all the values
list(china_data.values())
```

```{code-cell} python
more_china_data = {"irrigated_land": 690_070, "top_religions": {"buddhist": 18.2, "christian" : 5.1, "muslim": 1.8}}

# Add all key-value pairs in mydict2 to mydict.
# if the key already appears in mydict, overwrite the
# value with the value in mydict2
china_data.update(more_china_data)
china_data
```

```{code-cell} python
# Get the value associated with a key or return a default value
# use this to avoid the NameError we saw above if you have a reasonable
# default value
china_data.get("irrigated_land", "Data Not Available")
```

```{code-cell} python
china_data.get("death_rate", "Data Not Available")
```


````{admonition} Exercise
:name: dir2-2-7

See exercise 7 in the {ref}`exercise list <ex2-2>`.
````


````{admonition} Exercise
:name: dir2-2-8

See exercise 8 in the {ref}`exercise list <ex2-2>`.
````

### Sets (Optional)

Python has an additional way to represent collections of items: sets.

Sets come up infrequently, but you should be aware of them.

If you are familiar with the mathematical concept of sets, then you will
understand the majority of Python sets already.

If you don't know the math behind sets, don't worry: we'll cover the
basics of Python's sets here.

A set is an *unordered* collection of *unique* elements.

The syntax for creating a set uses curly bracket `{` and `}`.

```{code-block} python
{item1, item2, ..., itemN}
```

Here is an example.

```{code-cell} python
s = {1, "hello", 3.0}
print("s has type", type(s))
s
```

````{admonition} Exercise
:name: dir2-2-9

See exercise 9 in the {ref}`exercise list <ex2-2>`.
````

As with lists and tuples, we can check if something is `in` the set
and check the set's length:

```{code-cell} python
print("len(s) =", len(s))
"hello" in s
```

Unlike lists and tuples, we can't extract elements of a set `s` using
`s[N]` where `N` is a number.

```{code-cell} python
---
tags: [raises-exception]
---
# Uncomment the line below to see what happens
# s[1]
```

This is because sets are not ordered, so the notion of getting the
second element (`s[1]`) is not well defined.

We add elements to a set `s` using `s.add`.

```{code-cell} python
s.add(100)
s
```

```{code-cell} python
s.add("hello") # nothing happens, why?
s
```

We can also do set operations.

Consider the set `s` from above and the set
`s2 = {"hello", "world"}`.

- `s.union(s2)`: returns a set with all elements in either `s` or
  `s2`
- `s.intersection(s2)`: returns a set with all elements in both `s`
  and `s2`
- `s.difference(s2)`: returns a set with all elements in `s` that
  aren't in `s2`
- `s.symmetric_difference(s2)`: returns a set with all elements in
  only one of `s` and `s2`


````{admonition} Exercise
:name: dir2-2-10

See exercise 10 in the {ref}`exercise list <ex2-2>`.
````

As with tuples and lists, a `set` function can convert other
collections to sets.

```{code-cell} python
x = [1, 2, 3, 1]
set(x)
```

```{code-cell} python
t = (1, 2, 3, 1)
set(t)
```

Likewise, we can convert sets to lists and tuples.

```{code-cell} python
list(s)
```

```{code-cell} python
tuple(s)
```

(ex2-2)=
## Exercises

### Exercise 1

In the first cell, try `y.append(z)`.

In the second cell try `y.extend(z)`.

Explain the behavior.

```{hint}
When you are trying to explain use `y.append?` and `y.extend?` to
see a description of what these methods are supposed to do.
```

```{code-cell} python
:tags: ["remove-output"]
y = ["a", "b", "c"]
z = [1, 2, 3]
# your code here
print(y)
```

```{code-cell} python
:tags: ["remove-output"]
y = ["a", "b", "c"]
z = [1, 2, 3]
# your code here
print(y)
```
({ref}`back to text <dir2-2-1>`)

### Exercise 2

Experiment with the other two versions of the `range` function.

```{code-cell} python
# try list(range(a, N)) -- you pick `a` and `N`
```

```{code-cell} python
# try list(range(a, N, d)) -- you pick `a`, `N`, and `d`
```
({ref}`back to text <dir2-2-2>`)

### Exercise 3

Verify that tuples are indeed immutable by attempting the following:

- Changing the first element of `t` to be `100`
- Appending a new element `"!!"` to the end of `t` (remember with a
  list `x` we would use `x.append("!!")` to do this
- Sorting `t`
- Reversing `t`

```{code-cell} python
# change first element of t
```

```{code-cell} python
# appending to t
```

```{code-cell} python
# sorting t
```

```{code-cell} python
# reversing t
```

({ref}`back to text <dir2-2-3>`)

### Exercise 4

**Challenging** For the tuple `foo` below, use a combination of `zip`,
`range`, and `len` to mimic `enumerate(foo)`.

Verify that your proposed solution is correct by converting each to a list
and checking equality with `==`.

```{hint}
You can see what the answer should look like by starting with
`list(enumerate(foo))`.
```

```{code-cell} python
foo = ("good", "luck!")
```

({ref}`back to text <dir2-2-4>`)

### Exercise 5

Create a new dict which associates stock tickers with its stock price.

Here are some tickers and a price.

- AAPL: 175.96
- GOOGL: 1047.43
- TVIX: 8.38

```{code-cell} python
# your code here
```
({ref}`back to text <dir2-2-5>`)

### Exercise 6

Look at the [World Factbook for Australia](https://www.cia.gov/the-world-factbook/countries/australia)
and create a dictionary with data containing the following types:
float, string, integer, list, and dict.  Choose any data you wish.

To confirm, you should have a dictionary that you identified via a key.

```{code-cell} python
# your code here
```

({ref}`back to text <dir2-2-6>`)

### Exercise 7

Use Jupyter's help facilities to learn how to use the `pop` method to
remove the key `"irrigated_land"` (and its value) from the dict.

```{code-cell} python
# uncomment and use the Inspector or ?
#china_data.pop()
```

({ref}`back to text <dir2-2-7>`)

### Exercise 8

Explain what happens to the value you popped.

Experiment with calling `pop` twice.

```{code-cell} python
# your code here
```
({ref}`back to text <dir2-2-8>`)

### Exercise 9

Try creating a set with repeated elements (e.g. `{1, 2, 1, 2, 1, 2}`).

What happens?

Why?

```{code-cell} python
# your code here
```
({ref}`back to text <dir2-2-9>`)

### Exercise 10

Test out two of the operations described above using the original set we
created, `s`, and the set created below `s2`.

```{code-cell} python
s2 = {"hello", "world"}
```

```{code-cell} python
# Operation 1
```

```{code-cell} python
# Operation 2
```

({ref}`back to text <dir2-2-10>`)
