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

# Basics

**Prerequisites**

- {doc}`Getting Started <../introduction/getting_started>`

**Outcomes**

- Programming concepts
    - Understand variable assignment
    - Know what a function is and how to figure out what it does
    - Be able to use tab completion
- Numbers in Python
    - Understand how Python represents numbers
    - Know the distinction between `int` and `float`
    - Be familiar with various binary operators for numbers
    - Introduction to the `math` library
- Text (strings) in Python
    - Understand what a string is and when it is useful
    - Learn some of the methods associated with strings
    - Combining strings and output
- True and False (booleans) in Python
    - Understand what a boolean is
    - Become familiar with all binary operators that return booleans


## First Steps

We are ready to begin writing code!

In this section, we will teach you some basic concepts of programming
and where to search for help.

### Variable Assignment

The first thing we will learn is the idea of *variable assignment*.

Variable assignment associates a value to a variable.

Below, we assign the value "Hello World" to the variable `x`

```{code-cell} python
x = "Hello World"
```

Once we have assigned a value to a variable, Python will remember this
variable as long as the *current* session of Python is still running.

Notice how writing `x` into the prompt below outputs the value
"Hello World".

```{code-cell} python
x
```

However, Python returns an error if we ask it about variables that have not yet
been created.

```{code-cell} python
# uncomment (delete the # and the space) the line below and run
# y
```

It is also useful to understand the order in which operations happen.

First, the right side of the equal sign is computed.

Then, that computed value is stored as the variable to the left of the
equal sign.

````{admonition} Exercise
:name: dir1

See exercise 1 in the {ref}`exercise list <ex>`.
````

Keep in mind that the variable binds a name to something stored in memory.

The name can even be bound to a value of a completely different type.

```{code-cell} python
x = 2
print(x)
x = "something else"
print(x)
```

### Code Comments

Comments are short notes that you leave for yourself and for others who read your
code.

They should be used to explain what the code does.

A comment is made with the `#`. Python ignores everything in a line that follows a `#`.

Let's practice making some comments.

```{code-cell} python
i = 1  # Assign the value 1 to variable i
j = 2  # Assign the value 2 to variable j

# We add i and j below this line
i + j
```

## Functions

Functions are processes that take an input (or inputs) and produce an output.

If we had a function called `f` that took two arguments `x` and
`y`, we would write `f(x, y)` to use the function.

For example, the function `print` simply prints whatever it is given.
Recall the variable we created called `x`.

```{code-cell} python
print(x)
```

### Getting Help

We can figure out what a function does by asking for help.

In Jupyter notebooks, this is done by placing a `?` after the function
name (without using parenthesis) and evaluating the cell.

For example, we can ask for help on the print function by writing
`print?`.

Depending on how you launched Jupyter, this will either launch

- JupyterLab: display the help in text below the cell.
- Classic Jupyter Notebooks: display a new panel at the bottom of your
  screen.  You can exit this panel by hitting the escape key or clicking the x at
  the top right of the panel.

```{code-cell} python
# print? # remove the comment and <Shift-Enter>
```

````{admonition} Exercise
:name: dir2

See exercise 2 in the {ref}`exercise list <ex>`.
````

JupyterLab also has a "Contextual Help" (previously called "Inspector") window.  To use,

- Go to the Commands and choose Contextual Help (or Inspector), or select `<Ctrl-I>` (`<Cmd-I>` for OSX users).
- Drag the new inspector pain to dock in the screen next to your code.
- Then, type `print` or any other function
  into a cell and see the help.
  
```{code-cell} python
# len? # remove the comment and <Shift-Enter>
```
  

We will learn much more about functions, including how to write our own, in a
future lecture.

## Objects and Types

Everything in Python is an *object*.

Objects are "things" that contain 1) data and 2) functions that can operate on
the data.

Sometimes we refer to the functions inside an object as *methods*.

We can investigate what data is inside an object and which methods
it supports by typing `.` after that particular variable, then
hitting `TAB`.

It should then list data and method names to the right of the
variable name like this:

```{figure} https://datascience.quantecon.org/assets/_static/python_fundamentals/introspection.png
:alt: introspection.png
```

You can scroll through this list by using the up and down arrows.

We often refer to this as "tab completion" or "introspection".

Let's do this together below. Keep going down until you find the method
`split`.

```{code-cell} python
# Type a period after `x` and then press TAB.
x
```

Once you have found the method `split`, you can use the method by adding
parenthesis after it.

Let's call the `split` method, which doesn't have any other required
parameters. (Quiz: how would we check that?)

```{code-cell} python
x.split()
```

We often want to identify what kind of object some value is--
called its "type".

A "type" is an abstraction which defines a set of behavior for any
"instance" of that type i.e. `2.0` and `3.0` are instances
of `float`, where `float` has a set of particular common behaviors.

In particular, the type determines:

- the available data for any "instance" of the type (where each
  instance may have different values of the data).
- the methods that can be applied on the object and its data.

We can figure this out by using the `type` function.

The `type` function takes a single argument and outputs the type of
that argument.

```{code-cell} python
type(3)
```

```{code-cell} python
type("Hello World")
```

```{code-cell} python
type([1, 2, 3])
```

We will learn more about each of these types (and others!) and how to use them
soon, so stay tuned!

(modules)=
## Modules

Python takes a modular approach to tools.

By this we mean that sets of related tools are bundled together into *packages*.
(You may also hear the term modules to describe the same thing.)

For example:

- `pandas` is a package that implements the tools necessary to do
  scalable data analysis.
- `matplotlib` is a package that implements visualization tools.
- `requests` and `urllib` are packages that allow Python to
  interface with the internet.

As we move further into the class, being able to
access these packages will become very important.

We can bring a package's functionality into our current Python session
by writing

```{code-block} python
import package
```

Once we have done this, any function or object from that package can
be accessed by using `package.name`.

Here's an example.

```{code-cell} python
import sys   # for dealing with your computer's system
sys.version  # information about the Python version in use
```

````{admonition} Exercise
:name: dir3

See exercise 3 in the {ref}`exercise list <ex>`.
````

### Module Aliases

Some packages have long names (see `matplotlib`, for example) which
makes accessing the package functionality somewhat inconvenient.

To ease this burden, Python allows us to give aliases or "nicknames" to packages.

For example we can write:

```{code-block} python
import package as p
```

This statement allows us to access the packages functionality as
`p.function_name` rather than `package.function_name`.

Some common aliases for packages are

- `import pandas as pd`
- `import numpy as np`
- `import matplotlib as mpl`
- `import datetime as dt`

While you *can* choose any name for an alias, we suggest that you stick
to the common ones.

You will learn what these common ones are over time.

````{admonition} Exercise
:name: dir4

See exercise 4 in the {ref}`exercise list <ex>`.
````

## Good Code Habits

A common saying in the software engineering world is:

> Always code as if the guy who ends up maintaining your code will be
> a violent psychopath who knows where you live. Code for readability.

This might be a dramatic take, but the most important feature of your code
after correctness is readability.

We encourage you to do **everything** in your power to make your code as readable as possible.

Here are some suggestions for how to do so:

- Comment frequently. Leaving short notes not only will help others who
  use your code, but will also help you interpret your code
  after some time has passed.
- **Anytime** you use a comma, place a space immediately afterwards.
- Whitespace is your friend. Don't write line after line of code -- use
  blank lines to break it up.
- Don't let your lines run too long. Some people reading your code will
  be on a laptop, so you want to ensure that they don't need to scroll horizontally
  and right to read your code. We recommend no more than 80 characters per line.

## Numbers

Python has two types of numbers.

1. Integer (`int`): These can only take the values of the integers
   i.e. $\{\dots, -2, -1, 0, 1, 2, \dots\}$
1. Floating Point Number (`float`): Think of these as any real number
   such as $1.0$, $3.1415$, or $-100.022358923223$...

The easiest way to differentiate these types of numbers is to find a decimal place
after the number.

A float will have a decimal place, but an integer will not.

Below, we assign integers to the variables `xi` and `zi` and assign
floating point numbers to the variables `xf` and `zf`.

```{code-cell} python
xi = 1
xf = 1.0
zi = 123
zf = 1230.5  # Notice -- There are no commas!
zf2 = 1_230.5  # If needed, we use `_` to separate numbers for readability
```

````{admonition} Exercise
:name: dir5

See exercise 5 in the {ref}`exercise list <ex>`.
````

### Python as a Calculator

You can use Python to perform mathematical calculations.

```{code-cell} python
a = 4
b = 2

print("a + b is", a + b)
print("a - b is", a - b)
print("a * b is", a * b)
print("a / b is", a / b)
print("a ** b is", a**b)
print("a ^ b is", a^b)
```

You likely could have guessed all except the last two.

**Warning**: Python uses `**`, not `^`, for exponentiation (raising a number
to a power)!

Notice also that above `+`, `-` and `**` all returned an integer
type, but `/` converted the result to a float.

When possible, operations between integers return an integer type.

All operations involving a float will result in a float.

```{code-cell} python
a = 4
b = 2.0

print("a + b is", a + b)
print("a - b is", a - b)
print("a * b is", a * b)
print("a / b is", a / b)
print("a ** b is", a**b)
```

We can also chain together operations.

When doing this, Python follows the standard [order of operations](https://en.wikipedia.org/wiki/Order_of_operations) — parenthesis, exponents,
multiplication and division, followed by addition and subtraction.

For example,

```{code-cell} python
x = 2.0
y = 3.0
z1 = x + y * x
z2 = (x + y) * x
```

What do you think `z1` is?

How about `z2`?

````{admonition} Exercise
:name: dir6

See exercise 6 in the {ref}`exercise list <ex>`.
````

### Other Math Functions

We often want to use other math functions on our numbers. Let's try to
calculate sin(2.5).

```{code-cell} python
sin(2.5)
```

As seen above, Python complains that `sin` isn't defined.

The problem here is that the `sin` function -- as well as many other
standard math functions -- are contained in the `math` package.

We must begin by importing the math package.

```{code-cell} python
import math
```

Now, we can use `math.[TAB]` to see what functions are available to us.

```{code-cell} python
# uncomment, add a period (`.`) and pres TAB
# math
```

```{code-cell} python
# found math.sin!
math.sin(2.5)
```

````{admonition} Exercise
:name: dir7

See exercise 7 in the {ref}`exercise list <ex>`.
````

#### Floor/Modulus Division Operators

You are less likely to run into the following operators, but understanding
that they exist is useful.

For two numbers assigned to the variables `x` and `y`,

- Floor division: `x // y`
- Modulus division: `x % y`

Remember when you first learned how to do division and you were asked to talk about the quotient
and the remainder?

That's what these operators correspond to...

Floor division returns the number of times the divisor goes into the dividend (the quotient)
and modulus division returns the remainder.

An example would be 37 divided by 7:

- Floor division would return 5 (7 * 5 = 35)
- Modulus division would return 2 (2 + 35 = 37)

Try it!

```{code-cell} python
37 // 7
```

```{code-cell} python
37 % 7
```

## Strings

Textual information is stored in a data type called a string.

To denote that you would like something to be stored as a string, you place it
inside of quotation marks.

For example,

```{code-block} python
"this is a string"  # Notice the quotation marks
'this is a string'  # Notice the quotation marks
this is not a string  # No quotation marks
```

You can use either `"` or `'` to create a string. Just make sure
that you start and end the string with the same one!

Notice that if we ask Python to tell us the type of a string, it abbreviates
its answer to `str`.

```{code-cell} python
type("this is a string")
```

````{admonition} Exercise
:name: dir8

See exercise 8 in the {ref}`exercise list <ex>`.
````


### String Operations

Some of the arithmetic operators we saw in the numbers lecture also work
on strings:

- Put two strings together: `x + y`.
- Repeat the string `x` a total of `n` times: `n * x` (or `x * n`).

```{code-cell} python
x = "Hello"
y = "World"
```

```{code-cell} python
x + y
```

```{code-cell} python
3 * x
```

What happens if we try `*` with two strings, or `-` or `/`?

The best way to find out is to try it!

```{code-cell} python
a = "1"
b = "2"
a * b
```

```{code-cell} python
a - b
```

````{admonition} Exercise
:name: dir9

See exercise 9 in the {ref}`exercise list <ex>`.
````

### String Methods

We can use many *methods* to manipulate strings.

We will not be able to cover all of them here, but let's take a look at
some of the most useful ones.

```{code-cell} python
x
```

```{code-cell} python
x.lower()  # Makes all letters lower case
```

```{code-cell} python
x.upper()  # Makes all letters upper case
```

```{code-cell} python
x.count("l")  # Counts number of a particular string
```

```{code-cell} python
x.count("ll")
```

````{admonition} Exercise
:name: dir10

See exercise 10 in the {ref}`exercise list <ex>`.
````

````{admonition} Exercise
:name: dir11

See exercise 11 in the {ref}`exercise list <ex>`.
````

### String Formatting

Sometimes we'd like to reuse some portion of a string repeatedly, but
still make some relatively small changes at each usage.

We can do this with *string formatting*, which done by using `{}` as a
*placeholder* where we'd like to change the string, with a variable name
or expression.

Let's look at an example.

```{code-cell} python
country = "Vietnam"
GDP = 223.9
year = 2017
my_string = f"{country} had ${GDP} billion GDP in {year}"
print(my_string)
```

Rather than just substituting a variable name, you can use a calculation
or expression.

```{code-cell} python
print(f"{5}**2 = {5**2}")
```

Or, using our previous example

```{code-cell} python
my_string = f"{country} had ${GDP * 1_000_000} GDP in {year}"
print(my_string)
```

In these cases, the `f` in front of the string causes Python interpolate
any valid expression within the `{}` braces.

````{admonition} Exercise
:name: dir12

See exercise 12 in the {ref}`exercise list <ex>`.
````

Alternatively, to reuse a formatted string, you can call the `format` method (noting that you do **not** put `f` in front).

```{code-cell} python
gdp_string = "{country} had ${GDP} billion in {year}"

gdp_string.format(country = "Vietnam", GDP = 223.9, year = 2017)
```

````{admonition} Exercise
:name: dir13

See exercise 13 in the {ref}`exercise list <ex>`.
````
````{admonition} Exercise
:name: dir14

See exercise 14 in the {ref}`exercise list <ex>`.
````

For more information on what you can do with string formatting (there is *a lot*
that can be done...), see the [official Python documentation](https://docs.python.org/3.6/library/string.html) on the subject.

## Booleans

A boolean is a type that denotes true or false.

As you will soon see in the {doc}`control flow chapter <control_flow>`, using
boolean values allows you to perform or skip operations depending on whether or
not a condition is met.

Let's start by creating some booleans and looking at them.

```{code-cell} python
x = True
y = False

type(x)
```

```{code-cell} python
x
```

```{code-cell} python
y
```

### Comparison Operators

Rather than directly write `True` or `False`, you will usually
create booleans by making a comparison.

For example, you might want to evaluate whether the price of a particular asset
is greater than or less than some price.

For two variables `x` and `y`, we can do the following comparisons:

- Greater than: `x > y`
- Less than: `x < y`
- Equal to: `==`
- Greater than or equal to: `x >= y`
- Less than or equal to: `x <= y`

We demonstrate these below.

```{code-cell} python
a = 4
b = 2

print("a > b", "is", a > b)
print("a < b", "is", a < b)
print("a == b", "is", a == b)
print("a >= b", "is", a >= b)
print("a <= b", "is", a <= b)
```

### Negation

Occasionally, determining whether a statement is
"not true" or "not false" is more convenient than simply "true" or "false".

This is known as *negating* a statement.

In Python, we can negate a boolean using the word `not`.

```{code-cell} python
not False
```

```{code-cell} python
not True
```

### Multiple Comparisons (and/or)

Sometimes we need to evaluate multiple comparisons at once.

This is done by using the words `and` and `or`.

However, these are the "mathematical" *and*s and *or*s -- so they
don't carry the same meaning as you'd use them in colloquial English.

- `a and b` is true only when **both** `a` and `b` are true.
- `a or b` is true whenever at least one of `a` or `b` is true.

For example

- The statement "I will accept the new job if the salary is higher
  *and* I receive more vacation days" means that you would only accept
  the new job if you both receive a higher salary and are given more
  vacation days.
- The statement "I will accept the new job if the salary is higher *or*
  I receive more vacation days" means that you would accept the job if
  (1) they raised your salary, (2) you are given more vacation days, or
  (3) they raise your salary and give you more vacation days.

Let's see some examples.

```{code-cell} python
True and False
```

```{code-cell} python
True and True
```

```{code-cell} python
True or False
```

```{code-cell} python
False or False
```

```{code-cell} python
# Can chain multiple comparisons together.
True and (False or True)
```

````{admonition} Exercise
:name: dir15

See exercise 15 in the {ref}`exercise list <ex>`.
````

### `all` and `any`

We have seen how we can use the words `and` and `or` to process two booleans
at a time.

The functions `all` and `any` allow us to process an unlimited number of
booleans at once.

`all(bools)` will return `True` if and only if all the booleans in `bools`
is `True` and returns `False` otherwise.

`any(bools)` returns `True` whenever one or more of `bools` is `True`.

The exercise below will give you a chance to practice.

````{admonition} Exercise
:name: dir16

See exercise 16 in the {ref}`exercise list <ex>`.
````

(ex)=
## Exercises

###### Exercise 1

What do you think the value of `z` is after running the code below?

```{code-cell} python
z = 3
z = z + 4
print("z is", z)
```

({ref}`back to text <dir1>`)


###### Exercise 2

Read about out what the `len` function does (by writing len?).

What will it produce if we give it the variable `x`?

Check whether you were right by running the code `len(x)`.

({ref}`back to text <dir2>`)

###### Exercise 3

We can use our introspection skills to investigate a package's contents.

In the cell below, use tab completion to find a function from the `time`
module that will display the **local** time.

Use `time.FUNC_NAME?` (where `FUNC_NAME` is replaced with the
function you found) to see information about that function and
then call the function.

```{hint} 
Look for something to do with the word `local`

```

```{code-cell} python
import time
# your code here -- notice the comment!
```
({ref}`back to text <dir3>`)

###### Exercise 4

Try running `import time as t` in the cell below, then see if you can
call the function you identified above.

Does it work?

({ref}`back to text <dir4>`)

###### Exercise 5

Create the following variables:

- `D`: A floating point number with the value 10,000
- `r`: A floating point number with value 0.025
- `T`: An integer with value 30

We will use them in a later exercise.

```{code-cell} python
# your code here!
```

({ref}`back to text <dir5>`)

###### Exercise 6

Remember the variables we created earlier?

Let's compute the present discounted value of a payment ($D$) made
in $T$ years assuming an interest rate of 2.5%. Save this value to
a new variable called `PDV` and print your output.

```{hint}
The formula is

$$
\text{PDV} = \frac{D}{(1 + r)^T}
$$

```

```{code-cell} python
# your code here
```

({ref}`back to text <dir6>`)

###### Exercise 7

Verify the "trick" where the percent difference ($\frac{x - y}{x}$)
between two numbers close to 1 can be well approximated by the difference
between the log of the two numbers ($\log(x) - \log(y)$).

Use the numbers `x` and `y` below.

```{hint}
you will want to use the
`math.log` function
```

```{code-cell} python
x = 1.05
y = 1.02
```

({ref}`back to text <dir7>`)

###### Exercise 8

The code below is invalid Python code

```{code-cell} markdown
x = 'What's wrong with this string'
```

Can you fix it?

```{hint}
Try creating a code cell below and testing things out until you
find a solution.
```

({ref}`back to text <dir8>`)

###### Exercise 9

Using the variables `x` and `y`, how could you create the sentence
`Hello World`?

```{hint}
Think about how to represent a space as a string.
```

({ref}`back to text <dir9>`)

###### Exercise 10

One of our favorite (and most frequently used) string methods is
`replace`.

It substitutes all occurrences of a particular pattern with a different pattern.

For the variable `test` below, use the `replace` method to change the
`c` to a `d`.

```{hint}
Type `test.replace?` to get some help for how to use the method
replace.
```

```{code-cell} python
test = "abc"
```

({ref}`back to text <dir10>`)

###### Exercise 11

Suppose you are working with price data and encounter the value
`"$6.50"`.

We recognize this as being a number representing the quantity "six dollars and fifty cents."

However, Python interprets the value as the string
`"$6.50"`. (Quiz: why is this a problem? Think about the examples above.)

In this exercise, your task is to convert the variable `price` below
into a number.

```{hint}
Once the string is in a suitable format, you can call write
`float(clean_price)` to make it a number.
```

```{code-cell} python
price = "$6.50"
```

({ref}`back to text <dir11>`)

###### Exercise 12

Lookup a country in [World Bank database](https://data.worldbank.org), and
format a string showing the growth rate of GDP over the last 2 years.

({ref}`back to text <dir12>`)

###### Exercise 13

Instead of hard-coding the values above, try to use the `country`, `GDP` and
`year` variables you previously defined.

({ref}`back to text <dir13>`)

###### Exercise 14

Create a new string and use formatting to produce each of the following
statements

- "The 1st quarter revenue was 110M"
- "The 2nd quarter revenue was 95M"
- "The 3rd quarter revenue was 100M"
- "The 4th quarter revenue was 130M"

({ref}`back to text <dir14>`)

###### Exercise 15

Without typing the commands, determine whether the following statements are
true or false.

Once you have evaluated whether the command is `True` or `False`, run the
code in Python.

```{code-cell} python
x = 2
y = 2
z = 4

# Statement 1
x > z

# Statement 1
x == y

# Statement 3
(x < y) and (x > y)

# Statement 4
(x < y) or (x > y)

# Statement 5
(x <= y) and (x >= y)

# Statement 6
True and ((x < z) or (x < y))
```

```{code-cell} python
# code here!
```

({ref}`back to text <dir15>`)

###### Exercise 16

For each of the code cells below, think carefully about what you expect to
be returned *before* evaluating the cell.

Then evaluate the cell to check your intuitions.

NOTE: For now, do not worry about what the `[` and `]` mean -- they
allow us to create lists which we will learn about in an upcoming lecture.

```{code-cell} python
all([True, True, True])
```

```{code-cell} python
all([False, True, False])
```

```{code-cell} python
all([False, False, False])
```

```{code-cell} python
any([True, True, True])
```

```{code-cell} python
any([False, True, False])
```

```{code-cell} python
any([False, False, False])
```

({ref}`back to text <dir16>`)
