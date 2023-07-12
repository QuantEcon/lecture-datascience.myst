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

# Functions

**Prerequisites**

- {doc}`Getting Started <../introduction/getting_started>`
- {doc}`Basics <basics>`
- {doc}`Collections <collections>`
- {doc}`Control Flow <control_flow>`

**Outcomes**

- Economic Production Functions
    - Understand the basics of production functions in economics
- Functions
    - Know how to define your own function
    - Know how to find and write your own function documentation
    - Know why we use functions
    - Understand scoping rules and blocks


(production_functions)=
## Application: Production Functions

Production functions are useful when modeling the economics of firms producing
goods or the aggregate output in an economy.

Though the term "function" is used in a mathematical sense here, we will be making
tight connections between the programming of mathematical functions and Python
functions.

### Factors of Production

The [factors of production](https://en.wikipedia.org/wiki/Factors_of_production)
are the inputs used in the production of some sort of output.

Some example factors of production include

- [Physical capital](https://en.wikipedia.org/wiki/Physical_capital), e.g.
  machines, buildings, computers, and power stations.
- Labor, e.g. all of the hours of work from different types of employees of a
  firm.
- [Human Capital](https://en.wikipedia.org/wiki/Human_capital), e.g. the
  knowledge of employees within a firm.

A [production function](https://en.wikipedia.org/wiki/Production_function)
maps a set of inputs to the output, e.g. the amount of wheat produced by a
farm, or widgets produced in a factory.

As an example of the notation, we denote the total units of labor and
physical capital used in a factory as $L$ and $K$ respectively.

If we denote the physical output of the factory as $Y$, then a production
function $F$ that transforms labor and capital into output might have the
form:

$$
Y = F(K, L)
$$

(cobb_douglas_example)=
### An Example Production Function

Throughout this lecture, we will use the
[Cobb-Douglas](https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function)
production function to help us understand how to create Python
functions and why they are useful.

The Cobb-Douglas production function has appealing statistical properties when brought to data.

This function is displayed below.

$$
Y = z K^{\alpha} L^{1-\alpha}
$$

The function is parameterized by:

- A parameter $\alpha \in [0,1]$, called the "output elasticity of
  capital".
- A value $z$ called the [Total Factor Productivity](https://en.wikipedia.org/wiki/Total_factor_productivity) (TFP).

## What are (Python) Functions?

In this class, we will often talk about `function`s.

So what is a function?

We like to think of a function as a production line in a
manufacturing plant: we pass zero or more things to it, operations take place in a
set linear sequence, and zero or more things come out.

We use functions for the following purposes:

- **Re-usability**: Writing code to do a specific task just once, and
  reuse the code by calling the function.
- **Organization**: Keep the code for distinct operations separated and
  organized.
- **Sharing/collaboration**: Sharing code across multiple projects or
  sharing pieces of code with collaborators.

## How to Define (Python) Functions?

The basic syntax to create our own function is as follows:

```{code-block} python
def function_name(inputs):
    # step 1
    # step 2
    # ...
    return outputs
```

Here we see two new *keywords*: `def` and `return`.

- `def` is used to tell Python we would like to define a new function.
- `return` is used to tell Python what we would like to **return** from a
  function.

Let's look at an example and then discuss each part:

```{code-cell} python
def mean(numbers):
    total = sum(numbers)
    N = len(numbers)
    answer = total / N

    return answer
```

Here we defined a function `mean` that has one input (`numbers`),
does three steps, and has one output (`answer`).

Let's see what happens when we call this function on the list of numbers
`[1, 2, 3, 4]`.

```{code-cell} python
x = [1, 2, 3, 4]
the_mean = mean(x)
the_mean
```

Additionally, as we saw in the {doc}`control flow <control_flow>` lecture, indentation
controls blocks of code (along with the {ref}`scope <scope>` rules).

To see this, compare a function with no inputs or return values.

```{code-cell} python
def f():
    print("1")
    print("2")
f()
```

With the following change of indentation...

```{code-cell} python
def f():
    print("1")
print("2")
f()
```

(scope)=
### Scope

Notice that we named the input to the function `x` and we called the output
`the_mean`.

When we defined the function, the input was called `numbers` and the output
`answer`... what gives?

This is an example of a programming concept called
[variable scope](http://python-textbok.readthedocs.io/en/1.0/Variables_and_Scope.html).

In Python, functions define their own scope for variables.

In English, this means that regardless of what name we give an input variable (`x` in this example),
the input will always be referred to as `numbers` *inside* the body of the `mean`
function.

It also means that although we called the output `answer` inside of the
function `mean`, that this variable name was only valid inside of our
function.

To use the output of the function, we had to give it our own name (`the_mean`
in this example).

Another point to make here is that the intermediate variables we defined inside
`mean` (`total` and `N`) are only defined inside of the `mean` function
-- we can't access them from outside. We can verify this by trying to see what
the value of `total` is:

```{code-cell} python
def mean(numbers):
    total = sum(numbers)
    N = len(numbers)
    answer = total / N
    return answer # or directly return total / N

# uncomment the line below and execute to see the error
# total
```

This point can be taken even further:  the same name can be bound
to variables inside of blocks of code and in the outer "scope".

```{code-cell} python
x = 4
print(f"x = {x}")
def f():
    x = 5 # a different "x"
    print(f"x = {x}")
f() # calls function
print(f"x = {x}")
```

The final point we want to make about scope is that function inputs and output
don't have to be given a name outside the function.

```{code-cell} python
mean([10, 20, 30])
```

Notice that we didn't name the input or the output, but the function was
called successfully.

Now, we'll use our new knowledge to define a function which computes the output
from a Cobb-Douglas production function with parameters $z = 1$ and
$\alpha = 0.33$ and takes inputs $K$ and $L$.

```{code-cell} python
def cobb_douglas(K, L):

    # Create alpha and z
    z = 1
    alpha = 0.33

    return z * K**alpha * L**(1 - alpha)
```

We can use this function as we did the mean function.

```{code-cell} python
cobb_douglas(1.0, 0.5)
```

(returns_to_scale)=
### Re-using Functions

Economists are often interested in this question: how much does output
change if we modify our inputs?

For example, take a production function $Y_1 = F(K_1,L_1)$ which produces
$Y_1$ units of the goods.

If we then multiply the inputs each by $\gamma$, so that
$K_2 = \gamma K_1$ and $L_2 = \gamma L_1$, then the output is

$$
Y_2 = F(K_2, L_2) = F(\gamma K_1, \gamma L_1)
$$

How does $Y_1$ compare to $Y_2$?

Answering this question involves something called *returns to scale*.

Returns to scale tells us whether our inputs are more or less productive as we
have more of them.

For example, imagine that you run a restaurant. How would you expect the amount
of food you could produce would change if you could build an exact replica of
your restaurant and kitchen and hire the same number of cooks and waiters? You
would probably expect it to double.

If, for any $K, L$, we multiply $K, L$ by a value $\gamma$
then

* If $\frac{Y_2}{Y_1} < \gamma$ then we say the production function has
  decreasing returns to scale.
* If $\frac{Y_2}{Y_1} = \gamma$ then we say the production function has
  constant returns to scale.
* If $\frac{Y_2}{Y_1} > \gamma$ then we say the production function has
  increasing returns to scale.

Let's try it and see what our function is!

```{code-cell} python
y1 = cobb_douglas(1.0, 0.5)
print(y1)
y2 = cobb_douglas(2*1.0, 2*0.5)
print(y2)
```

How did $Y_1$ and $Y_2$ relate?

```{code-cell} python
y2 / y1
```

$Y_2$ was exactly double $Y_1$!

Let's write a function that will compute the returns to scale for different
values of $K$ and $L$.

This is an example of how writing functions can allow us to re-use code
in ways we might not originally anticipate. (You didn't know we'd be
writing a `returns_to_scale` function when we wrote `cobb_douglas`.)

```{code-cell} python
def returns_to_scale(K, L, gamma):
    y1 = cobb_douglas(K, L)
    y2 = cobb_douglas(gamma*K, gamma*L)
    y_ratio = y2 / y1
    return y_ratio / gamma
```

```{code-cell} python
returns_to_scale(1.0, 0.5, 2.0)
```

````{admonition} Exercise
:name: dir2-4-1

See exercise 1 in the {ref}`exercise list <ex2-4>`.
````

It turns out that with a little bit of algebra, we can check that this will
always hold for our {ref}`Cobb-Douglas example <cobb_douglas_example>` above.

To show this, take an arbitrary $K, L$ and multiply the inputs by an
arbitrary $\gamma$.

$$
\begin{aligned}
    F(\gamma K, \gamma L) &= z (\gamma K)^{\alpha} (\gamma L)^{1-\alpha}\\
    &=  z \gamma^{\alpha}\gamma^{1-\alpha} K^{\alpha} L^{1-\alpha}\\
    &= \gamma z K^{\alpha} L^{1-\alpha} = \gamma F(K, L)
\end{aligned}
$$

For an example of a production function that is not CRS, look at a
generalization of the Cobb-Douglas production function that has different
"output elasticities" for the 2 inputs.

$$
Y = z K^{\alpha_1} L^{\alpha_2}
$$

Note that if $\alpha_2 = 1 - \alpha_1$, this is our Cobb-Douglas
production function.

````{admonition} Exercise
:name: dir2-4-2

See exercise 2 in the {ref}`exercise list <ex2-4>`.
````

(marginal_products)=
### Multiple Returns

Another valuable element to analyze on production functions is how
output changes as we change only one of the inputs. We will call this the
marginal product.

For example, compare the output using $K, L$ units of inputs to that with
an $\epsilon$ units of labor.

Then the marginal product of labor (MPL) is defined as

$$
\frac{F(K, L + \varepsilon) - F(K, L)}{\varepsilon}
$$

This tells us how much additional output is created relative to the additional
input. (Spoiler alert: This should look like the definition for a partial
derivative!)

If the input can be divided into small units, then we can use calculus to take
this limit, using the partial derivative of the production function relative to
that input.

In this case, we define the marginal product of labor (MPL) and marginal product
of capital (MPK) as

$$
\begin{aligned}
MPL(K, L) &= \frac{\partial F(K, L)}{\partial L}\\
MPK(K, L) &= \frac{\partial F(K, L)}{\partial K}
\end{aligned}
$$

In the {ref}`Cobb-Douglas <cobb_douglas_example>` example above, this becomes

$$
\begin{aligned}
MPK(K, L) &= z  \alpha \left(\frac{K}{L} \right)^{\alpha - 1}\\
MPL(K, L) &= (1-\alpha) z \left(\frac{K}{L} \right)^{\alpha}\\
\end{aligned}
$$

Let's test it out with Python! We'll also see that we can actually return
multiple things in a Python function.

The syntax for a return statement with multiple items is return item1, item2, ....

In this case, we'll compute both the MPL and the MPK and then return both.

```{code-cell} python
def marginal_products(K, L, epsilon):

    mpl = (cobb_douglas(K, L + epsilon) - cobb_douglas(K, L)) / epsilon
    mpk = (cobb_douglas(K + epsilon, L) - cobb_douglas(K, L)) / epsilon

    return mpl, mpk
```

```{code-cell} python
tup = marginal_products(1.0, 0.5,  1e-4)
print(tup)
```

Instead of using the tuple, these can be directly unpacked to variables.

```{code-cell} python
mpl, mpk = marginal_products(1.0, 0.5,  1e-4)
print(f"mpl = {mpl}, mpk = {mpk}")
```

We can use this to calculate the marginal products for different `K`, fixing `L`
using a comprehension.

```{code-cell} python
Ks = [1.0, 2.0, 3.0]
[marginal_products(K, 0.5, 1e-4) for K in Ks] # create a tuple for each K
```

### Documentation

In a previous exercise, we asked you to find help for the `cobb_douglas` and
`returns_to_scale` functions using `?`.

It didn't provide any useful information.

To provide this type of help information, we need to
add what Python programmers call a "docstring" to our functions.

This is done by putting a string (not assigned to any variable name) as
the first line of the *body* of the function (after the line with
`def`).

Below is a new version of the template we used to define functions.

```{code-block} python
def function_name(inputs):
    """
    Docstring
    """
    # step 1
    # step 2
    # ...
    return outputs
```

Let's re-define our `cobb_douglas` function to include a docstring.

```{code-cell} python
def cobb_douglas(K, L):
    """
    Computes the production F(K, L) for a Cobb-Douglas production function

    Takes the form F(K, L) = z K^{\alpha} L^{1 - \alpha}

    We restrict z = 1 and alpha = 0.33
    """
    return 1.0 * K**(0.33) * L**(1.0 - 0.33)
```

Now when we have Jupyter evaluate `cobb_douglas?`, our message is
displayed (or use the Contextual Help window with Jupyterlab and `Ctrl-I` or `Cmd-I`).

```{code-cell} python
cobb_douglas?
```

We recommend that you always include at least a very simple docstring for
nontrivial functions.

This is in the same spirit as adding comments to your code — it makes it easier
for future readers/users (including yourself) to understand what the code does.

````{admonition} Exercise
:name: dir2-4-3

See exercise 3 in the {ref}`exercise list <ex2-4>`.
````

### Default and Keyword Arguments

Functions can have optional arguments.

To accomplish this, we must these arguments a *default value* by saying
`name=default_value` instead of just `name` as we list the arguments.

To demonstrate this functionality, let's now make $z$ and $\alpha$
arguments to our cobb_douglas function!

```{code-cell} python
def cobb_douglas(K, L, alpha=0.33, z=1):
    """
    Computes the production F(K, L) for a Cobb-Douglas production function

    Takes the form F(K, L) = z K^{\alpha} L^{1 - \alpha}
    """
    return z * K**(alpha) * L**(1.0 - alpha)
```

We can now call this function by passing in just K and L. Notice that it will
produce same result as earlier because `alpha` and `z` are the same as earlier.

```{code-cell} python
cobb_douglas(1.0, 0.5)
```

However, we can also set the other arguments of the function by passing
more than just K/L.

```{code-cell} python
cobb_douglas(1.0, 0.5, 0.35, 1.6)
```

In the example above, we used `alpha = 0.35`, `z = 1.6`.

We can also refer to function arguments by their name, instead of only their
position (order).

To do this, we would write `func_name(arg=value)` for as many of the arguments
as we want.

Here's how to do that with our `cobb_douglas` example.

```{code-cell} python
cobb_douglas(1.0, 0.5, z = 1.5)
```

````{admonition} Exercise
:name: dir2-4-4

See exercise 4 in the {ref}`exercise list <ex2-4>`.
````

In terms of variable scope, the `z` name within the function is
different from any other `z` in the outer scope.

To be clear,

```{code-cell} python
x = 5
def f(x):
    return x
f(x) # "coincidence" that it has the same name
```

This is also true with named function arguments, above.

```{code-cell} python
z = 1.5
cobb_douglas(1.0, 0.5, z = z) # no problem!
```

In that example, the `z` on the left hand side of `z = z` refers
to the local variable name in the function whereas the `z` on the
right hand side refers to the `z` in the outer scope.


### Creating Custom Types

IN PROGRESS!

- we aer used to doing `x = dict("a": 1, "b": 2)` and then `x["a"]` to access

Used both internal and etemrianl

We can create new one

```{code-cell} python
class A
  def __init__(self, x, y):
    self.x = x
    self.y = y
```

Explain the self and the special  `__init__`

Exlain the difference of an instance/object vs. the class

```{code-cell} python
a = A(1, 2)
b = A(3, 4)
# Notice that these are different objects
a == b
```

Tell people how to see the `type`
Point at the debugger to see the `a.x` etc. fields

Show a method

```{code-cell} python
class B
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def add(self):
    return self.x + self.y
```

```{code-cell} python
a = B(1, 2)
print a.add()
```




### Aside: Methods

As we learned earlier, all variables in Python have a type associated
with them.

Different types of variables have different functions or operations
defined for them.

For example, I can divide one number by another or make a string uppercase.

It wouldn't make sense to divide one string by another or make a number
uppercase.

When certain functionality is closely tied to the type of an object, it
is often implemented as a special kind of function known as a **method**.

For now, you only need to know two things about methods:

1. We call them by doing `variable.method_name(other_arguments)`
   instead of `function_name(variable, other_arguments)`.
1. A method is a function, even though we call it using a different
   notation.

When we introduced the core data types, we saw many methods defined on
these types.

Let's revisit them for the `str`, or string type.

Notice that we call each of these functions using the `dot` syntax
described above.

```{code-cell} python
s = "This is my handy string!"
```

```{code-cell} python
s.upper()
```

```{code-cell} python
s.title()
```

## More on Scope (Optional)

Keep in mind that with mathematical functions, the arguments are just dummy names
that can be interchanged.

That is, the following are identical.

$$
\begin{eqnarray}
    f(K, L) &= z\, K^{\alpha} L^{1-\alpha}\\
    f(K_2, L_2) &= z\, K_2^{\alpha} L_2^{1-\alpha}
\end{eqnarray}
$$

The same concept applies to Python functions, where the arguments are just
placeholder names, and our `cobb_douglas` function is identical to

```{code-cell} python
def cobb_douglas2(K2, L2): # changed dummy variable names

    # Create alpha and z
    z = 1
    alpha = 0.33

    return z * K2**alpha * L2**(1 - alpha)

cobb_douglas2(1.0, 0.5)
```

This is an appealing feature of functions for avoiding coding errors: names of variables
within the function are localized and won't clash with those on the outside (with
more examples in {ref}`scope <scope>`).

Importantly, when Python looks for variables
matching a particular name, it begins in the most local scope.

That is, note that having an `alpha` in the outer scope does not impact the local one.

```{code-cell} python
def cobb_douglas3(K, L, alpha): # added new argument

    # Create alpha and z
    z = 1

    return z * K**alpha * L**(1 - alpha) # sees local argument alpha

print(cobb_douglas3(1.0, 0.5, 0.2))
print("Setting alpha, does the result change?")
alpha = 0.5 # in the outer scope
print(cobb_douglas3(1.0, 0.5, 0.2))
```

A crucial element of the above function is that the `alpha` variable
was available in the local scope of the function.

Consider the alternative where it is not. We have removed the `alpha`
function parameter as well as the local definition of `alpha`.

```{code-cell} python
def cobb_douglas4(K, L): # added new argument

    # Create alpha and z
    z = 1

    # there are no local alpha in scope!
    return z * K**alpha * L**(1 - alpha)

alpha = 0.2 # in the outer scope
print(f"alpha = {alpha} gives {cobb_douglas4(1.0, 0.5)}")
alpha = 0.3
print(f"alpha = {alpha} gives {cobb_douglas4(1.0, 0.5)}")
```

The intuition of scoping does not apply only for the "global" vs. "function"
naming of variables, but also for nesting.

For example, we can define a version of `cobb_douglas` which
is also missing a `z` in its inner-most scope, then put the function
inside of another function.

```{code-cell} python
z = 1
def output_given_alpha(alpha):
    # Scoping logic:
    # 1. local function name doesn't clash with global one
    # 2. alpha comes from the function parameter
    # 3. z comes from the outer global scope
    def cobb_douglas(K, L):
        return z * K**alpha * L**(1 - alpha)

    # using this function
    return cobb_douglas(1.0, 0.5)

alpha = 100 # ignored
alphas = [0.2, 0.3, 0.5]
# comprehension variables also have local scope
# and don't clash with the alpha = 100
[output_given_alpha(alpha) for alpha in alphas]
```
(ex2-4)=
## Exercises

### Exercise 1

What happens if we try different inputs in our Cobb-Douglas production
function?

```{code-cell} python
# Compute returns to scale with different values of `K` and `L` and `gamma`
```

({ref}`back to text <dir2-4-1>`)

### Exercise 2

Define a function named `var` that takes a list (call it `x`) and
computes the variance. This function should use the mean function that we
defined earlier.

```{hint}

$\text{variance} = \frac{1}{N-1} \sum_i (x_i - \text{mean}(x))^2$

```

```{code-cell} python
# Your code here.
```

({ref}`back to text <dir2-4-2>`)

### Exercise 3

Redefine the `returns_to_scale` function and add a docstring.

Confirm that it works by running the cell containing `returns_to_scale?` below.

*Note*: You do not need to change the actual code in the function — just
copy/paste and add a docstring in the correct line.

```{code-cell} python
# re-define the `returns_to_scale` function here
```

```{code-cell} python
:tags: ["remove-output"]
# test it here

returns_to_scale?
```

({ref}`back to text <dir2-4-3>`)

### Exercise 4

Experiment with the `sep` and `end` arguments to the `print` function.

These can *only* be set by name.

```{code-cell} python
# Your code here.
```

({ref}`back to text <dir2-4-4>`)
