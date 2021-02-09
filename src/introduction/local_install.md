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

# Local Installation

## Installation

Visit [continuum.io](https://www.anaconda.com/download) and download the
Anaconda Python distribution for your operating system (Windows/Mac OS/Linux).

Be sure to download the Python 3.X (where X is some number greater than or equal to 7) version, not
the 2.7 version.

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/install_python.png

```

<br>

Make sure that during the installation [Anaconda](https://www.anaconda.com/distribution/)
is added to your environment/path.

On Mac OS and Linux, this should happen by default.

For Windows users, we recommend installing for "just me" instead of "all users". Windows users will need to **check the upper box** when the page shown below appears (disregard the "not recommended" warning from Anaconda).

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/install_python_path.png

```

## Downloading the QuantEcon Data Science Lectures

To download the QuantEcon Data Science lectures, we use the `Clone` button on the toolbar
as seen in the following image.

<br>

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/clone_button.png

```

<br>

You can download the lectures through either **Github Desktop** or **Terminal**:

**Github Desktop** (Mac/Windows only), recommended for most users.

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/download_lectures_github_desktop.png

```

1. Install [Github Desktop](https://desktop.github.com/).
1. Click the "Open in Github Desktop" option in the clone button menu. It should open a Github
   Desktop popup that looks like this:
   
   You should choose the path (folder) where you would like to download the repository. The default path on
   Windows should be `C:/Users/YOUR_USERNAME/Documents/GitHub`.

**Terminal**

1. Make sure that `git` is installed on your computer. (`git` is not installed on Windows by default. You can download and install it from [here](https://git-scm.com/download/win)).
1. Open a terminal.
1. Set the path to where you would like to download the lectures. The default one is your home directory.
1. Run `git clone https://github.com/QuantEcon/quantecon-notebooks-datascience` which will
   download the repository with notebooks in your working directory. *Pro tip*: If you would rather
   not type this command on your own, you can click "Copy clone command to clipboard" on the clone
   button menu and paste it into the terminal.

## Package Management

In addition to Jupyter, the Anaconda Python distribution comes with two package management tools `conda` and `pip`.

These will help you ensure that you have the right packages (think of these as "add-ons" to Python
that give you additional functionality... We will discuss these more in depth later!) and help you
keep them all up to date.

We will work through an example below to install some new package functionality needed for some
later lectures. Generally, packages can be installed by using `conda install <package name>` or
`pip install <package name>`.

Please install the packages you will need later by following the instructions below for your
computer's operating system.

**Linux/Mac**

- Open a terminal.
- Run the following commands:
  
  ```{code-block} bash
  # Install Python packages
  conda install python-graphviz
  conda install -c conda-forge "nodejs>=10.0" xgboost
  pip install qeds fiona geopandas pyLDAvis gensim folium descartes pyarrow --upgrade
  
  # Activate jlab extensions
  jupyter labextension install @jupyterlab/toc  --no-build
  jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
  jupyter labextension install plotlywidget --no-build
  jupyter labextension install jupyterlab-plotly --no-build
  jupyter lab build
  ```
  Press `y` and enter whenever you see `Proceed [y]/n` from your terminal.
- Close the terminal when the installation finishes.

**Windows**

- Open a command prompt by pressing Windows + R to open the `run` box, type `powershell`, and press
  Enter.
- Run the following commands in order:
  
  ```{code-block} bash
  # Install Python packages
  conda install geopandas python-graphviz
  conda install -c conda-forge nodejs
  pip install qeds pyLDAvis gensim folium xgboost descartes pyarrow graphviz --upgrade
  
  # Activate jlab extensions
  jupyter labextension install @jupyterlab/toc  --no-build
  jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
  jupyter labextension install plotlywidget --no-build
  jupyter labextension install jupyterlab-plotly --no-build
  jupyter lab build
  ```
  Press `y` and enter whenever you see `Proceed [y]/n` from your terminal.
- Close the command window after the installation finishes, log out of Windows, and then log in.

If you are told that you are missing a package at any point in time, we recommend trying to install
the package with `conda` first and, if that doesn't work, installing with `pip`.

You can update a package by running:

- `conda update <package name>` for conda
- `pip install <package name> --upgrade` for pip

**Note:** If you have errors using `graphviz` on Windows, then open a `powershell` terminal and execute the following two lines:

```{code-block} powershell
$pp = (python -c "import sys; print(sys.exec_prefix)")

Set-ItemProperty -path HKCU:\Environment\ -Name Path -Value "$((Get-ItemProperty -path HKCU:\Environment\ -Name Path).Path);$($pp)\Library\bin\graphviz"
```

## Starting Jupyter

Start JupyterLab by following these steps:

1. Open a new terminal (for Windows, you should use the Powershell: press Win + R and type
   `powershell` in the run box, then hit enter).
1. Type `jupyter lab` and press Enter.

If a web browser doesn't open by default, look at the terminal text and find something that looks
like:

```{code-block} md
Copy/paste this URL into your browser when you connect for the first time,
to login with a token:
    http://localhost:8888/?token=9a39d3741a4f0b200c6e4b07d8e5c04a089899cddc72e7f8
```

and copy/paste the line starting with `http://` into your web browser.

```{note}
The terminal you opened must stay open while you are editing the notebooks.
```

### Opening a Jupyter Notebook

Once the web browser is open, you should see the JupyterLab dashboard. You can open a new Jupyter
notebook by clicking Python 3 when you see something like the following image in your browser:

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/jupyter_lab.png

```

<br>

Once the notebook is open, you should something similar to the following image:

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/jupyter_lab_notebook.png

```

<br>

Note that:

- The filenames on the left will be different.
- It *should* list the contents of your personal home directory (folder).

````{admonition} Exercise
:name: dir1-2-1

See exercise 1 in the {ref}`exercise list <ex1-2>`.
````


(ex1-2)=
## Exercises

###### Exercise 1

Open this file in Jupyter by navigating to the QuantEcon Data Science folder that we downloaded
earlier, then click on the `introduction` folder, and select the `getting_started.ipynb` file.

({ref}`back to text <dir1-2-1>`)

