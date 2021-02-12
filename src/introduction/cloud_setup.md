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

# Cloud Setup

## Launch Environments

Various cloud-based Jupyter server environments have been configured to work with
[QuantEcon Data Science](https://datascience.quantecon.org).

These environments provide a Jupyter interface which displays in your browser, but the code is hosted
and run on the cloud.

This allows you to interact with this lecture material without requiring you to install Python or
any other required software on your own computer.

The `Launch Notebook` button opens a new tab in your browser where a Jupyter notebook version of the
current lecture page will be opened with the selected cloud service.

The settings icon allows you to change your cloud environment from the default.  To use
click the over the settings icon on the bottom right of the page (see image).

<br>

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/cloud_launch.png

```

<br>

Making a selection will open a new tab in your browser where a Jupyter notebook version of the
current lecture page will be opened with the selected cloud service.

We discuss each of the options below.

### Pacific Institute for Mathematical Sciences: syzygy

This cloud service is provided by the
[Pacific Institute for Mathematical Sciences (PIMS)](https://www.pims.math.ca) at UBC.

A major benefit of this environment is that we have worked with PIMS to install all of the software
that shows up in our lectures.

Additionally, they are serving an unaltered version of Jupyter so it will be easy to find help
online, and it will feel very similar to a local installation if you have one configured.

Finally, it is the only service that has persistent storage.
**Note:**  PIMS is providing the hub "as is", and may be turned off at any time, so backup your notebooks.  Storage used by accounts with more than 30 days of inactivity may be recycled.

This is the service that we typically recommend.

#### Launching on pims.syzygy.ca

As pims.syzygy.ca is the default hub, you will not need to change your configuration.  However,
if you chose a different hub, select the settings icon on the launchbar to select the PIMS hub.

**1.** Click the `Launch Notebook` button.

**2.** After you have done this, you will be brought to a screen that looks like the following

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/syzygy_login.png

```

Once you reach this page, you should click the red home button and sign in with your Google account.

The server will create an account for you associated with that Google account and you will have
access to all files that you have worked on anytime you log in with that Google account.

**3.** After you sign in with your Google account, then you will see something similar to the
following picture

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/syzygy_jupyter.png

```

#### File Management on syzygy

By default, syzygy will open the notebook that you clicked "Launch" from, but it downloads the other
notebooks as well.

You can navigate to these other notebooks in the left bar (the JupyterLab file explorer).

You can also create new notebooks and text files.

We recommend spending some time playing with the environment and working through some of the Help
menu at the top.

If you have made changes to one of the notebooks and decide that you want to reset them (and put
the notebook back to the original state), then follow these steps:

- Delete the notebook and close the browser tab with syzygy instance.
- Click `Launch Notebook` again from the website.

Once you've done this, syzygy will restore the notebook back to what the original lecture material.

As a reminder, while the PIMS syzygy will maintain persistent files between sessions, you should backup
your files as the service is provided "as is", and inactive accounts may have their storage recycled.

### Google Colab

[Google Colab](https://research.google.com/colaboratory/faq.html) is a cloud service hosted by
Google.

With this environment, you can potentially use GPUs and other specialized
computational platforms.

This won't make a difference at first, but having access to a
GPU or TPU could improve performance.

We recommend starting with the other cloud options because the environment provided isn't
quite the same as what you would get on your computer because Google has made their own modifications
to underlying Jupyter software.

#### Launching Colab

To launch course material through Google Colab:

**1.** Choose the settings icon on the launch bar to change the backend hub

You will need to select choose `google colab` from the public options.

**2.** Click on the `Launch Notebook` button.

**3.** You will be asked to sign in with your Google account and you will see something similar to
the following picture.

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/colab_jupyter.png

```

**4.** Once you have launched a Colab notebook, you will need to make sure that any software missing
from Colab gets installed --- this step isn't required for all notebooks.

For lectures where this step is required, we have provided a script that automatically configures missing
software.

To run this script, you will need to uncomment the code at the top of the notebook and execute the
cell --- when we say "uncomment", all we mean is to remove the `#` that precedes the `!` in the
code that follows.

See the code below to see what we mean:

```{literalinclude} ../_static/colab_light.raw
```

**5.** To navigate sections within a Colab notebook, click on the little arrow at the top left corner
of the page,

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/colab_table_of_contents_arrow.png

```

Then, "table of contents" will pop up. You can click on sections or subsections for different parts
of the notebook.

```{figure} https://datascience.quantecon.org/assets/_static/introduction_files/colab_table_of_contents.png

```

#### File Management on Colab

By default, Colab will erase any work that you have done after you have exited a notebook.

If you would like to store your work, you can save it onto your Google Drive by clicking the
`Copy to Drive` button.

You can create a new notebook by clicking `File` on the menubar and selecting
`New Python 3 notebook`.

