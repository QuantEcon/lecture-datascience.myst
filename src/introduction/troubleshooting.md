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

(troubleshooting)=
# Troubleshooting


This troubleshooting page is to help ensure your environment is setup correctly
to run this lecture. You can follow {doc}`cloud setup instructions <../introduction/cloud_setup>` or {doc}`local setup instructions <../introduction/local_install>` to set up a standard environment.

## Resetting Lectures

Here are instructions to restore some or all lectures to their original states.

### Hubs

To refresh a **single** notebook:

1. Open the lecture folder in the JupyterHub where it's installed (e.g., the [PIMS Syzygy](https://quantecon.syzygy.ca) server.)
1. Navigate to the lecture in question (i.e., by double-clicking on the folder for that section), and delete it by right-clicking.
1. Go back to the {doc}`lectures website <../index>`, navigate to the page for that lecture.
1. Make sure the settings wheel next to the launch bar is pointing at the right hub, and click the link again. This will repopulate the file freshly from the server.

Otherwise, follow these steps:

1. Open the lecture folder in the JupyterHub where it's installed (e.g., the [PIMS Syzygy](https://pims.syzygy.ca) server.)
1. Close all open lectures by clicking the `x` in their tabs.
1. Open the `utilities.ipynb` notebook (which is in the top level of the repository, outside folders like `introduction`)
1. To **completely reset** your lectures to the latest from the server, run the second cell. This will **also delete** any additional files you have added
1. To **update** files you haven't changed, run the first cell. This will **also pull** any new files that we have added to the GitHub repository (e.g., new lectures.)

### Local Machines

The workflow is a bit different on a local machine. We are assuming that you have followed the {doc}`local setup instructions <../introduction/local_install>`, and have installed GitHub desktop.

1. Open GitHub desktop, and navigate to the repository (you can click "find" under "edit" in the top menu bar, and then type `quantecon-notebooks-datascience`, if you are having trouble.)
1. To **reset everything**, click "discard all changes" under "branch" in the top menu bar.
1. To **reset a specific notebook**, right-click the specific file in the changes side tab, and then click "discard changes."
1. To **pull the latest from the server**, first make sure you don't have any conflicting changes (i.e., do step (2) above), and then click "pull" under "repository" in the top menu bar.

## Reporting an Issue

One way to give feedback is to raise an issue through our [issue tracker](https://github.com/QuantEcon/lecture-datascience.myst/issues).

Please be as specific as possible. Tell us where the problem is and as much
detail about your local set up as you can provide.

Another feedback option is to use our [discourse forum](https://discourse.quantecon.org/).

Finally, you can provide direct feedback to [contact@quantecon.org](mailto:contact@quantecon.org)

