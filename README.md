# lecture-datascience.myst

[MIGRATE] Source repository for https://datascience.quantecon.org

This is currently **not** the live published source

Website: https://quantecon.github.io/lecture-datascience.myst/.

## Development 

### Setup

1. Install [`conda`](https://www.anaconda.com/products/individual)

2. Download this repo, and create a conda environment for it: 

```
conda env create -f environment.yml
```

This will install all packages required to edit and build the lectures.

**Note**: Make sure you activate this environment whenever working on the lectures, by running `conda activate lecture-datascience`

3. Try building the lectures

```
jupyter-book build lectures
```

This will take a while. But it will populate your cache, so future iteration is faster. 

4. To clean up (i.e., delete the build.)

```
jupyter-book clean lectures
```

### Releasing updates to GH-PAGES

To make a release you need to setup a tagged release using `publish-` tag. 

Detailed instructions are avaiable in the [quantecon manual](https://manual.quantecon.org/publish/publishing.html#build-and-publish-automatically-via-github)
