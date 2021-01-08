# lecture-datascience.myst

[MIGRATE] Source repository for https://datascience.quantecon.org

This is currently **not** the live published source

Website: https://quantecon.github.io/lecture-datascience.myst/.

## Development 

1. Install [`conda`](https://www.anaconda.com/products/individual)

2. Download this repo, and create a conda environment for it: 

```
conda env create -f environment.yml
```

:warning: **Note**: Make sure you activate this environment whenever working on the lectures, by running `conda activate lecture-datascience`

3. Try building the lectures

```
jupyter-book build src
```

This will take a while. But it will populate your cache, so future iteration is faster. 

4. To clean up (i.e., delete the build.)

```
jupyter-book clean src
```