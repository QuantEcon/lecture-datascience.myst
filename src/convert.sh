FILES=$(find . -name '*.md')
for FILE in $FILES
do
    echo $FILE
    jupytext --to ipynb $FILE
done