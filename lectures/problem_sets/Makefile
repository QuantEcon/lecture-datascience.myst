mdfiles := $(wildcard problem_set*.md)
targets := $(mdfiles:.md=)

all:
	echo $(targets)

$(targets): % : %/student/%.ipynb

%.ipynb : %.md
	jupytext $< --to notebook --execute

%/student/%.ipynb : %.ipynb
	otter assign $< % --v1
