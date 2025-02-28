mdfiles := $(wildcard problem_set*.md)
targets := $(mdfiles:.md=.out)

all:
	echo $(targets)

$(targets): %.out : %/student/%.ipynb

%.ipynb : %.md
	jupytext $< --to notebook --execute

%/student/%.ipynb : %.ipynb
	otter assign $< $* --v1
