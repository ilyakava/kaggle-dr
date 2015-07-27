test: my_code/test
	python -m unittest discover -s my_code/test -p '*_test.py'

pdf: about.md
	pandoc about.md -o about.pdf

html: about.md
	pandoc about.md -o about.html
