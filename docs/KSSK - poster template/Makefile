.PHONY: main clean FORCE

main: poster.pdf

poster.pdf: FORCE
	latexmk -pdf -interaction=nonstopmode poster.tex

clean:
	latexmk -pdf -C
