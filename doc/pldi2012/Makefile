DOC=legion

$(DOC).pdf: $(DOC).tex bibliography.bib
	pdflatex $(DOC).tex
	bibtex $(DOC)
	pdflatex $(DOC).tex
