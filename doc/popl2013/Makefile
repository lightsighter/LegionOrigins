DOC=legion-types

# dependencies via include files
INCLUDED_TEX = conclusion.tex outline.tex langdef.tex intro.tex
INCLUDED_FIGS = 

$(DOC).pdf: $(DOC).tex bibliography.bib $(INCLUDED_TEX) $(INCLUDED_FIGS)
	pdflatex -halt-on-error $(DOC).tex
	bibtex $(DOC)
	pdflatex -halt-on-error $(DOC).tex
	pdflatex -halt-on-error $(DOC).tex

clean:
	rm -f *.bbl *.aux *.log *.blg *.lot *.lof *.dvi $(DOC).pdf
