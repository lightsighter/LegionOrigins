DOC=legion-types

# dependencies via include files
INCLUDED_TEX = conclusion.tex outline.tex langdef.tex
INCLUDED_FIGS = 

$(DOC).pdf: $(DOC).tex bibliography.bib $(INCLUDED_TEX) $(INCLUDED_FIGS)
	pdflatex $(DOC).tex
	bibtex $(DOC)
	pdflatex $(DOC).tex
	pdflatex $(DOC).tex

clean:
	rm -f *.bbl *.aux *.log *.blg *.lot *.lof *.dvi $(DOC).pdf
