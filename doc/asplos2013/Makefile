DOC=legion-llr

# dependencies via include files
INCLUDED_TEX = conclusion.tex outline.tex impl.tex interface.tex intro.tex \
	micro.tex related.tex results.tex
INCLUDED_FIGS = 

$(DOC).pdf: $(DOC).tex bibliography.bib $(INCLUDED_TEX) $(INCLUDED_FIGS)
	pdflatex -t letter -halt-on-error $(DOC).tex
	bibtex $(DOC)
	pdflatex -t letter -halt-on-error $(DOC).tex
	pdflatex -t letter -halt-on-error $(DOC).tex

clean:
	rm -f *.bbl *.aux *.log *.blg *.lot *.lof *.dvi $(DOC).pdf
