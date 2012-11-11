DOC=legion-types

# dependencies via include files
INCLUDED_TEX = conclusion.tex langdef.tex intro.tex evaluation.tex related.tex example.tex alt_example.tex example_full.tex langdef_full.tex
INCLUDED_FIGS = 

$(DOC).pdf: $(DOC).tex bibliography.bib $(INCLUDED_TEX) $(INCLUDED_FIGS)
	pdflatex -halt-on-error $(DOC).tex
	bibtex $(DOC)
	pdflatex -halt-on-error $(DOC).tex
	pdflatex -halt-on-error $(DOC).tex
	pdftk legion-types.pdf cat 1-10 output legion-types-noapp.pdf
	pdftk legion-types.pdf cat 11-end output legion-types-apponly.pdf

clean:
	rm -f *.bbl *.aux *.log *.blg *.lot *.lof *.dvi $(DOC).pdf
