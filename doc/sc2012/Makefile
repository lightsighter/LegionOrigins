DOC=legion

$(DOC).pdf: $(DOC).tex bibliography.bib  circuit.tex code_ex.tex conclusion.tex experiments.tex related.tex mapping.tex soop.tex part_fig.tex
	pdflatex $(DOC).tex
	bibtex $(DOC)
	pdflatex $(DOC).tex
	pdflatex $(DOC).tex

# dependencies via include files
#exec.tex lowlevel.tex circuit.tex code_ex.tex part_fig.tex highlevel.tex related.tex experiments.tex
#legion.pdf : circuit.tex code_ex.tex part_fig.tex
#legion.pdf : type_system.tex
#legion.pdf : highlevel.tex mapping_fig.tex
#legion.pdf : lowlevel.tex
#legion.pdf : experiments.tex

clean:
	rm -f *.bbl *.aux *.log *.blg *.lot *.lof *.dvi legion.pdf
