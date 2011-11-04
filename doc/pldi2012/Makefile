DOC=legion

$(DOC).pdf: $(DOC).tex bibliography.bib
	pdflatex $(DOC).tex
	bibtex $(DOC)
	pdflatex $(DOC).tex

# dependencies via include files
legion.pdf : circuit.tex code_ex.tex part_fig.tex
legion.pdf : type_system.tex
legion.pdf : highlevel.tex mapping_fig.tex
legion.pdf : lowlevel.tex