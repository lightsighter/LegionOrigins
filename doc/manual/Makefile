################################################################################
# Top-level Dependencies:
################################################################################

all: model 
clean: clean_model

################################################################################
# Documentation Dependencies:
################################################################################

TEX = pdflatex
BIB = bibtex

OBJECTS = parallax_model.tex

OBJECTDIV = parallax_model.dvi

OBJECTPDF = ParallaxModel.pdf

model: $(OBJECTS) 
	$(TEX) $(OBJECTS)
	$(BIB) parallax_model.aux
	$(TEX) $(OBJECTS)
	$(TEX) $(OBJECTS)

clean_model:
	rm -f *dvi *bak *pdf *log *lot *toc *aux *bbl *blg
