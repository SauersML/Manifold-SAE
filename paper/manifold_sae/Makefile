PAPER=manifold_sae

.PHONY: all clean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib
	@which pdflatex > /dev/null 2>&1 || { echo "pdflatex not found; install MacTeX or TeXLive"; exit 1; }
	pdflatex -interaction=nonstopmode $(PAPER).tex
	-bibtex $(PAPER) || true
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
