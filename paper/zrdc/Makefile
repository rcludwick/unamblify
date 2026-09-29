# Build the paper. Tectonic is preferred (it fetches the packages it needs);
# pdflatex works too, but needs newtx, pgfplots and booktabs installed.
#
#   make            -> unamblify-zrdc.pdf
#   make ENGINE=pdflatex
#   make clean

DOC    ?= unamblify-zrdc
ENGINE ?= tectonic
FIGS   := spectrogram_comparison.pdf PXL_20260914_234231998.MACRO_FOCUS.jpg

$(DOC).pdf: $(DOC).tex bibliography.bib $(FIGS)
ifeq ($(ENGINE),tectonic)
	tectonic -X compile --keep-logs $(DOC).tex
else
	$(ENGINE) -interaction=nonstopmode $(DOC).tex
	$(ENGINE) -interaction=nonstopmode $(DOC).tex
endif

.PHONY: clean pages watch
clean:
	rm -f $(DOC).pdf $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc

pages: $(DOC).pdf
	@mdls -name kMDItemNumberOfPages -raw $(DOC).pdf 2>/dev/null || \
	 pdfinfo $(DOC).pdf | grep Pages

watch:
	@echo "Watching for changes to $(DOC).tex and data files... (Ctrl+C to stop)"
	@PATH="/opt/homebrew/bin:$$PATH" fswatch -o $(DOC).tex bibliography.bib $(FIGS) | while read -r _; do \
		echo "\n==> Change detected at $$(date +%T). Recompiling $(DOC).pdf..."; \
		PATH="/opt/homebrew/bin:$$PATH" $(MAKE) $(DOC).pdf && echo "==> Rebuilt $(DOC).pdf successfully."; \
	done
