SCRIPTS_DIR = ../scripts
INSERT_LINE = $(SCRIPTS_DIR)/insert.sh
APPEND_LINE = $(SCRIPTS_DIR)/append.sh
FIRST_LINE_SCRIPT = $(SCRIPTS_DIR)/getFirstLine.pl

K_MAUDE = $(K_BASE)/core/kompile
K_PRELUDE = $(K_BASE)/core/maude/lib/k-prelude
K_SOCKETS = $(K_BASE)/core/maude/lib/socket
K_TCP = $(K_BASE)/core/maude/lib/TCPSupport
K-PL-BUILTINS = $(K_BASE)/core/maude/lib/pl-builtins
K_PL_MODEL_CHECKER = $(K_BASE)/core/maude/lib/k-model-checker
K_PL_META_LEVEL = $(K_BASE)/core/maude/lib/pl-meta-level
K_LATEX_STYLE = $(K_BASE)/core/latex/k.sty


K_BASE_FILES = $(K_PRELUDE).maude $(K_SOCKETS).maude $(K-PL-BUILTINS).maude $(K_PL_MODEL_CHECKER).maude $(K_PL_META_LEVEL).maude
K_FILES = $(K_BASE_FILES) $(K_MAUDE)

#COMPILE_C = perl -d:DProf $(K_MAUDE) cu.k -u -language CU
COMPILE_C = $(K_MAUDE) cu.k -u -language CU -anywhere "anywhere" -transition "interpRule"
# -lib maudeLib/common-c-builtins.maude
# -transition "computational"
COMPILE_C_ND = $(K_MAUDE) cu.k -u -language CU -anywhere "anywhere" -transition "observable computational" -superheat "ndheat" -supercool "ndlocal"
COMPILE_C_ND_THREAD = $(K_MAUDE) cu.k -u -language CU -anywhere "anywhere" -transition "observable computational"
#-lib maudeLib/common-c-builtins.maude

CUDA_FILES = cuda-headers.k cuda-semantics.k cuda-syntax.k cuda-typing.k cuda-memory.k \
cuda-threads.k cuda-declarations.k cuda-datatypes.k cuda-streams.k cuda-devices.k cuda-math.k \
cuda-racecheck.k cuda-properties.k cuda-errors.k cuda-execution.k

COMMON_C_FILES = cu-syntax.k $(CUDA_FILES) \
common-c-syntax.k common-c-helpers.k common-c-configuration.k common-c-expressions.k common-c-semantics.k common-c-statements.k common-c-declarations.k common-c-typing.k
DYNAMIC_C_FILES = cu.k dynamic-c-semantics.k dynamic-c-configuration.k dynamic-c-expressions.k dynamic-c-errors.k dynamic-c-statements.k dynamic-c-conversions.k dynamic-c-typing.k dynamic-c-declarations.k dynamic-c-memory.k dynamic-c-standard-lib.k

LATEX_CLEAN_FILES = temp-c-*.tex temp-c-*.log temp.pdflatex.out *.pdf temp.kpdf.out *.tex *.log *.aux
K_CLEAN_FILES = *.maude .k kcompile_* \
cu-compiled.maude.tmp.bak cu-compiled.maude.0 cu-compiled.maude.1 cu-compiled.maude.1.bak cu-compiled-nd.maude.0 cu-compiled-nd.maude.1 cu-compiled-nd.maude.1.bak cu-compiled-nd.maude.bak static-cu.k.bak cu-compiled.maude \
duplicateLabels.txt 

.PHONY: all clean force semantics pdf common-pdf-modules common-c-%-fake

.SECONDARY:

all: semantics

check-vars:
ifeq ($(K_BASE),)
	@echo "ERROR: Please set K_BASE to the full path of your K installation."
	@echo "Make sure you do NOT include a trailing slash\\"
	@echo "One way to do this is to type 'export K_BASE=/path/to/k/framework', and then rerun 'make'"
	@exit 1
endif

temp-c-syntax.tex: PDF_FILES = common-c-syntax
temp-c-statements.tex: PDF_FILES = common-c-statements dynamic-c-statements
temp-c-expressions.tex: PDF_FILES = common-c-expressions dynamic-c-expressions
temp-c-typing.tex: PDF_FILES = common-c-typing dynamic-c-typing
temp-c-declarations.tex: PDF_FILES = common-c-declarations dynamic-c-declarations
temp-c-memory.tex: PDF_FILES = dynamic-c-memory
temp-c-library.tex: PDF_FILES = dynamic-c-standard-lib
temp-c-errors.tex: PDF_FILES = dynamic-c-errors
temp-c-misc.tex: PDF_FILES = common-c-semantics common-c-helpers dynamic-c-semantics
temp-c-%.tex: PDF_MODULES = `grep -o '^module.* is' $(addsuffix .k,$(PDF_FILES)) | grep -o ' .* '`
temp-c-%.tex: check-vars $(K_MAUDE) cu.k $(addsuffix .k,$(PDF_FILES))
	@echo "Creating latex..."
	$(K_MAUDE) cu.k -l CU -package-options "style=bubble,magicTight" -latex $(PDF_MODULES) -output temp-c-$* 1> temp.kpdf.out

%.tex: PDF_FILES ?= $*
%.tex: temp-c-%.tex Makefile
	@echo "Modifying latex..."
	@cp $< $@.temp
	@perl -i -pe 's|\\documentclass{article}|\\documentclass[landscape]{article}|' $@.temp
# get rid of page breaks
	@perl -i -pe 's|\\newpage||' $@.temp
	@perl -i -pe 's|}mybracket|}\\mybracket|' $@.temp
	@perl -i -pe 's|\\mathrel{}||g' $@.temp
# include my custom style file
	@perl -i -pe 's|\\begin{document}|\\input{semantics.sty}\n\\begin{document}|' $@.temp
# get rid of the name attributes so they don't show up twice
	@perl -i -pe 's/(\\(?:kdefine|krule){([a-zA-Z0-9-]+)}.*)\\kattribute{\2}[ ]?/\1\3/' $@.temp
# get rid of structural annotations
	@perl -i -pe 's/\\kattribute{structural}[ ]?//' $@.temp
# get rid of klabel annotations
	@perl -i -pe 's/\\kattribute{klabel\([a-zA-Z0-9-]+\)}[ ]?//' $@.temp
# fix double quotes
	@perl -i -pe "s/ ' '/''/g" $@.temp	
	@perl -i -pe "s/{([^}]+)}_{([^}]+)}' '/{\1''}_{\2}/g" $@.temp	
# remove extraneous spaces at end of attributes
	@perl -i -pe 's/(\\kattribute{[a-zA-Z-()]+})[ ]?}/\1}/' $@.temp
	@mv $@.temp $@

%.pdf: %.tex $(K_LATEX_STYLE) semantics.sty
	@echo "Compiling latex..."
	@if ! pdflatex -halt-on-error -interaction=nonstopmode -jobname=temp-c-$* $* 1> temp.pdflatex.out; then cat temp.pdflatex.out; false; fi
#@if ! pdflatex -halt-on-error -interaction=nonstopmode -jobname=temp-c-$* $* 1> temp.pdflatex.out; then cat temp.pdflatex.out; false; fi
	@mv temp-c-$*.pdf $@
	@echo "Done."

configuration.pdf: custom-configuration.tex.orig
	cp custom-configuration.tex.orig custom-configuration.tex
	pdflatex custom-configuration.tex

pdf: syntax.pdf configuration.pdf statements.pdf expressions.pdf typing.pdf declarations.pdf memory.pdf library.pdf errors.pdf misc.pdf


cu-compiled.maude.0: check-vars $(K_FILES) $(COMMON_C_FILES) $(DYNAMIC_C_FILES) $(INSERT_LINE)
	@echo "Compiling the CUDA definition..."
	@$(COMPILE_C)
	@mv cu-compiled.maude cu-compiled.maude.0

cu-compiled.maude.tmp: FIRST_LINE=`perl $(FIRST_LINE_SCRIPT) cu-compiled.maude.0`
cu-compiled.maude.tmp: cu-compiled.maude.0 maudeLib/modelcheck.maude Makefile
	@cp cu-compiled.maude.0 cu-compiled.maude.1
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled.maude.1 "load maudeLib/common-c-builtins"
	@$(APPEND_LINE) cu-compiled.maude.1 "load maudeLib/modelcheck"
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled.maude.1 "load $(K_PL_META_LEVEL)"
	@cp cu-compiled.maude.1 cu-compiled.maude.tmp
	@echo "Done".

cu-compiled.maude: cu-compiled.maude.tmp
# does not get all labels, but will hopefully find duplicates :(
	@sed -n 's/^ \(eq\|ceq\|rl\|crl\) .* label \([^ ][^ ]*\) .*\(superheated\).*$$/\2:\3/p' cu-compiled.maude.tmp | sort | uniq -d > duplicateLabels.txt
	@if [ -s duplicateLabels.txt ]; then echo "Error: there are duplicate labels in the semantics.  This could cause profiling to give inaccurate results."; cat duplicateLabels.txt; exit 1; fi
	@mv cu-compiled.maude.tmp cu-compiled.maude

cu-total.maude: check-vars cu-compiled.maude $(K_BASE_FILES) $(K_TCP).maude Makefile
	@echo "cu-compiled.maude" | perl $(SCRIPTS_DIR)/slurpFrontEnd.pl > $@

cu-compiled-nd.maude.0: check-vars $(K_FILES) $(COMMON_C_FILES) $(DYNAMIC_C_FILES) $(INSERT_LINE)
	@echo "Compiling the C definition with nondeterminism..."
	@$(COMPILE_C_ND)
	@mv cu-compiled.maude cu-compiled-nd.maude.0
	
cu-compiled-nd.maude: FIRST_LINE=`perl $(FIRST_LINE_SCRIPT) cu-compiled-nd.maude.0`
cu-compiled-nd.maude: check-vars cu-compiled-nd.maude.0 $(K_FILES) $(COMMON_C_FILES) $(DYNAMIC_C_FILES) $(INSERT_LINE) Makefile
	@cp cu-compiled-nd.maude.0 cu-compiled-nd.maude.1
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd.maude.1 "load maudeLib/common-c-builtins"
	@$(APPEND_LINE) cu-compiled-nd.maude.1 "load maudeLib/modelcheck"
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd.maude.1 "load $(K_PL_META_LEVEL)"
	@cp cu-compiled-nd.maude.1 cu-compiled-nd.maude
	@echo "Done".
	
cu-total-nd.maude: check-vars cu-compiled-nd.maude $(K_BASE_FILES) $(K_TCP).maude Makefile
	@echo "cu-compiled-nd.maude" | perl $(SCRIPTS_DIR)/slurpFrontEnd.pl > $@

	
cu-compiled-nd-thread.maude.0: check-vars $(K_FILES) $(COMMON_C_FILES) $(DYNAMIC_C_FILES) $(INSERT_LINE)
	@echo "Compiling the C definition with thread-nondeterminism..."
	@$(COMPILE_C_ND_THREAD)
	@mv cu-compiled.maude cu-compiled-nd-thread.maude.0
	
cu-compiled-nd-thread.maude: FIRST_LINE=`perl $(FIRST_LINE_SCRIPT) cu-compiled-nd-thread.maude.0`
cu-compiled-nd-thread.maude: check-vars cu-compiled-nd-thread.maude.0 $(K_FILES) $(COMMON_C_FILES) $(DYNAMIC_C_FILES) $(INSERT_LINE) Makefile
	@cp cu-compiled-nd-thread.maude.0 cu-compiled-nd-thread.maude.1
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd-thread.maude.1 "load maudeLib/common-c-builtins"
	@$(APPEND_LINE) cu-compiled-nd-thread.maude.1 "load maudeLib/modelcheck"
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd-thread.maude.1 "load $(K_PL_META_LEVEL)"
	@cp cu-compiled-nd-thread.maude.1 cu-compiled-nd-thread.maude
	@echo "Done".
	
cu-total-nd-thread.maude: check-vars cu-compiled-nd-thread.maude $(K_BASE_FILES) $(K_TCP).maude Makefile
	@echo "cu-compiled-nd-thread.maude" | perl $(SCRIPTS_DIR)/slurpFrontEnd.pl > $@
	


#semantics: cu-total.maude cu-total-nd.maude cu-total-nd-thread.maude
semantics: cu-total.maude

fast: cu-total.maude
	@cp cu-total.maude cu-total-nd.maude
	@cp cu-total.maude cu-total-nd-thread.maude

nd: cu-total-nd.maude
	@cp cu-total-nd.maude cu-total.maude
	@cp cu-total-nd.maude cu-total-nd-thread.maude

thread: cu-total-nd-thread.maude
	@cp cu-total-nd-thread.maude cu-total.maude
	@cp cu-total-nd-thread.maude cu-total-nd.maude

clean:
	rm -rf $(LATEX_CLEAN_FILES) $(K_CLEAN_FILES) program-*-compiled.maude.tmp

force: ;
