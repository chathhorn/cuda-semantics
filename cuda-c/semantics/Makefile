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

#COMPILE_C = perl -d:DProf $(K_MAUDE) cu.k -u -language C
COMPILE_C = $(K_MAUDE) cu.k -u -language CU -anywhere "anywhere" -transition "interpRule"
# -lib maudeLib/common-c-builtins.maude
# -transition "computational"
COMPILE_C_ND = $(K_MAUDE) cu.k -u -language CU -anywhere "anywhere" -transition "computational" -superheat "ndheat" -supercool "ndlocal"
#-lib maudeLib/common-c-builtins.maude

# chathhorn: adding cuda-syntax.k, cuda-typing.k, and cuda-semantics.k.
COMMON_C_FILES = c-syntax.k cuda-syntax.k cuda-extern.k cuda-typing.k \
common-c-syntax.k common-c-helpers.k common-c-configuration.k common-c-expressions.k common-c-semantics.k common-c-statements.k common-c-declarations.k common-c-typing.k
DYNAMIC_C_FILES = cu.k cuda-semantics.k dynamic-c-semantics.k dynamic-c-configuration.k dynamic-c-expressions.k dynamic-c-errors.k dynamic-c-statements.k dynamic-c-conversions.k dynamic-c-typing.k dynamic-c-declarations.k dynamic-c-memory.k dynamic-c-standard-lib.k

LATEX_CLEAN_FILES = temp-c-*.tex temp-c-*.log temp.pdflatex.out *.pdf temp.kpdf.out *.tex *.log *.aux
K_CLEAN_FILES = *.maude .k kcompile_* \
cu-compiled.maude.tmp.bak cu-compiled.maude.0 cu-compiled.maude.1 cu-compiled.maude.1.bak cu-compiled-nd.maude.0 cu-compiled-nd.maude.1 cu-compiled-nd.maude.1.bak cu-compiled-nd.maude.bak static-c.k.bak cu-compiled.maude \
duplicateLabels.txt 

.PHONY: all clean force semantics pdf common-pdf-modules common-c-%-fake

.SECONDARY:

all: semantics

pdf: common-pdf-modules
#c-expressions.pdf


check-vars:
ifeq ($(K_BASE),)
	@echo "ERROR: Please set K_BASE to the full path of your K installation."
	@echo "Make sure you do NOT include a trailing slash\\"
	@echo "One way to do this is to type 'export K_BASE=/path/to/k/framework', and then rerun 'make'"
	@exit 1
endif

# k-includes.k: check-vars Makefile
# @echo > k-includes.k
#@echo "load $(K_PRELUDE)" >> k-includes.k
#@echo "load $(K_SOCKETS)" >> k-includes.k
temp-c-%.tex: PDF_MODULES=`grep -o '^module.* is' $*.k | grep -o ' .* '`
temp-c-%.tex: check-vars $(K_MAUDE) cu.k %.k
#@rm -f temp-c-$*.tex
	@echo "Creating latex..."
	$(K_MAUDE) cu.k -l C -package-options "style=bubble" -latex $(PDF_MODULES) -output temp-c-$* 1> temp.kpdf.out
	
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

all.tex: PDF_FILES=\
common-c-syntax.k \
common-c-configuration.k \
dynamic-c-configuration.k \
common-c-helpers.k \
common-c-statements.k \
dynamic-c-statements.k \
common-c-expressions.k \
dynamic-c-expressions.k \
common-c-declarations.k \
dynamic-c-declarations.k \
common-c-typing.k \
dynamic-c-typing.k \
dynamic-c-conversions.k \
dynamic-c-memory.k \
dynamic-c-standard-lib.k \
common-c-semantics.k \
dynamic-c-semantics.k \
cu.k
all.tex: PDF_MODULES=`grep -o '^kmod.* is' $(PDF_FILES) | grep -o ' .* '`
all.tex:
	@rm -f temp-c-$*.tex
	@echo "Creating latex..."
	$(K_MAUDE) cu.k -l C -style mm -latex $(PDF_MODULES) -output temp-c-$* 1> temp.kpdf.out
#@sed -i 's|}mybracket|}\\mybracket|' temp-c-$*.tex
#@sed -i 's|\\begin{document}|\\input{semantics.sty}\n\\begin{document}|' temp-c-$*.tex
	@mv temp-c-$*.tex $@
	
	

	
common-pdf-modules: \
cu.pdf \
common-c-configuration.pdf \
common-c-declarations.pdf \
common-c-expressions.pdf \
common-c-helpers.pdf \
common-c-semantics.pdf \
common-c-sized-operations.pdf \
common-c-statements.pdf \
common-c-syntax.pdf \
common-c-typing.pdf \
c-syntax.pdf \
dynamic-c-configuration.pdf \
dynamic-c-conversions.pdf \
dynamic-c-declarations.pdf \
dynamic-c-errors.pdf \
dynamic-c-expressions.pdf \
dynamic-c-memory.pdf \
dynamic-c-semantics.pdf \
dynamic-c-standard-lib.pdf \
dynamic-c-statements.pdf \
dynamic-c-typing.pdf

# c-syntax.pdf
#


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
#@$(INSERT_LINE) 2 cu-compiled.maude.1 "load $(K_PL_MODEL_CHECKER)"
#@$(INSERT_LINE) 2 cu-compiled.maude.1 "load $(K_PL_META_LEVEL)"
#@$(INSERT_LINE) $(FIRST_LINE) cu-compiled.maude.1 "load maudeLib/TCPSupport" 
	@cp cu-compiled.maude.1 cu-compiled.maude.tmp
	@echo "Done".

cu-compiled.maude: cu-compiled.maude.tmp
	@grep -o '^ \(eq\|ceq\|rl\|crl\) .* label [^ ][^ ]* ' cu-compiled.maude.tmp | grep -o ' label [^ ][^ ]*' | grep -o '[^ ]*$$' | sort | uniq -d > duplicateLabels.txt
	@if [ -s duplicateLabels.txt ]; then echo "Error: there are duplicate labels in the semantics.  This could cause profiling to give inaccurate results."; cat duplicateLabels.txt; exit 1; fi
	@mv cu-compiled.maude.tmp cu-compiled.maude

cu-compiled-nd.maude.0: check-vars $(K_FILES) $(COMMON_C_FILES) $(DYNAMIC_C_FILES) $(INSERT_LINE)
	@echo "Compiling the CUDA definition with nondeterminism..."
	@$(COMPILE_C_ND)
	@mv cu-compiled.maude cu-compiled-nd.maude.0
	
cu-compiled-nd.maude: FIRST_LINE=`perl $(FIRST_LINE_SCRIPT) cu-compiled-nd.maude.0`
cu-compiled-nd.maude: check-vars cu-compiled-nd.maude.0 $(K_FILES) $(COMMON_C_FILES) $(DYNAMIC_C_FILES) $(INSERT_LINE) Makefile
	@cp cu-compiled-nd.maude.0 cu-compiled-nd.maude.1
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd.maude.1 "load maudeLib/common-c-builtins"
	@$(APPEND_LINE) cu-compiled-nd.maude.1 "load maudeLib/modelcheck"
	@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd.maude.1 "load $(K_PL_META_LEVEL)"
#@$(APPEND_LINE) cu-compiled-nd.maude.1 "load maudeLib/modelcheck"
#@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd.maude.1 "load $(K_PL_META_LEVEL)"
#@$(INSERT_LINE) 2 cu-compiled-nd.maude "load $(K_PL_MODEL_CHECKER)"
#@$(INSERT_LINE) 3 cu-compiled-nd.maude "load $(K_SOCKETS)"
#@$(INSERT_LINE) $(FIRST_LINE) cu-compiled-nd.maude.1 "load maudeLib/TCPSupport"
	@cp cu-compiled-nd.maude.1 cu-compiled-nd.maude
	@echo "Done".

cu-total.maude: check-vars cu-compiled.maude $(K_BASE_FILES) $(K_TCP).maude Makefile
	@echo "cu-compiled.maude" | perl $(SCRIPTS_DIR)/slurpFrontEnd.pl > $@

cu-total-nd.maude: check-vars cu-compiled-nd.maude $(K_BASE_FILES) $(K_TCP).maude Makefile
	@echo "cu-compiled-nd.maude" | perl $(SCRIPTS_DIR)/slurpFrontEnd.pl > $@

#semantics: cu-total.maude cu-total-nd.maude
semantics: cu-total.maude

fast-nd: cu-total.maude
	@cp cu-total.maude cu-total-nd.maude

fast-normal: cu-total-nd.maude
	@cp cu-total-nd.maude cu-total.maude

fast: cu-total.maude fast-nd

nd: cu-total-nd.maude fast-normal

clean:
	rm -rf $(LATEX_CLEAN_FILES) $(K_CLEAN_FILES) program-*-compiled.maude.tmp

force: ;
