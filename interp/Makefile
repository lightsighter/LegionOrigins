# Makefile for region interpreter

RSIM = ./rsim.py

EPYDOC = epydoc

ALL_TESTS = listrr

tests:
	$(RSIM) $(ALL_TESTS)

docs:
	$(EPYDOC) --html -v -o html -n "Regions Interpreter" --graph=all *.py apps/*.py

