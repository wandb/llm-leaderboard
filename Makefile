.PHONY: setup

# Get the directory where the Makefile is located, to make all paths robust.
MAKEFILE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

setup:
	@echo "Setting up Python environment with vllm==0.9.0 ..."
	@python -c "import re; reqs = [line for line in open('$(MAKEFILE_DIR)requirements.txt') if not re.match(r'^vllm($$|\s|==|>=|<=|<|>|!=|~=)', line.strip())]; print(''.join(reqs) + 'vllm==0.9.0\n')" | pip install -r -
	@echo "Environment setup complete."