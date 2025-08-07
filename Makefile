.PHONY: setup

setup:
	@echo "Step 1: Installing base requirements from requirements.txt..."
	@pip install -r requirements.txt

	@echo "\nStep 2: Upgrading vllm to the target version (0.10.0)..."
	@# The -U flag stands for --upgrade. It will replace the old version of vllm
	@# and automatically handle its dependencies, like upgrading transformers.
	@pip install -U vllm==0.10.0

	@echo "\nEnvironment setup complete."
