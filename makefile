# Makefile for NLPDL Fall 2025 Homework

# Default command when just `make` is run
all: test-all

# ==============================================================================
#  VARIABLES
# ==============================================================================
# Define homework directories here
HW0_DIR = hw0_hello_world
# HW1_DIR = hw1_... (add new homework here)


# ==============================================================================
#  TESTING RULES
# ==============================================================================

# Rule to run tests for Homework 0
test-hw0:
	@echo "---------------------------------------"
	@echo ">> Running tests for Homework 0: Hello World"
	@echo "---------------------------------------"
	@uv run pytest $(HW0_DIR)/

# Add rules for new homework assignments below
# Example:
# test-hw1:
#	@echo "---------------------------------------"
#	@echo ">> Running tests for Homework 1"
#	@echo "---------------------------------------"
#	@uv run pytest $(HW1_DIR)/

# Rule to run all tests for all assignments
test-all: test-hw0 # test-hw1 ...
	@echo "\nAll specified tests completed."


# ==============================================================================
#  SETUP RULES
# ==============================================================================

# Rule to set up the environment using uv
setup:
	@echo "Setting up Python environment with uv..."
	@uv sync
	@echo "Environment setup complete. You can now run tests with 'make test-hw0' or 'uv run pytest'."

# ==============================================================================
#  UTILITY RULES
# ==============================================================================

# A phony rule to prevent conflicts with file names
.PHONY: all test-all test-hw0 setup clean

# Rule to clean up Python cache files
clean:
	@echo "Cleaning up Python cache files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "Cleanup complete."

# Help command to list available commands
help:
	@echo "Available commands:"
	@echo "  make setup        - Set up the Python environment using uv."
	@echo "  make all          - Run all tests for all homework assignments (default)."
	@echo "  make test-all     - Same as 'make all'."
	@echo "  make test-hw0     - Run tests specifically for Homework 0."
	@echo "  make clean        - Remove Python cache files."
	@echo "  make help         - Display this help message."
