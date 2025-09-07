# NLPDL - Fall 2025 Homework

Welcome to the official repository for the NLP course for Fall 2025. This repository contains all the homework assignments for the course.

## Repository Structure
This repository is organized into directories, with each directory corresponding to a specific homework assignment.

```bash
NLPDL-2025Fall/
├── .gitignore
├── .python-version         # Python version specification
├── Makefile
├── README.md
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock               # Locked dependency versions
├── hw0_hello_world/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── README.md
│   └── test_pipeline.py
└── hw1_.../
    └── ...
```

- `hw0_hello_world/`: An introductory assignment to familiarize you with the homework submission and testing process.

- `hw1_...`/: The first main assignment (to be added).

- `...`: Subsequent assignments.

Each homework directory is a self-contained Python package and includes:

- A `README.md` with specific instructions for that assignment.

- One or more Python files (e.g., pipeline.py) where you will implement your code.

- A testing file (e.g., `test_pipeline.py`) containing pytest tests to help you verify your implementation.

## Getting Started (for Students)

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for Python package management. uv is a fast Python package and project manager that provides better dependency resolution and faster installations compared to pip.

**Install uv:**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository_url>
cd <repo_name>
```

2. **Set up your Python environment:**
The project is configured to use Python 3.12. uv will automatically create a virtual environment and install the correct Python version if needed.

```bash
# Install dependencies and create virtual environment
uv sync
```

This command will:
- Create a virtual environment in `.venv/`
- Install Python 3.12 if not available
- Install all project dependencies

3. **Complete the assignment:**
Navigate to the specific homework directory (e.g., `cd hw0_hello_world`) and follow the instructions in its `README.md` to complete your implementation in the provided Python files.

4. **Run tests:**
Use uv to run tests in the managed environment:

```bash
# Run tests for a specific homework (recommended)
uv run pytest hw0_hello_world/

# Or run all tests
uv run pytest
```

You can also use the make commands:
```bash
make test-hw0
```

**Note:** All commands should be run with `uv run` prefix to ensure they execute in the correct virtual environment with the right dependencies.

### Additional uv Commands

```bash
# Add a new dependency
uv add <package_name>

# Remove a dependency  
uv remove <package_name>

# Run any Python command in the environment
uv run python <script.py>

# Activate the virtual environment manually (optional)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

You should see all tests passing if your implementation is correct.

## Guidelines for TAs

This repository is designed to be a template for creating new homework assignments. To contribute a new assignment (e.g., `hw1_new_task`):

1. Create a new directory:

```bash
mkdir hw1_new_task
```

2. Follow the established structure:

- Add a `README.md` with clear instructions.
- Provide a skeleton Python file (e.g., `pipeline.py`) with function stubs for students to complete. Use comments like # TODO: Implement this function.
- Create a `test_pipeline.py` file with comprehensive pytest tests. Include both public tests (visible to students) and potentially hidden tests for grading.
- Add an empty `__init__.py` file to make the directory a package.

3. Update the root Makefile:
Add a new command to run tests for your new assignment. For example:
```bash
test-hw1:
    @echo "Running tests for Homework 1"
    @uv run pytest hw1_new_task/
```
Also, add your new command to the `test-all` rule.

4. Update dependencies if needed:
If your assignment requires additional Python packages, add them using:
```bash
uv add <package_name>
```
This will automatically update both `pyproject.toml` and `uv.lock` files.

By following this template, we can ensure consistency and a smooth experience for everyone involved in the course.

## Homework Submission

**Important:** Students should NOT submit the entire repository. Please refer to the [SUBMISSION_GUIDELINES.md](SUBMISSION_GUIDELINES.md) file for detailed instructions on:

- What files to include in your submission
- How to create a proper submission ZIP file
- Naming conventions for submissions
- Testing your code before submission
- Troubleshooting common issues

**Quick Summary for Students:**
1. Test your implementation: `make test-hw0`
2. Create a ZIP file containing ONLY the homework directory (e.g., `hw0_hello_world/`)
3. Name your submission: `hwX_submission_LASTNAME_FIRSTNAME_STUDENTID.zip`
4. Submit the ZIP file through the course submission system
