# NLPDL - Fall 2025 Homework

Welcome to the official repository for the NLP course for Fall 2025. This repository contains all the homework assignments for the course.

## Repository Structure
This repository is organized into directories, with each directory corresponding to a specific homework assignment.

```bash
NLPDL-2025Fall/
├── .gitignore
├── README.md
├── hw0_hello_world/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── README.md
│   ├── test_pipeline.py
│   └── pyproject.toml      # Assignment-specific environment (uv)
└── hw1_bpe_and_lm/
    ├── assignment1.md
    ├── pyproject.toml      # Assignment-specific environment (uv)
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

2. **Set up your Python environment (per assignment):**
Each homework directory is its own uv project. Work inside the assignment directory so uv can manage a dedicated `.venv/` for that assignment.

```bash
# Example: Homework 0
cd hw0_hello_world
uv sync   # creates .venv/ in hw0_hello_world and installs deps

# Example: Homework 1
cd ../hw1_bpe_and_lm
uv sync
```

This will:
- Create a virtual environment in the assignment directory’s `.venv/`
- Install a compatible Python version (e.g., 3.12) if needed
- Install that assignment’s dependencies

3. **Complete the assignment:**
Navigate to the specific homework directory (e.g., `cd hw0_hello_world`) and follow the instructions in its `README.md` to complete your implementation in the provided Python files.

4. **Run tests:**
Run tests from within each assignment directory (recommended), or use uv’s directory flag.

```bash
# From inside an assignment directory
uv run pytest

# Or from the repo root using the assignment directory
uv run --directory hw0_hello_world pytest
uv run --directory hw1_bpe_and_lm pytest
```

**Note:** Use `uv run` within the assignment’s directory (or `--directory`) to ensure the correct per-assignment environment is used.

### Additional uv Commands

```bash
# Add a new dependency
uv add <package_name>

# Remove a dependency  
uv remove <package_name>

# Run any Python command in the assignment environment
uv run python <script.py>

# Activate the virtual environment manually (optional, from assignment dir)
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

You should see all tests passing if your implementation is correct.

## Homework Submission

**Important:** Do NOT submit the entire repository. Submit only the specific homework directory. See [SUBMISSION_GUIDELINES.md](SUBMISSION_GUIDELINES.md) for details.

### Preferred: per-assignment submission scripts

From inside each assignment directory, run:

```bash
# hw0
cd hw0_hello_world
./make_submission.sh <LASTNAME> <FIRSTNAME> <STUDENTID>

# hw1
cd ../hw1_bpe_and_lm
./make_submission.sh <LASTNAME> <FIRSTNAME> <STUDENTID>
```

Each script:
- Runs tests via the assignment’s `uv` environment
- Creates `hwX_submission_LASTNAME_FIRSTNAME_STUDENTID.zip`
- Prints contents for verification

Alternative unified helper (from repo root):

```bash
./submit_hw.sh <hw_directory> <LASTNAME> <FIRSTNAME> <STUDENTID>
```

Refer to the guidelines for exclusions and large files.
