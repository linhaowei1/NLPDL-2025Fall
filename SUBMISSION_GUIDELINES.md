# Homework Submission Guidelines

## Important: What NOT to Submit

**DO NOT submit the entire repository.** This includes generated files and directories that are not your direct source code.

-   `.venv/` directory (virtual environment)
-   `.git/` directory (git history)
-   `__pycache__/` directories (Python cache)
-   `.pytest_cache/` directory
-   `uv.lock` file (dependency lock file)
-   Any other generated files or directories

## Handling Large Files (e.g., Model Checkpoints)

**DO NOT submit large files** like trained model checkpoints (`.pth`, `.safetensors`), datasets, or large embedding files. Your submission must contain only the source code needed to run your implementation. The grading environment will provide the necessary large files for testing.

To exclude these files, you can use a `.submission_ignore` file:

1.  **Create the file**: Inside your homework directory (e.g., `hw1_some_assignment/`), create a file named `.submission_ignore`.
2.  **List files/directories to exclude**: Add the names of the files or directories you want to exclude, one per line. You can use wildcards (`*`).

### Example .submission_ignore

If your large files are in a `checkpoints` directory, your `.submission_ignore` file would contain:

```bash
# Ignore the entire checkpoints directory
checkpoints/ 

# You can also ignore specific file types
*.pth
*.safetensors
```

The quick submission script provided below is already configured to use this file and will automatically exclude these items from your final ZIP file.


## What TO Submit

For each homework assignment, create a ZIP file containing **only the specific homework directory** with your implementation.

### Example: Submitting Homework 0

1. **Navigate to your local repository:**

   ```bash
   cd NLPDL-2025Fall
   ```

2. **Ensure your implementation is complete:**

   ```bash
   # Test your implementation first
   make test-hw0
   # or
   uv run pytest hw0_hello_world/
   ```

3. **Create the submission ZIP:**

   ```bash
   # Create a clean copy of just the homework directory
   cp -r hw0_hello_world hw0_hello_world_submission
   
   # Remove any cache directories
   find hw0_hello_world_submission -name "__pycache__" -type d -exec rm -rf {} +
   find hw0_hello_world_submission -name "*.pyc" -delete
   
   # Create the ZIP file (replace with your info)
   zip -r hw0_submission_DOE_JANE_12345678.zip hw0_hello_world_submission/
   
   # Clean up the temporary directory
   rm -rf hw0_hello_world_submission
   ```

4. **Verify your ZIP contents:**

   ```bash
   unzip -l hw0_submission_DOE_JANE_12345678.zip
   ```

   Your ZIP should contain a structure like this:

   ```
   hw0_hello_world_submission/
   ├── __init__.py
   ├── pipeline.py          # ← Your implementation here
   ├── README.md
   └── test_pipeline.py
   ```

## General Submission Format

For any homework assignment `hwX_name`, your submission should be a ZIP file named:

`hwX_submission_LASTNAME_FIRSTNAME_STUDENTID.zip`

### Examples:

-   `hw0_submission_SMITH_JOHN_11223344.zip`
-   `hw1_submission_DOE_JANE_12345678.zip`
-   `hw2_submission_WANG_LI_87654321.zip`


## Submission Checklist

Before submitting, ensure:

-   [ ] **Tests pass locally**: Run `make test-hwX` or `uv run pytest hwX_*/` and verify all tests pass.
-   [ ] **Only homework directory included**: Your ZIP contains only the `hwX_*` directory with your implementation.
-   [ ] **No cache files**: No `__pycache__`, `.pyc`, or other generated files in your ZIP.
-   [ ] **No large files**: Model checkpoints or large data files are excluded (using `.submission_ignore` is recommended).
-   [ ] **Correct naming**: ZIP file follows the naming convention `hwX_submission_LASTNAME_FIRSTNAME_STUDENTID.zip`.
-   [ ] **File integrity**: Unzip and verify the contents are correct and complete.


## Quick Submission Script

You can use this script to automate the submission process. It handles cleaning cache files and excluding files listed in `.submission_ignore`.

```bash
#!/bin/bash
# save as submit_hw.sh and run: bash submit_hw.sh hw0_hello_world LASTNAME FIRSTNAME STUDENTID

HW_DIR=$1
LASTNAME=$2
FIRSTNAME=$3
STUDENTID=$4

if [ $# -ne 4 ]; then
    echo "Usage: bash submit_hw.sh <hw_directory> <LASTNAME> <FIRSTNAME> <STUDENTID>"
    echo "Example: bash submit_hw.sh hw0_hello_world SMITH JOHN 11223344"
    exit 1
fi

# Extract homework number from directory name
HW_NUM=$(echo $HW_DIR | sed 's/hw\([0-9]\+\).*/\1/')

# Test the homework first
echo "Testing homework implementation..."
uv run pytest $HW_DIR/
if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Please fix your implementation before submitting."
    exit 1
fi

# Create submission
SUBMISSION_NAME="${HW_DIR}_submission"
SUBMISSION_ZIP="hw${HW_NUM}_submission_${LASTNAME}_${FIRSTNAME}_${STUDENTID}.zip"
IGNORE_FILE="${HW_DIR}/.submission_ignore"
EXCLUDE_ARGS=()

echo "Creating submission ZIP..."
cp -r $HW_DIR $SUBMISSION_NAME

# Clean cache files
find $SUBMISSION_NAME -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find $SUBMISSION_NAME -name "*.pyc" -delete 2>/dev/null

# Add files from .submission_ignore to exclude list
if [ -f "$IGNORE_FILE" ]; then
    echo "Found .submission_ignore. Excluding specified files/directories."
    while IFS= read -r pattern; do
        # Skip empty lines or comments
        [[ -z "$pattern" || "$pattern" == \#* ]] && continue
        EXCLUDE_ARGS+=("-x" "${SUBMISSION_NAME}/${pattern}")
    done < "$IGNORE_FILE"
fi

# Create the zip file with exclusions
zip -r "$SUBMISSION_ZIP" "$SUBMISSION_NAME/" "${EXCLUDE_ARGS[@]}"

rm -rf $SUBMISSION_NAME

echo "✅ Submission created: $SUBMISSION_ZIP"
echo "📝 Contents:"
unzip -l $SUBMISSION_ZIP
```

### To use this script:

1. Save the code above into a file named `submit_hw.sh`.

2. Make it executable:

   Bash

   ```
   chmod +x submit_hw.sh
   ```

3. Run it from your main repository directory (`NLPDL-2025Fall`):

   Bash

   ```
   ./submit_hw.sh hw0_hello_world SMITH JOHN 11223344
   ```

## Troubleshooting

- **Q: My tests are failing. What should I do?**
  - A: Do not submit until all tests pass. Review the assignment README and your implementation. Ask a TA for help if needed.
- **Q: My code needs a model checkpoint. How do I submit it?**
  - A: **Do not include the model checkpoint in your ZIP file.** Your code should be able to load a model from a specified path (e.g., `checkpoints/model.pth`). The grading script will place the necessary model file there before running your code. Use the `.submission_ignore` file to exclude it.
- **Q: Should I include my own additional test files?**
  - A: Only include additional test files if explicitly allowed in the assignment instructions. Generally, submit only the files that were originally provided.
- **Q: Can I modify the test files?**
  - A: No, do not modify the provided test files. Your implementation should pass the original tests as provided.
- **Q: What if I added extra dependencies?**
  - A: Contact the course staff before submission. Generally, you should only use the dependencies specified in the assignment.