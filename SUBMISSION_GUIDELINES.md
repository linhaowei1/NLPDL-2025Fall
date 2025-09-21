# Homework Submission Guidelines

## Important: What NOT to Submit

**DO NOT submit the entire repository.** This includes generated files and directories that are not your direct source code.

-   `.venv/` directories inside each assignment (per-assignment virtual environments)
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

For each assignment, submit a ZIP containing only the specific homework directory with your implementation. Preferred flow:

```bash
# from inside the assignment directory
./make_submission.sh <LASTNAME> <FIRSTNAME> <STUDENTID>

# examples
cd hw0_hello_world && ./make_submission.sh DOE JANE 12345678
cd ../hw1_bpe_and_lm && ./make_submission.sh SMITH JOHN 11223344
```
the scripts run tests and produce `hwX_submission_LASTNAME_FIRSTNAME_STUDENTID.zip`, printing contents for verification.

## General Submission Format

For any homework assignment `hwX_name`, your submission should be a ZIP file named:

`hwX_submission_LASTNAME_FIRSTNAME_STUDENTID.zip`

### Examples:

-   `hw0_submission_SMITH_JOHN_11223344.zip`
-   `hw1_submission_DOE_JANE_12345678.zip`
-   `hw2_submission_WANG_LI_87654321.zip`


## Submission Checklist

Before submitting, ensure:

-   [ ] **Tests pass locally**: Run `./submit_hw.sh ...` or `uv run --directory <hw_dir> pytest` and verify tests pass.
-   [ ] **Only homework directory included**: Your ZIP contains only the `hwX_*` directory with your implementation.
-   [ ] **No cache files**: No `__pycache__`, `.pyc`, or other generated files in your ZIP.
-   [ ] **No large files**: Model checkpoints or large data files are excluded (using `.submission_ignore` is recommended).
-   [ ] **Correct naming**: ZIP file follows the naming convention `hwX_submission_LASTNAME_FIRSTNAME_STUDENTID.zip`.
-   [ ] **File integrity**: Unzip and verify the contents are correct and complete.


## Built-in submission helper

Use the provided `submit_hw.sh` in the repo root. It cleans caches and respects `.submission_ignore` for excluding large files (e.g., checkpoints, datasets). Make sure `zip` is installed on your system.

## Troubleshooting

- **Q: My tests are failing. What should I do?**
  - A: You could submit without all tests passing, but you cannot get the full scores if you do. Review the assignment README and your implementation. Ask a TA for help if needed.
- **Q: My code needs a model checkpoint. How do I submit it?**
  - A: **Do not include the model checkpoint in your ZIP file.** Your code should be able to load a model from a specified path (e.g., `checkpoints/model.pth`). The grading script will place the necessary model file there before running your code. Use the `.submission_ignore` file to exclude it.
- **Q: Should I include my own additional test files?**
  - A: Only include additional test files if explicitly allowed in the assignment instructions. Generally, submit only the files that were originally provided.
- **Q: Can I modify the test files?**
  - A: No, do not modify the provided test files. Your implementation should pass the original tests as provided.
- **Q: What if I added extra dependencies?**
  - A: Contact the course staff before submission. Generally, you should only use the dependencies specified in the assignment.



