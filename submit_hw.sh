#!/usr/bin/env bash
set -euo pipefail

# submit_hw.sh
# Unified homework submission helper for all assignments.
# Usage: ./submit_hw.sh <hw_directory> <LASTNAME> <FIRSTNAME> <STUDENTID>

if [ $# -ne 4 ]; then
  echo "Usage: $0 <hw_directory> <LASTNAME> <FIRSTNAME> <STUDENTID>"
  echo "Example: $0 hw0_hello_world SMITH JOHN 11223344"
  exit 1
fi

HW_DIR=$1
LASTNAME=$2
FIRSTNAME=$3
STUDENTID=$4

if [ ! -d "$HW_DIR" ]; then
  echo "Error: directory '$HW_DIR' not found. Run this from the repo root."
  exit 1
fi

# Extract homework number from directory name (e.g., hw1_bpe_and_lm -> 1)
HW_NUM=$(echo "$HW_DIR" | sed -n 's/^hw\([0-9]\+\).*/\1/p')
if [ -z "$HW_NUM" ]; then
  echo "Error: could not infer homework number from '$HW_DIR' (expected name like hw1_*)."
  exit 1
fi

echo "Running tests for $HW_DIR..."
if ! command -v uv >/dev/null 2>&1; then
  echo "Warning: 'uv' not found on PATH. Ensure dependencies are installed."
fi

# Always run tests with the assignment's environment
set +e
uv run --directory "$HW_DIR" pytest -q
TEST_STATUS=$?
set -e
if [ $TEST_STATUS -ne 0 ]; then
  echo "❌ Tests failed. Fix failures before submitting."
  exit 1
fi
echo "✅ Tests passed. Creating submission..."

SUBMISSION_DIR="${HW_DIR}_submission"
SUBMISSION_ZIP="hw${HW_NUM}_submission_${LASTNAME}_${FIRSTNAME}_${STUDENTID}.zip"
IGNORE_FILE="${HW_DIR}/.submission_ignore"

# Clean previous outputs if any
rm -rf "$SUBMISSION_DIR" "$SUBMISSION_ZIP" 2>/dev/null || true

# Create clean copy of the homework directory
cp -r "$HW_DIR" "$SUBMISSION_DIR"

# Clean common cache artifacts
find "$SUBMISSION_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SUBMISSION_DIR" -name "*.pyc" -delete 2>/dev/null || true
rm -rf "$SUBMISSION_DIR/.venv" "$SUBMISSION_DIR/.pytest_cache" 2>/dev/null || true

# Build exclusion list from optional .submission_ignore
EXCLUDE_ARGS=()
if [ -f "$IGNORE_FILE" ]; then
  echo "Using .submission_ignore entries from $IGNORE_FILE"
  while IFS= read -r pattern; do
    # Skip empty or comment lines
    [[ -z "$pattern" || "$pattern" == \#* ]] && continue
    EXCLUDE_ARGS+=("-x" "${SUBMISSION_DIR}/${pattern}")
  done < "$IGNORE_FILE"
fi

# Create zip (requires 'zip' command)
if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command not found. Please install 'zip' and retry."
  rm -rf "$SUBMISSION_DIR"
  exit 1
fi

zip -r "$SUBMISSION_ZIP" "$SUBMISSION_DIR/" "${EXCLUDE_ARGS[@]}"

# Cleanup temp directory
rm -rf "$SUBMISSION_DIR"

echo "✅ Submission created: $SUBMISSION_ZIP"
echo "📝 Contents:"
unzip -l "$SUBMISSION_ZIP" | cat


