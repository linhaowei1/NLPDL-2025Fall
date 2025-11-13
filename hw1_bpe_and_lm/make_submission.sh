#!/usr/bin/env bash
set -euo pipefail

# Self-contained submission script for hw1.
# Usage: ./make_submission.sh <LASTNAME> <FIRSTNAME> <STUDENTID>

# Check args
if [ $# -ne 3 ]; then
  echo "Usage: $0 <LASTNAME> <FIRSTNAME> <STUDENTID>"
  echo "Example: $0 DOE JANE 12345678"
  exit 1
fi

LASTNAME=$1
FIRSTNAME=$2
STUDENTID=$3

# Infer hw information
ASSIGN_DIR="$(pwd)"
ASSIGN_NAME="$(basename "$ASSIGN_DIR")"  # expected hw1_bpe_and_lm

HW_NUM=$(echo "$ASSIGN_NAME" | sed -n 's/^hw\([0-9]\+\).*/\1/p')
if [ -z "$HW_NUM" ]; then
  echo "Error: could not infer homework number from '$ASSIGN_NAME'"
  exit 1
fi

# Run pytest
echo "Running tests in $ASSIGN_NAME..."
set +e
uv run pytest -q
TEST_STATUS=$?
set -e
if [ $TEST_STATUS -ne 0 ]; then
  echo "Tests failed. Fix failures before submitting."
  exit 1
fi
echo "Tests passed. Creating submission..."

# Make tmp dir
SUBMISSION_DIR="${ASSIGN_NAME}_submission"
SUBMISSION_ZIP="hw${HW_NUM}_submission_${LASTNAME}_${FIRSTNAME}_${STUDENTID}.zip"

rm -rf "$SUBMISSION_DIR" "$SUBMISSION_ZIP" 2>/dev/null || true
mkdir -p "$SUBMISSION_DIR"

# Copy files
find "$ASSIGN_DIR" -mindepth 1 -maxdepth 1 \
  -not -name "$SUBMISSION_DIR" \
  -not -name "$SUBMISSION_ZIP" \
  -exec cp -r {} "$SUBMISSION_DIR/" \;

# Clean common cache artifacts
find "$SUBMISSION_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SUBMISSION_DIR" -name "*.pyc" -delete 2>/dev/null || true
rm -rf "$SUBMISSION_DIR/.venv" "$SUBMISSION_DIR/.pytest_cache"  "$SUBMISSION_DIR/tests/fixtures" "$SUBMISSION_DIR/tests/_snapshots" 2>/dev/null || true

# Zip
if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command not found. Please install 'zip' and retry."
  rm -rf "$SUBMISSION_DIR"
  exit 1
fi

zip -r "$SUBMISSION_ZIP" "$SUBMISSION_DIR/" "${EXCLUDE_ARGS[@]}"

# Clear tmp dir
rm -rf "$SUBMISSION_DIR"

# Finish
echo "Submission created: $SUBMISSION_ZIP"
echo "Contents:"
unzip -l "$SUBMISSION_ZIP" | cat