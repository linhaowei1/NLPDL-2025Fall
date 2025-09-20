#!/usr/bin/env bash
set -euo pipefail

# Self-contained submission script for hw1.
# Usage: ./make_submission.sh <LASTNAME> <FIRSTNAME> <STUDENTID>

if [ $# -ne 3 ]; then
  echo "Usage: $0 <LASTNAME> <FIRSTNAME> <STUDENTID>"
  echo "Example: $0 DOE JANE 12345678"
  exit 1
fi

LASTNAME=$1
FIRSTNAME=$2
STUDENTID=$3

ASSIGN_DIR="$(pwd)"
ASSIGN_NAME="$(basename "$ASSIGN_DIR")"  # expected hw1_bpe_and_lm

HW_NUM=$(echo "$ASSIGN_NAME" | sed -n 's/^hw\([0-9]\+\).*/\1/p')
if [ -z "$HW_NUM" ]; then
  echo "Error: could not infer homework number from '$ASSIGN_NAME'"
  exit 1
fi

echo "Running tests in $ASSIGN_NAME..."
set +e
uv run pytest -q
TEST_STATUS=$?
set -e
if [ $TEST_STATUS -ne 0 ]; then
  echo "❌ Tests failed. Fix failures before submitting."
  exit 1
fi
echo "✅ Tests passed. Creating submission..."

SUBMISSION_DIR="${ASSIGN_NAME}_submission"
SUBMISSION_ZIP="hw${HW_NUM}_submission_${LASTNAME}_${FIRSTNAME}_${STUDENTID}.zip"
IGNORE_FILE=".submission_ignore"

rm -rf "$SUBMISSION_DIR" "$SUBMISSION_ZIP" 2>/dev/null || true
cp -r "$ASSIGN_DIR" "$SUBMISSION_DIR"

# Clean common cache artifacts
find "$SUBMISSION_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SUBMISSION_DIR" -name "*.pyc" -delete 2>/dev/null || true
rm -rf "$SUBMISSION_DIR/.venv" "$SUBMISSION_DIR/.pytest_cache" 2>/dev/null || true

# Respect optional .submission_ignore
EXCLUDE_ARGS=()
if [ -f "$IGNORE_FILE" ]; then
  echo "Using .submission_ignore entries from $IGNORE_FILE"
  while IFS= read -r pattern; do
    [[ -z "$pattern" || "$pattern" == \#* ]] && continue
    EXCLUDE_ARGS+=("-x" "${SUBMISSION_DIR}/${pattern}")
  done < "$IGNORE_FILE"
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command not found. Please install 'zip' and retry."
  rm -rf "$SUBMISSION_DIR"
  exit 1
fi

zip -r "$SUBMISSION_ZIP" "$SUBMISSION_DIR/" "${EXCLUDE_ARGS[@]}"

rm -rf "$SUBMISSION_DIR"
echo "✅ Submission created: $SUBMISSION_ZIP"
echo "📝 Contents:"
unzip -l "$SUBMISSION_ZIP" | cat