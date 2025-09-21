#!/usr/bin/env bash
set -euo pipefail

# Usage: ./make_submission.sh <LASTNAME> <FIRSTNAME> <STUDENTID> [--notest]

SKIP_TESTS=false
if [ "$#" -eq 4 ] && [ "$4" == "--notest" ]; then
  SKIP_TESTS=true
elif [ "$#" -ne 3 ]; then
  echo "Usage: $0 <LASTNAME> <FIRSTNAME> <STUDENTID> [--notest]"
  echo "Example: $0 DOE JANE 12345678"
  exit 1
fi

LASTNAME=$1
FIRSTNAME=$2
STUDENTID=$3

ASSIGN_DIR="$(pwd)"
ASSIGN_NAME="$(basename "$ASSIGN_DIR")"  # e.g., hw0_hello_world

# Extract homework number from directory name
HW_NUM=$(echo "$ASSIGN_NAME" | sed -n 's/^hw\([0-9]\+\).*/\1/p')
if [ -z "$HW_NUM" ]; then
  echo "Error: could not infer homework number from '$ASSIGN_NAME'"
  exit 1
fi

if [ "$SKIP_TESTS" = false ]; then
  echo "Running tests in $ASSIGN_NAME..."
  set +e
  uv run pytest -q --ignore=LLaMA-Factory
  TEST_STATUS=$?
  set -e
  if [ $TEST_STATUS -ne 0 ]; then
    echo "❌ Tests failed. Fix failures before submitting."
    exit 1
  fi
  echo "✅ Tests passed. Creating submission..."
else
  echo "⚠️  Skipping tests as requested. Creating submission..."
fi

SUBMISSION_DIR="${ASSIGN_NAME}_submission"
SUBMISSION_ZIP="hw${HW_NUM}_submission_${LASTNAME}_${FIRSTNAME}_${STUDENTID}.zip"
IGNORE_FILE=".submission_ignore"

rm -rf "$SUBMISSION_DIR" "$SUBMISSION_ZIP" 2>/dev/null || true
mkdir -p "$SUBMISSION_DIR"

echo "Creating submission directory..."

# Build rsync command arguments
RSYNC_ARGS=("-a")
RSYNC_ARGS+=("--exclude=$SUBMISSION_DIR")
RSYNC_ARGS+=("--exclude=$SUBMISSION_ZIP")
RSYNC_ARGS+=("--exclude=.git")
RSYNC_ARGS+=("--exclude=.venv")
RSYNC_ARGS+=("--exclude=.pytest_cache")
RSYNC_ARGS+=("--exclude=__pycache__")
RSYNC_ARGS+=("--exclude=*.pyc")

# Use .submission_ignore if it exists
if [ -f "$IGNORE_FILE" ]; then
  echo "Excluding patterns from $IGNORE_FILE"
  RSYNC_ARGS+=("--exclude-from=$IGNORE_FILE")
fi

# Use rsync to copy files, excluding specified patterns.
rsync "${RSYNC_ARGS[@]}" . "$SUBMISSION_DIR/"

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command not found. Please install 'zip' and retry."
  rm -rf "$SUBMISSION_DIR"
  exit 1
fi

# The zip command can be simplified as we are zipping a clean directory
zip -r "$SUBMISSION_ZIP" "$SUBMISSION_DIR"

rm -rf "$SUBMISSION_DIR"
echo "✅ Submission created: $SUBMISSION_ZIP"
echo "📝 Contents:"
unzip -l "$SUBMISSION_ZIP" | cat


