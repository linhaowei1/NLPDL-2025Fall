#!/bin/bash
# Homework Submission Helper Script
# Usage: bash submit_hw.sh <hw_directory> <LASTNAME> <FIRSTNAME>
# Example: bash submit_hw.sh hw0_hello_world SMITH JOHN

HW_DIR=$1
LASTNAME=$2
FIRSTNAME=$3

if [ $# -ne 3 ]; then
    echo "Usage: bash submit_hw.sh <hw_directory> <LASTNAME> <FIRSTNAME>"
    echo "Example: bash submit_hw.sh hw0_hello_world SMITH JOHN"
    exit 1
fi

# Check if homework directory exists
if [ ! -d "$HW_DIR" ]; then
    echo "❌ Error: Directory '$HW_DIR' does not exist."
    exit 1
fi

# Extract homework number from directory name
HW_NUM=$(echo $HW_DIR | sed 's/hw\([0-9]\+\).*/\1/')

echo "📚 Preparing submission for $HW_DIR..."
echo "👤 Student: $FIRSTNAME $LASTNAME"

# Test the homework first
echo "🧪 Testing homework implementation..."
uv run pytest $HW_DIR/
if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Please fix your implementation before submitting."
    echo "💡 Run 'make test-hw$HW_NUM' to see detailed test results."
    exit 1
fi

echo "✅ All tests passed!"

# Create submission
SUBMISSION_NAME="${HW_DIR}_submission"
SUBMISSION_ZIP="hw${HW_NUM}_submission_${LASTNAME}_${FIRSTNAME}.zip"

echo "📦 Creating submission ZIP..."

# Remove existing submission files if they exist
rm -rf $SUBMISSION_NAME
rm -f $SUBMISSION_ZIP

# Copy homework directory
cp -r $HW_DIR $SUBMISSION_NAME

# Clean up cache files
echo "🧹 Cleaning up cache files..."
find $SUBMISSION_NAME -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find $SUBMISSION_NAME -name "*.pyc" -delete 2>/dev/null || true

# Create the ZIP file
zip -r $SUBMISSION_ZIP $SUBMISSION_NAME/ > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create ZIP file."
    rm -rf $SUBMISSION_NAME
    exit 1
fi

# Clean up temporary directory
rm -rf $SUBMISSION_NAME

echo "✅ Submission created: $SUBMISSION_ZIP"
echo ""
echo "📋 Submission Summary:"
echo "   File: $SUBMISSION_ZIP"
echo "   Size: $(du -h $SUBMISSION_ZIP | cut -f1)"
echo ""
echo "📝 Contents:"
unzip -l $SUBMISSION_ZIP

echo ""
echo "🎯 Next steps:"
echo "   1. Verify the contents above are correct"
echo "   2. Submit '$SUBMISSION_ZIP' through the course submission system"
echo "   3. Keep a backup of your submission file"
echo ""
echo "✨ Submission ready! Good luck! ✨"
