#!/bin/bash

# Navigate to the directory containing this script
cd "$(dirname "$0")"

echo "🚀 Starting Mac AI Assistant..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install it from python.org"
    read -p "Press enter to exit..."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install/Update dependencies
echo "⬇️  Checking dependencies..."
# Install the core mlx-use library (in editable mode)
pip install -e . --quiet
# Install app-specific dependencies
pip install -r gradio_app/requirements.txt --quiet

# Run the application
echo "✨ Launching App..."
python gradio_app/app.py

# Keep terminal open if it crashes
if [ $? -ne 0 ]; then
    echo "❌ App crashed or closed unexpectedly."
    read -p "Press enter to exit..."
fi
