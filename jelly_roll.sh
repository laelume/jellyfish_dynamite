#!/bin/bash

echo "🎐 Jellyfish Dynamite - Audio Analysis Tool 🎐"
echo "================================================"
echo
echo "Setting up your environment..."
echo "This may take 2-3 minutes depending on your internet connection."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo
    echo "Please install Python 3.9+ from https://python.org"
    echo "Or use Homebrew: brew install python3"
    exit 1
fi

echo "✅ Found Python $(python3 --version)"

# Create virtual environment
echo
echo "📦 Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo
echo "📚 Installing required packages..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

echo
echo "✅ Installation complete!"
echo
echo "🚀 Starting Jellyfish Dynamite..."
echo
echo "When you see 'Running on http://127.0.0.1:5000',"
echo "open your web browser and go to: http://localhost:5000"
echo
echo "Press Ctrl+C to stop the server when you're done."
echo

# Start the Flask app
python jelly_app.py

echo
echo "👋 Jellyfish Dynamite has stopped."
read -p "Press Enter to exit..."