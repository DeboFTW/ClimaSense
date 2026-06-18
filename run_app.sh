#!/bin/bash

# ClimaSense - Quick Run Script
echo "🚀 Starting ClimaSense Hybrid ML..."
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Please create one: python3 -m venv venv"
    exit 1
fi

# Run the application
python3 main.py
