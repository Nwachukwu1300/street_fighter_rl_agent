#!/bin/bash
# Fix DIAMBRA installation issues

echo "Fixing DIAMBRA Arena installation..."
echo "======================================"

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Step 1: Uninstalling conflicting packages..."
pip uninstall -y diambra diambra-arena diambra-engine

echo ""
echo "Step 2: Installing DIAMBRA Arena with Stable Baselines3 support..."
pip install diambra-arena[stable-baselines3]

echo ""
echo "Step 3: Verifying installation..."
python diagnose.py

echo ""
echo "======================================"
echo "Installation fix complete!"
