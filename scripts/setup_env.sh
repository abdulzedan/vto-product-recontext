#!/bin/bash

# Setup script for bulk image processor environment

set -e

echo "Setting up bulk image processor environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "Error: Python 3.10 or higher is required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p output/virtual_try_on
mkdir -p output/product_recontext
mkdir -p logs
mkdir -p image_folder

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your credentials"
else
    echo ".env file already exists"
fi

# Create sample CSV file if it doesn't exist
if [ ! -f image_folder/img_conversion_table.csv ]; then
    echo "Creating sample CSV file..."
    cat > image_folder/img_conversion_table.csv << EOF
ID,Image Src,Image Command,Image Position
sample_001,https://example.com/sample1.jpg,process,front
sample_002,https://example.com/sample2.jpg,enhance,center
sample_003,https://example.com/sample3.jpg,recontextualize,side
EOF
    echo "Sample CSV created at image_folder/img_conversion_table.csv"
fi

# Test imports
echo "Testing imports..."
python3 -c "import bulk_image_processor; print('✓ Package imports successfully')"

# Run tests
echo "Running tests..."
pytest tests/ -v

echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Google Cloud credentials"
echo "2. Add your image URLs to image_folder/img_conversion_table.csv"
echo "3. Run the processor: python -m bulk_image_processor"
echo ""
echo "For development:"
echo "- Activate virtual environment: source .venv/bin/activate"
echo "- Run tests: make test"
echo "- Format code: make format"
echo "- Run linter: make lint"