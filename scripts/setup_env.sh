#!/bin/bash

# Setup script for bulk image processor environment

set -e

echo "Setting up bulk image processor environment..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/bulk_image_processor" ]; then
    echo "Error: Please run this script from the repository root directory"
    echo "Expected files: pyproject.toml, src/bulk_image_processor/"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "Error: Python 3.10 or higher is required (found $python_version)"
    exit 1
fi

# Check if virtual environment already exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, removing old one..."
    rm -rf .venv
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
if ! pip install -e ".[dev]"; then
    echo "Error: Failed to install dependencies"
    echo "This might be due to missing system dependencies or network issues"
    exit 1
fi

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
if ! pre-commit install; then
    echo "Warning: Failed to install pre-commit hooks (non-fatal)"
fi

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
    echo ""
    echo "⚠️  IMPORTANT: The .env file contains placeholder values that MUST be updated!"
    echo ""
else
    echo ".env file already exists"
    # Check for placeholder values
    if grep -q "your-project-id\|your-bucket-name\|your-gemini-api-key" .env; then
        echo ""
        echo "⚠️  WARNING: Your .env file still contains placeholder values!"
        echo "   Please update all 'your-*' values with actual credentials"
        echo ""
    fi
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
python3 -c "import sys; sys.path.insert(0, 'src'); import bulk_image_processor; print('✓ Package imports successfully')"

# Run basic tests (skip integration tests that require credentials)
echo "Running basic tests..."
pytest tests/test_exceptions.py tests/test_config.py::TestProcessingConfig -v || echo "⚠ Some tests may require credentials - this is normal for initial setup"

echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. REQUIRED: Edit .env file with your credentials:"
echo "   - PROJECT_ID: Replace 'your-project-id' with your Google Cloud project ID"
echo "   - GOOGLE_CLOUD_STORAGE: Replace 'your-bucket-name' with your GCS bucket name (with uniform access disabled)"
echo "   - GEMINI_API_KEY: Replace 'your-gemini-api-key' with your API key from Google AI Studio"
echo "2. Run Google Cloud setup script:"
echo "   ./scripts/setup_gcloud.sh"
echo "   This will configure your GCP project, enable APIs, and set up your bucket"
echo "3. Test setup: python -m bulk_image_processor --dry-run"
echo "4. Process sample: python -m bulk_image_processor --csv image_folder/test_mixed_10.csv --pipeline"
echo ""
echo "For development:"
echo "- Activate virtual environment: source .venv/bin/activate"
echo "- Run tests: make test"
echo "- Format code: make format"
echo "- Run linter: make lint"