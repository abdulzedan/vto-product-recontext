#!/bin/bash

# Verification script for bulk image processor setup

set -e

echo "🔍 Verifying bulk image processor setup..."
echo ""

# Check directory structure
echo "📁 Checking directory structure..."
required_dirs=(
    "src/bulk_image_processor"
    "src/bulk_image_processor/processors"
    "tests"
    "image_folder"
    "scripts"
)

required_files=(
    "pyproject.toml"
    ".env.example" 
    ".pre-commit-config.yaml"
    "Makefile"
    "README.md"
    "src/bulk_image_processor/__init__.py"
    "src/bulk_image_processor/__main__.py"
    "src/bulk_image_processor/main.py"
    "src/bulk_image_processor/main_pipeline.py"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Directory: $dir"
    else
        echo "❌ Missing directory: $dir"
    fi
done

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ File: $file"
    else
        echo "❌ Missing file: $file"
    fi
done

echo ""

# Check Python and virtual environment
echo "🐍 Checking Python environment..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "✅ Python version: $python_version"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        echo "✅ Python version is 3.10+"
    else
        echo "❌ Python version is less than 3.10"
    fi
else
    echo "❌ Python 3 is not installed"
fi

if [ -d ".venv" ]; then
    echo "✅ Virtual environment exists"
    
    if [ -f ".venv/bin/activate" ]; then
        echo "✅ Virtual environment activation script exists"
    else
        echo "❌ Virtual environment activation script missing"
    fi
else
    echo "❌ Virtual environment missing"
fi

echo ""

# Test package import
echo "📦 Testing package import..."
if source .venv/bin/activate 2>/dev/null && python3 -c "import sys; sys.path.insert(0, 'src'); import bulk_image_processor; print('✅ Package imports successfully')" 2>/dev/null; then
    echo "✅ Package import successful"
else
    echo "❌ Package import failed"
fi

echo ""

# Check configuration
echo "⚙️ Checking configuration..."
if [ -f ".env" ]; then
    echo "✅ .env file exists"
    
    # Check for required variables
    required_vars=(
        "PROJECT_ID"
        "GOOGLE_CLOUD_STORAGE"
        "GEMINI_API_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if grep -q "^$var=" .env && ! grep -q "^$var=your-" .env; then
            echo "✅ $var is configured"
        else
            echo "⚠️  $var needs to be configured"
        fi
    done
else
    echo "⚠️  .env file missing (copy from .env.example)"
fi

echo ""

# Check sample CSV files
echo "📄 Checking sample CSV files..."
csv_files=(
    "image_folder/img_conversion_table.csv"
    "image_folder/test_mixed_10.csv"
    "image_folder/test_accessories_3.csv"
)

for csv in "${csv_files[@]}"; do
    if [ -f "$csv" ]; then
        line_count=$(wc -l < "$csv")
        echo "✅ $csv ($line_count lines)"
    else
        echo "❌ Missing: $csv"
    fi
done

echo ""

# Check Google Cloud authentication
echo "☁️ Checking Google Cloud authentication..."
if command -v gcloud &> /dev/null; then
    echo "✅ gcloud CLI is installed"
    
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
        echo "✅ Active gcloud account: $active_account"
    else
        echo "⚠️  No active gcloud account (run: gcloud auth application-default login)"
    fi
else
    echo "⚠️  gcloud CLI not installed (optional but recommended)"
fi

if [ -n "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
    if [ -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
        echo "✅ Service account key file exists: $GOOGLE_APPLICATION_CREDENTIALS"
    else
        echo "❌ Service account key file not found: $GOOGLE_APPLICATION_CREDENTIALS"
    fi
else
    echo "ℹ️  GOOGLE_APPLICATION_CREDENTIALS not set (using application default credentials)"
fi

echo ""

# Test basic functionality (if credentials are configured)
echo "🧪 Testing basic functionality..."
if [ -f ".env" ] && source .venv/bin/activate 2>/dev/null; then
    if python3 -m bulk_image_processor --dry-run 2>/dev/null; then
        echo "✅ Dry run successful - setup is working!"
    else
        echo "⚠️  Dry run failed - may need credential configuration"
    fi
else
    echo "⚠️  Skipping functional test - complete setup first"
fi

echo ""
echo "📋 Setup Summary:"
echo "================"

if [ -f ".env" ] && [ -d ".venv" ] && python3 -c "import sys; sys.path.insert(0, 'src'); import bulk_image_processor" 2>/dev/null; then
    echo "✅ Core setup is complete"
    echo ""
    echo "Next steps to start processing:"
    echo "1. Configure credentials in .env file"
    echo "2. Set up Google Cloud authentication"
    echo "3. Test with: python -m bulk_image_processor --dry-run"
    echo "4. Process sample: python -m bulk_image_processor --csv image_folder/test_mixed_10.csv --pipeline"
else
    echo "❌ Setup needs completion"
    echo ""
    echo "Run the setup script: ./scripts/setup_env.sh"
fi

echo ""