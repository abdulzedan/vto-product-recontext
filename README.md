# VTO Product Recontext

[![Tests](https://github.com/abdulzedan/vto-product-recontext/actions/workflows/test.yml/badge.svg)](https://github.com/abdulzedan/vto-product-recontext/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Bulk image processing system that routes product images to Virtual Try-On for apparel or Product Recontext for other items using Gemini AI classification.

## Features

- Automatic image classification (apparel vs products)
- Virtual Try-On processing for clothing items
- Product recontextualization for retail scenes
- Parallel processing with configurable workers
- Google Cloud Storage integration
- CSV-based batch processing
- Retry mechanism with quality validation

## Quick Setup

```bash
git clone https://github.com/abdulzedan/vto-product-recontext.git
cd vto-product-recontext
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # Edit with your credentials
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required
PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_STORAGE=your-bucket-name
GEMINI_API_KEY=your-gemini-api-key

# Optional (has defaults)
MAX_WORKERS=10
MAX_RETRIES=5
LOCATION=us-central1
```

**Important**: Disable uniform bucket-level access on your GCS bucket:
```bash
gsutil uniformbucketlevelaccess set off gs://your-bucket-name
```

## Usage

### Basic Usage
```bash
# Process default CSV
python -m bulk_image_processor

# Process specific CSV
python -m bulk_image_processor --csv your-file.csv

# Use pipeline mode (faster first results)
python -m bulk_image_processor --csv your-file.csv --pipeline

# Adjust workers
python -m bulk_image_processor --max-workers 15
```

### CSV Format
```csv
ID,Image Src,Image Command,Image Position
9848782618913,https://example.com/shirt.jpg,MERGE,1
accessory_001,https://example.com/belt.jpg,brown belt,waist
```

- **MERGE** + numeric position → Virtual Try-On processing
- Descriptive text + body position → Product Recontext processing

## Project Structure

```
src/bulk_image_processor/
├── __main__.py              # CLI entry point
├── main.py                  # Standard processing mode
├── main_pipeline.py         # Pipeline mode (parallel processing)
├── analyzer.py              # Gemini AI integration
├── downloader.py            # Image downloading
├── storage.py               # Local/GCS storage
├── processors/
│   ├── virtual_try_on.py    # VTO processor
│   └── product_recontext.py # Product recontext processor
└── utils.py

tests/                       # Test suite
scripts/                     # Setup scripts
.github/workflows/           # CI/CD
```

## Testing

```bash
make test                        # Run tests
make test-cov                    # With coverage report
pytest tests/test_config.py -v   # Specific module

# Verify setup
python -m bulk_image_processor --dry-run
```

## How It Works

### Pipeline Mode (Default)
1. **Download Stage**: Downloads images from CSV URLs in parallel
2. **Classification Stage**: Classifies images using Gemini AI
3. **Parallel Processing**: 
   - Apparel → Virtual Try-On processor
   - Products/Accessories → Product Recontext processor
4. **Quality Validation**: Validates output quality and saves results
5. **Storage**: Saves to local directory and uploads to GCS with public access

### Processing Flow
- **VTO Processing**: Fashion coordination, model selection, quality validation
- **Product Recontext**: Scene recontextualization for retail environments
- **True Parallelism**: Both processors run simultaneously for optimal performance

## Requirements

- Python 3.10+
- Google Cloud Project
- Gemini API access
- Virtual environment recommended
