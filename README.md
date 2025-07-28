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

## Installation

```bash
git clone https://github.com/abdulzedan/vto-product-recontext.git
cd vto-product-recontext
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
PROJECT_ID=your-gcp-project
GOOGLE_CLOUD_STORAGE=your-bucket-name
GEMINI_API_KEY=your-gemini-key
MAX_WORKERS=10
MAX_RETRIES=5
```

## Usage

1. Place your CSV file at `image_folder/img_conversion_table.csv`:
```csv
ID,Image Src,Image Command,Image Position
001,https://example.com/shirt.jpg,process,front
002,https://example.com/vase.jpg,enhance,center
```

2. Run the processor:
```bash
python -m bulk_image_processor
```

## Project Structure

```
src/bulk_image_processor/
├── main.py                  # Main processing logic
├── analyzer.py              # Gemini integration
├── downloader.py            # Image downloading
├── processors/
│   ├── virtual_try_on.py
│   └── product_recontext.py
└── utils.py

tests/                       # Test suite
.github/workflows/           # CI/CD
```

## Testing

```bash
pytest                           # Run tests
pytest --cov=src                 # With coverage
pytest tests/test_exceptions.py  # Specific module
```

## How It Works

1. Downloads images from CSV URLs
2. Classifies each image using Gemini
3. Routes apparel to Virtual Try-On processor
4. Routes products to Product Recontext processor
5. Validates output quality
6. Saves results to local and cloud storage

## Requirements

- Python 3.10+
- Google Cloud Project
- Gemini API access
- Virtual environment recommended
