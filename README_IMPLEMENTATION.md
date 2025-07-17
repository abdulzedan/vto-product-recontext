# Bulk Image Processing POC

A Python-based bulk image processing system that integrates Google's Virtual Try-On and Product Recontext features with intelligent routing and automated quality control.

## Features

- **Bulk Image Processing**: Downloads and processes images from CSV URLs
- **Intelligent Routing**: Uses Gemini AI to classify images as apparel or products
- **Virtual Try-On**: Automatically pairs apparel with appropriate models
- **Product Recontext**: Generates high-end retail scenes for products
- **Quality Assurance**: Automated feedback loop with retry mechanisms
- **Multi-threading**: Parallel processing for high throughput
- **Dual Storage**: Outputs to both Google Cloud Storage and local folders
- **Comprehensive Logging**: Structured logging with performance metrics

## Architecture

```
Input CSV → Image Download → Gemini Classification → Processing → Quality Check → Output
                                     ↓                      ↓
                              [Apparel/Product]    [VTO/Recontext]
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Google Cloud SDK with authentication configured
- Access to Google Cloud Vertex AI and Gemini APIs

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd vto-product-recontext
   chmod +x scripts/setup_env.sh
   ./scripts/setup_env.sh
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Prepare your data**:
   - Place your CSV file at `image_folder/img_conversion_table.csv`
   - CSV format: `ID,Image Src,Image Command,Image Position`

### Usage

```bash
# Basic usage
python -m bulk_image_processor

# With custom options
python -m bulk_image_processor --csv /path/to/images.csv --max-workers 20

# Dry run to test configuration
python -m bulk_image_processor --dry-run

# Run example
python scripts/run_example.py
```

## Configuration

### Environment Variables

```bash
# Google Cloud
PROJECT_ID=your-project-id
LOCATION=us-central1
MODEL_ENDPOINT=virtual-try-on-exp-05-31
MODEL_ENDPOINT_PRODUCT=imagen-product-recontext-preview-06-30
GOOGLE_CLOUD_STORAGE=your-bucket-name

# API Keys
GEMINI_API_KEY=your-gemini-api-key

# Processing
MAX_WORKERS=10
MAX_RETRIES=5
DOWNLOAD_TIMEOUT=30
PROCESSING_TIMEOUT=300

# Storage
LOCAL_OUTPUT_DIR=./output
ENABLE_GCS_UPLOAD=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### CSV Format

Your input CSV should have these columns:

| Column | Description |
|--------|-------------|
| ID | Unique identifier for the image |
| Image Src | URL of the image to process |
| Image Command | Processing instruction (optional) |
| Image Position | Position information (optional) |

Example:
```csv
ID,Image Src,Image Command,Image Position
001,https://example.com/shirt.jpg,process,front
002,https://example.com/vase.jpg,enhance,center
```

## Output Structure

```
output/
├── virtual_try_on/
│   ├── 20240117_143022_001/
│   │   ├── result.jpg
│   │   └── metadata.json
└── product_recontext/
    ├── 20240117_143023_002/
    │   ├── result.jpg
    │   ├── prompt.txt
    │   └── metadata.json
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_downloader.py -v
```

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **mypy**: Static type checking
- **isort**: Import sorting
- **pre-commit**: Git hooks for code quality

## System Requirements

### Minimum Requirements
- Python 3.10+
- 4GB RAM
- 10GB disk space
- Internet connection for API calls

### Recommended Requirements
- Python 3.11+
- 8GB RAM
- 50GB disk space
- High-speed internet connection

## Google Cloud Setup

### Required APIs
- Vertex AI API
- Gemini API
- Cloud Storage API

### IAM Permissions
- Vertex AI User
- Storage Object Admin
- AI Platform Developer

### Bucket Structure
```
gs://your-bucket/
├── virtual-try-on/
│   ├── inputs/
│   ├── outputs/
│   └── metadata/
├── product-recontext/
│   ├── inputs/
│   ├── outputs/
│   └── metadata/
└── logs/
```

## Performance Tuning

### Optimize for Throughput
```bash
# Increase workers for I/O bound tasks
python -m bulk_image_processor --max-workers 20

# Batch processing for large datasets
python -m bulk_image_processor --batch-size 500
```

### Optimize for Quality
```bash
# Increase retries for better success rate
python -m bulk_image_processor --max-retries 8

# Enable comprehensive logging
python -m bulk_image_processor --log-level DEBUG
```

## Monitoring

### Logs
- Processing logs: `./logs/processing.log`
- Error logs: `./logs/errors.log`
- Performance metrics in structured JSON format

### Metrics
- Processing speed (images/minute)
- Success rate (%)
- Quality scores
- Retry statistics
- Storage usage

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   # Check credentials
   gcloud auth list
   gcloud auth application-default login
   ```

2. **Memory Issues**
   ```bash
   # Reduce workers
   python -m bulk_image_processor --max-workers 5
   ```

3. **API Rate Limits**
   ```bash
   # Increase timeouts
   export PROCESSING_TIMEOUT=600
   ```

### Debug Mode
```bash
# Enable debug logging
python -m bulk_image_processor --log-level DEBUG

# Dry run to test setup
python -m bulk_image_processor --dry-run
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `make format lint test`
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs in `./logs/`
- Create an issue in the repository