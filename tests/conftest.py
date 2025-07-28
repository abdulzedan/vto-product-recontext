"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from bulk_image_processor.config import Settings
from bulk_image_processor.downloader import ImageRecord


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    import os
    # Set required environment variables for testing
    os.environ.update({
        'PROJECT_ID': 'test-project',
        'GOOGLE_CLOUD_STORAGE': 'test-bucket',
        'GEMINI_API_KEY': 'test-gemini-key'
    })
    
    try:
        return Settings(
            project_id="test-project",
            location="us-central1",
            model_endpoint="test-vto-endpoint",
            model_endpoint_product="test-product-endpoint",
            google_cloud_storage="test-bucket",
            gemini_api_key="test-gemini-key",
            max_workers=2,
            max_retries=2,
            download_timeout=10,
            processing_timeout=30,
            local_output_dir="./test_output",
            enable_gcs_upload=False,
            log_level="INFO",
            log_format="json",
        )
    finally:
        # Clean up environment variables
        for key in ['PROJECT_ID', 'GOOGLE_CLOUD_STORAGE', 'GEMINI_API_KEY']:
            os.environ.pop(key, None)


@pytest.fixture
def sample_image_record():
    """Sample image record for testing."""
    return ImageRecord(
        id="test_001",
        image_url="https://example.com/test.jpg",
        image_command="test command",
        image_position="center",
        row_index=0,
    )


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return [
        ["ID", "Image Src", "Image Command", "Image Position"],
        ["001", "https://example.com/shirt.jpg", "process", "front"],
        ["002", "https://example.com/pants.jpg", "enhance", "side"],
        ["003", "https://example.com/vase.jpg", "recontextualize", "center"],
    ]


@pytest.fixture
def temp_csv_file(tmp_path, sample_csv_data):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_images.csv"
    with open(csv_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerows(sample_csv_data)
    return csv_path


@pytest.fixture
def mock_analyzer():
    """Mock analyzer for testing."""
    analyzer = MagicMock()
    analyzer.classify_image.return_value = MagicMock(
        category="apparel",
        confidence=0.9,
        reasoning="Test classification",
        detected_items=["shirt"],
        metadata={"style": "casual"},
    )
    return analyzer


@pytest.fixture
def mock_storage_manager():
    """Mock storage manager for testing."""
    storage_manager = MagicMock()
    storage_manager.get_local_output_dir.return_value = Path("./test_output")
    storage_manager.upload_file.return_value = "gs://test-bucket/test/path"
    return storage_manager