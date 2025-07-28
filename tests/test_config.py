"""Tests for configuration management."""

import pytest
import os
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from bulk_image_processor.config import (
    Settings, ProcessingConfig, GoogleCloudConfig, 
    GeminiConfig, StorageConfig, LoggingConfig, get_settings
)
from bulk_image_processor.exceptions import ConfigurationError


class TestProcessingConfig:
    """Test ProcessingConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        
        assert config.max_workers == 10
        assert config.max_retries == 5
        assert config.download_timeout == 30
        assert config.processing_timeout == 300
        assert config.batch_size == 100
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            max_workers=20,
            max_retries=3,
            download_timeout=60,
            processing_timeout=600,
            batch_size=200
        )
        
        assert config.max_workers == 20
        assert config.max_retries == 3
        assert config.download_timeout == 60
        assert config.processing_timeout == 600
        assert config.batch_size == 200
    
    def test_validation_boundaries(self):
        """Test validation of boundary values."""
        # Test minimum values
        config = ProcessingConfig(max_workers=1, max_retries=1)
        assert config.max_workers == 1
        assert config.max_retries == 1
        
        # Test maximum values
        config = ProcessingConfig(max_workers=50, max_retries=10)
        assert config.max_workers == 50
        assert config.max_retries == 10
    
    def test_invalid_values(self):
        """Test validation errors for invalid values."""
        with pytest.raises(ValidationError):
            ProcessingConfig(max_workers=0)  # Too low
        
        with pytest.raises(ValidationError):
            ProcessingConfig(max_workers=51)  # Too high
        
        with pytest.raises(ValidationError):
            ProcessingConfig(max_retries=11)  # Too high


class TestGoogleCloudConfig:
    """Test GoogleCloudConfig model."""
    
    def test_required_fields(self):
        """Test required fields validation."""
        # Valid config
        config = GoogleCloudConfig(
            project_id="test-project",
            storage_bucket="test-bucket"
        )
        
        assert config.project_id == "test-project"
        assert config.storage_bucket == "test-bucket"
        assert config.location == "us-central1"  # default
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError):
            GoogleCloudConfig(storage_bucket="test-bucket")  # Missing project_id
        
        with pytest.raises(ValidationError):
            GoogleCloudConfig(project_id="test-project")  # Missing storage_bucket
    
    def test_empty_strings(self):
        """Test that empty strings are rejected."""
        with pytest.raises(ValidationError):
            GoogleCloudConfig(project_id="", storage_bucket="test-bucket")


class TestGeminiConfig:
    """Test GeminiConfig model."""
    
    def test_default_values(self):
        """Test default Gemini configuration."""
        config = GeminiConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.model_name == "gemini-1.5-flash"
        assert config.temperature == 0.1
        assert config.max_output_tokens == 1024
    
    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid temperature
        config = GeminiConfig(api_key="key", temperature=1.5)
        assert config.temperature == 1.5
        
        # Boundary values
        config = GeminiConfig(api_key="key", temperature=0.0)
        assert config.temperature == 0.0
        
        config = GeminiConfig(api_key="key", temperature=2.0)
        assert config.temperature == 2.0
        
        # Invalid values
        with pytest.raises(ValidationError):
            GeminiConfig(api_key="key", temperature=-0.1)
        
        with pytest.raises(ValidationError):
            GeminiConfig(api_key="key", temperature=2.1)


class TestStorageConfig:
    """Test StorageConfig model."""
    
    def test_default_values(self):
        """Test default storage configuration."""
        config = StorageConfig()
        
        assert config.local_output_dir == Path("./output")
        assert config.enable_gcs_upload is True
        assert config.gcs_virtual_try_on_path == "virtual-try-on"
        assert config.gcs_product_recontext_path == "product-recontext"
        assert config.gcs_logs_path == "logs"
    
    def test_custom_paths(self):
        """Test custom storage paths."""
        config = StorageConfig(
            local_output_dir=Path("/custom/output"),
            enable_gcs_upload=False,
            gcs_virtual_try_on_path="custom-vto",
            gcs_product_recontext_path="custom-pr"
        )
        
        assert config.local_output_dir == Path("/custom/output")
        assert config.enable_gcs_upload is False
        assert config.gcs_virtual_try_on_path == "custom-vto"


class TestLoggingConfig:
    """Test LoggingConfig model."""
    
    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.log_file == Path("./logs/processing.log")
    
    def test_custom_values(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            level="DEBUG",
            format="text",
            log_file=Path("/var/log/processor.log")
        )
        
        assert config.level == "DEBUG"
        assert config.format == "text"
        assert config.log_file == Path("/var/log/processor.log")


class TestSettings:
    """Test main Settings class."""
    
    @patch.dict(os.environ, {
        'PROJECT_ID': 'test-project-env',
        'LOCATION': 'europe-west1',
        'MODEL_ENDPOINT': 'custom-vto-endpoint',
        'MODEL_ENDPOINT_PRODUCT': 'custom-product-endpoint',
        'GOOGLE_CLOUD_STORAGE': 'test-bucket-env',
        'GEMINI_API_KEY': 'test-gemini-key-env',
        'MAX_WORKERS': '15',
        'MAX_RETRIES': '3',
        'DOWNLOAD_TIMEOUT': '45',
        'PROCESSING_TIMEOUT': '400',
        'LOCAL_OUTPUT_DIR': '/custom/output',
        'ENABLE_GCS_UPLOAD': 'false',
        'LOG_LEVEL': 'DEBUG',
        'LOG_FORMAT': 'text'
    })
    def test_env_loading(self):
        """Test loading settings from environment variables."""
        settings = Settings()
        
        assert settings.project_id == 'test-project-env'
        assert settings.location == 'europe-west1'
        assert settings.model_endpoint == 'custom-vto-endpoint'
        assert settings.model_endpoint_product == 'custom-product-endpoint'
        assert settings.google_cloud_storage == 'test-bucket-env'
        assert settings.gemini_api_key == 'test-gemini-key-env'
        assert settings.max_workers == 15
        assert settings.max_retries == 3
        assert settings.download_timeout == 45
        assert settings.processing_timeout == 400
        assert settings.local_output_dir == '/custom/output'
        assert settings.enable_gcs_upload is False
        assert settings.log_level == 'DEBUG'
        assert settings.log_format == 'text'
    
    def test_required_env_vars(self):
        """Test that required environment variables are validated."""
        # Remove required env vars
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            
            # Just verify that validation error occurs - don't check specific fields
            assert len(exc_info.value.errors()) > 0
    
    @patch.dict(os.environ, {
        'PROJECT_ID': 'test-project',
        'GOOGLE_CLOUD_STORAGE': 'test-bucket',
        'GEMINI_API_KEY': 'test-key'
    })
    def test_property_accessors(self):
        """Test property accessors for configuration groups."""
        settings = Settings(
            max_workers=20,
            max_retries=3,
            download_timeout=60,
            processing_timeout=600,
            local_output_dir="/test/output",
            enable_gcs_upload=True,
            log_level="INFO",
            log_format="json"
        )
        
        # Test processing config
        processing = settings.processing
        assert isinstance(processing, ProcessingConfig)
        assert processing.max_workers == 20
        assert processing.max_retries == 3
        
        # Test Google Cloud config
        gcp = settings.google_cloud
        assert isinstance(gcp, GoogleCloudConfig)
        assert gcp.project_id == "test-project"
        assert gcp.storage_bucket == "test-bucket"
        
        # Test Gemini config
        gemini = settings.gemini
        assert isinstance(gemini, GeminiConfig)
        assert gemini.api_key == "test-key"
        
        # Test storage config
        storage = settings.storage
        assert isinstance(storage, StorageConfig)
        assert storage.local_output_dir == Path("/test/output")
        assert storage.enable_gcs_upload is True
        
        # Test logging config
        logging = settings.logging
        assert isinstance(logging, LoggingConfig)
        assert logging.level == "INFO"
        assert logging.format == "json"
    
    @patch.dict(os.environ, {
        'PROJECT_ID': 'env-project', 
        'GOOGLE_CLOUD_STORAGE': 'env-bucket', 
        'GEMINI_API_KEY': 'env-key'
    })
    def test_get_settings_function(self):
        """Test get_settings helper function."""
        settings = get_settings()
        
        assert isinstance(settings, Settings)
        assert settings.project_id == 'env-project'
        assert settings.google_cloud_storage == 'env-bucket'
        assert settings.gemini_api_key == 'env-key'
    
    def test_env_file_loading(self, tmp_path):
        """Test loading settings from .env file."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("""
PROJECT_ID=file-project
GOOGLE_CLOUD_STORAGE=file-bucket
GEMINI_API_KEY=file-key
MAX_WORKERS=25
LOG_LEVEL=WARNING
""")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            settings = Settings()
            
            assert settings.project_id == 'file-project'
            assert settings.google_cloud_storage == 'file-bucket'
            assert settings.gemini_api_key == 'file-key'
            assert settings.max_workers == 25
            assert settings.log_level == 'WARNING'
        finally:
            os.chdir(original_cwd)