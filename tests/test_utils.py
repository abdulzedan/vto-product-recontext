"""Tests for utility functions."""

import pytest
import asyncio
import base64
import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from PIL import Image
import structlog

from bulk_image_processor.utils import (
    setup_logging, generate_unique_id, validate_image_url,
    ensure_directory, encode_image_to_base64, decode_base64_to_image,
    prediction_to_pil_image, prepare_image_for_api, save_image_with_metadata,
    upload_to_gcs, download_from_gcs, retry_with_backoff, async_retry_with_backoff,
    calculate_processing_stats, ProgressTracker
)
from bulk_image_processor.exceptions import StorageError


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    @patch('structlog.configure')
    def test_setup_logging_json(self, mock_configure):
        """Test setting up JSON logging."""
        setup_logging(level="INFO", format_type="json")
        
        mock_configure.assert_called_once()
        call_args = mock_configure.call_args[1]
        processors = call_args['processors']
        
        # Check that JSON renderer is used
        assert any('JSONRenderer' in str(p) for p in processors)
    
    @patch('structlog.configure')
    def test_setup_logging_console(self, mock_configure):
        """Test setting up console logging."""
        setup_logging(level="DEBUG", format_type="console")
        
        mock_configure.assert_called_once()
        call_args = mock_configure.call_args[1]
        processors = call_args['processors']
        
        # Check that console renderer is used
        assert any('ConsoleRenderer' in str(p) for p in processors)


class TestUniqueIdGeneration:
    """Test unique ID generation."""
    
    def test_generate_unique_id_no_prefix(self):
        """Test generating unique ID without prefix."""
        id1 = generate_unique_id()
        id2 = generate_unique_id()
        
        assert id1 != id2
        assert len(id1) >= 15  # Should include timestamp
    
    def test_generate_unique_id_with_prefix(self):
        """Test generating unique ID with prefix."""
        id1 = generate_unique_id("test")
        
        assert id1.startswith("test_")
        assert len(id1) > len("test_") + 15
    
    def test_unique_ids_in_quick_succession(self):
        """Test that IDs are unique even when generated quickly."""
        ids = [generate_unique_id("test") for _ in range(100)]
        
        assert len(set(ids)) == 100  # All should be unique


class TestUrlValidation:
    """Test URL validation."""
    
    def test_valid_image_urls(self):
        """Test validation of valid image URLs."""
        valid_urls = [
            "https://example.com/image.jpg",
            "http://test.com/photo.png",
            "https://cdn.example.com/path/to/image.jpeg",
            "https://example.com/image.GIF",
            "https://example.com/image.WEBP",
        ]
        
        for url in valid_urls:
            assert validate_image_url(url) is True
    
    def test_invalid_image_urls(self):
        """Test validation of invalid image URLs."""
        invalid_urls = [
            "not-a-url",
            "https://example.com/document.pdf",
            "https://example.com/video.mp4",
            "https://example.com/no-extension",
            "ftp://example.com/image.jpg",
            "",
            "https://",
            "example.com/image.jpg",  # Missing scheme
        ]
        
        for url in invalid_urls:
            result = validate_image_url(url)
            assert result is False, f"URL {url} should be invalid but was considered valid"


class TestDirectoryOperations:
    """Test directory operations."""
    
    def test_ensure_directory_creates_new(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "test" / "nested" / "dir"
        
        assert not new_dir.exists()
        ensure_directory(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_ensure_directory_existing(self, tmp_path):
        """Test ensure_directory with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        # Should not raise an error
        ensure_directory(existing_dir)
        assert existing_dir.exists()


class TestImageEncoding:
    """Test image encoding/decoding functions."""
    
    def test_encode_decode_image(self):
        """Test encoding and decoding an image."""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Encode to base64
        encoded = encode_image_to_base64(img)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Decode back
        decoded = decode_base64_to_image(encoded)
        assert isinstance(decoded, Image.Image)
        assert decoded.size == (100, 100)
        assert decoded.mode == 'RGB'
    
    def test_prediction_to_pil_image(self):
        """Test converting prediction to PIL image."""
        # Create a test image and encode it
        img = Image.new('RGB', (200, 200), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        prediction = {
            "bytesBase64Encoded": encoded
        }
        
        result = prediction_to_pil_image(prediction)
        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)
    
    def test_prediction_to_pil_image_with_size(self):
        """Test converting prediction with size limit."""
        # Create a large test image
        img = Image.new('RGB', (1000, 1000), color='green')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        prediction = {
            "bytesBase64Encoded": encoded
        }
        
        result = prediction_to_pil_image(prediction, size=(100, 100))
        assert isinstance(result, Image.Image)
        assert result.size[0] <= 100
        assert result.size[1] <= 100
    
    def test_prediction_to_pil_image_missing_field(self):
        """Test error handling for missing field."""
        prediction = {"someOtherField": "value"}
        
        with pytest.raises(ValueError, match="bytesBase64Encoded"):
            prediction_to_pil_image(prediction)


class TestImagePreparation:
    """Test image preparation for API."""
    
    def test_prepare_image_for_api(self, tmp_path):
        """Test preparing image for API consumption."""
        # Create a test image file
        img = Image.new('RGB', (2000, 2000), color='red')
        img_path = tmp_path / "large_image.jpg"
        img.save(img_path)
        
        with patch('structlog.get_logger') as mock_logger:
            logger_instance = MagicMock()
            mock_logger.return_value = logger_instance
            
            encoded = prepare_image_for_api(img_path)
            
            assert isinstance(encoded, str)
            assert len(encoded) > 0
            
            # Verify image was resized
            decoded = decode_base64_to_image(encoded)
            assert decoded.size[0] <= 1024
            assert decoded.size[1] <= 1024
    
    def test_prepare_image_for_api_small_image(self, tmp_path):
        """Test preparing small image (no resize needed)."""
        # Create a small test image
        img = Image.new('RGB', (100, 100), color='blue')
        img_path = tmp_path / "small_image.jpg"
        img.save(img_path)
        
        encoded = prepare_image_for_api(img_path)
        
        # Verify image was not resized
        decoded = decode_base64_to_image(encoded)
        assert decoded.size == (100, 100)


class TestImageSaving:
    """Test image saving with metadata."""
    
    def test_save_image_with_metadata(self, tmp_path):
        """Test saving image with metadata."""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='green')
        output_path = tmp_path / "output" / "test.jpg"
        metadata = {
            "processor": "test",
            "timestamp": "2024-01-01T00:00:00",
            "score": 0.95
        }
        
        img_path, meta_path = save_image_with_metadata(
            img, output_path, metadata, include_timestamp=False
        )
        
        # Check image was saved
        assert img_path.exists()
        assert img_path.name == "test.jpg"
        
        # Check metadata was saved
        assert meta_path.exists()
        assert meta_path.name == "test.json"
        
        # Verify metadata content
        import json
        with open(meta_path) as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["processor"] == "test"
        assert saved_metadata["score"] == 0.95
    
    def test_save_image_with_timestamp(self, tmp_path):
        """Test saving image with timestamp in filename."""
        img = Image.new('RGB', (50, 50), color='yellow')
        output_path = tmp_path / "test.jpg"
        metadata = {"test": "data"}
        
        img_path, meta_path = save_image_with_metadata(
            img, output_path, metadata, include_timestamp=True
        )
        
        # Check that timestamp was added
        assert img_path.exists()
        assert "test_" in img_path.name
        assert img_path.name != "test.jpg"


class TestGCSOperations:
    """Test Google Cloud Storage operations."""
    
    @patch('google.cloud.storage.Client')
    def test_upload_to_gcs(self, mock_storage_client, tmp_path):
        """Test uploading file to GCS."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Mock GCS client
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        with patch('structlog.get_logger') as mock_logger:
            logger_instance = MagicMock()
            mock_logger.return_value = logger_instance
            
            result = upload_to_gcs(
                test_file,
                "test-bucket",
                "path/to/test.txt",
                mock_client
            )
            
            assert result == "gs://test-bucket/path/to/test.txt"
            mock_blob.upload_from_filename.assert_called_once_with(str(test_file))
    
    @patch('google.cloud.storage.Client')
    def test_download_from_gcs(self, mock_storage_client, tmp_path):
        """Test downloading file from GCS."""
        # Mock GCS client
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        local_path = tmp_path / "downloaded.txt"
        
        with patch('structlog.get_logger') as mock_logger:
            logger_instance = MagicMock()
            mock_logger.return_value = logger_instance
            
            download_from_gcs(
                "gs://test-bucket/path/to/file.txt",
                local_path,
                mock_client
            )
            
            mock_client.bucket.assert_called_once_with("test-bucket")
            mock_bucket.blob.assert_called_once_with("path/to/file.txt")
            mock_blob.download_to_filename.assert_called_once_with(str(local_path))
    
    @patch('google.cloud.storage.Client')
    def test_download_from_gcs_invalid_uri(self, mock_client, tmp_path):
        """Test downloading with invalid GCS URI."""
        local_path = tmp_path / "test.txt"
        
        with pytest.raises(ValueError, match="Invalid GCS URI"):
            download_from_gcs("not-a-gcs-uri", local_path)


class TestRetryFunctions:
    """Test retry with backoff functions."""
    
    def test_retry_with_backoff_success(self):
        """Test retry function with successful execution."""
        call_count = 0
        
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = retry_with_backoff(func, max_retries=5)
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_with_backoff_exhausted(self):
        """Test retry function when retries are exhausted."""
        def func():
            raise ValueError("Persistent error")
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(ValueError, match="Persistent error"):
                retry_with_backoff(func, max_retries=3)
    
    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_success(self):
        """Test async retry function with successful execution."""
        call_count = 0
        
        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "async success"
        
        with patch('asyncio.sleep'):  # Mock sleep to speed up test
            result = await async_retry_with_backoff(async_func, max_retries=3)
        
        assert result == "async success"
        assert call_count == 2


class TestProcessingStats:
    """Test processing statistics calculation."""
    
    def test_calculate_processing_stats(self):
        """Test calculating processing statistics."""
        start_time = 1000.0  # Use fixed times for predictable results
        end_time = 1001.0    # 1 second duration
        
        stats = calculate_processing_stats(
            start_time=start_time,
            end_time=end_time,
            total_items=100,
            successful_items=95,
            failed_items=5
        )
        
        assert stats['total_items'] == 100
        assert stats['successful_items'] == 95
        assert stats['failed_items'] == 5
        assert stats['success_rate'] == 95.0
        assert stats['duration_seconds'] == 1.0
        assert stats['items_per_second'] == 100.0
        assert stats['average_time_per_item'] == 0.01
    
    def test_calculate_processing_stats_zero_items(self):
        """Test calculating stats with zero items."""
        start_time = time.time()
        end_time = start_time + 1.0
        
        stats = calculate_processing_stats(
            start_time=start_time,
            end_time=end_time,
            total_items=0,
            successful_items=0,
            failed_items=0
        )
        
        assert stats['success_rate'] == 0.0
        assert stats['items_per_second'] == 0.0


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    @patch('structlog.get_logger')
    def test_progress_tracker_initialization(self, mock_logger):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(total_items=100)
        
        assert tracker.total_items == 100
        assert tracker.completed_items == 0
        assert tracker.failed_items == 0
    
    @patch('bulk_image_processor.utils.logger')
    def test_progress_tracker_update(self, mock_logger):
        """Test updating progress."""
        tracker = ProgressTracker(total_items=20)
        
        # Update exactly 10 items to trigger logging
        for i in range(10):
            tracker.update(success=True)
        
        assert tracker.completed_items == 10
        assert tracker.failed_items == 0
        
        # Verify logging was called at the 10th item
        assert mock_logger.info.call_count == 1
        
        # Update with failures
        for i in range(5):
            tracker.update(success=False)
        
        assert tracker.completed_items == 15
        assert tracker.failed_items == 5
    
    @patch('structlog.get_logger')
    def test_progress_tracker_final_stats(self, mock_logger):
        """Test getting final statistics."""
        tracker = ProgressTracker(total_items=50)
        
        # Simulate some processing
        for i in range(45):
            tracker.update(success=True)
        for i in range(5):
            tracker.update(success=False)
        
        stats = tracker.get_final_stats()
        
        assert stats['total_items'] == 50
        assert stats['successful_items'] == 45
        assert stats['failed_items'] == 5
        assert stats['success_rate'] == 90.0