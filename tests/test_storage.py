"""Tests for the storage manager module."""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from google.cloud.exceptions import NotFound

from bulk_image_processor.storage import StorageManager, create_storage_manager


@pytest.fixture
def mock_gcs_enabled_settings():
    """Settings with GCS enabled."""
    import os
    os.environ.update({
        'PROJECT_ID': 'test-project',
        'GOOGLE_CLOUD_STORAGE': 'test-bucket',
        'GEMINI_API_KEY': 'test-gemini-key',
        'ENABLE_GCS_UPLOAD': 'true'
    })
    
    try:
        from bulk_image_processor.config import Settings
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
            enable_gcs_upload=True,
            log_level="INFO",
            log_format="json",
        )
    finally:
        for key in ['PROJECT_ID', 'GOOGLE_CLOUD_STORAGE', 'GEMINI_API_KEY', 'ENABLE_GCS_UPLOAD']:
            os.environ.pop(key, None)

@pytest.fixture
def mock_gcs_disabled_settings():
    """Settings with GCS disabled."""
    import os
    os.environ.update({
        'PROJECT_ID': 'test-project',
        'GOOGLE_CLOUD_STORAGE': 'test-bucket',
        'GEMINI_API_KEY': 'test-gemini-key',
        'ENABLE_GCS_UPLOAD': 'false'
    })
    
    try:
        from bulk_image_processor.config import Settings
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
        for key in ['PROJECT_ID', 'GOOGLE_CLOUD_STORAGE', 'GEMINI_API_KEY', 'ENABLE_GCS_UPLOAD']:
            os.environ.pop(key, None)

@pytest.fixture
def mock_gcs_client():
    """Mock Google Cloud Storage client."""
    with patch('google.cloud.storage.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        yield mock_client, mock_bucket


class TestStorageManager:
    """Test StorageManager class."""
    
    def test_initialization_gcs_disabled(self, mock_gcs_disabled_settings):
        """Test StorageManager initialization with GCS disabled."""
        with patch('bulk_image_processor.storage.ensure_directory') as mock_ensure_dir:
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            assert storage_manager.settings == mock_gcs_disabled_settings
            assert storage_manager.gcs_client is None
            assert storage_manager.bucket is None
            
            # Should have created local directories
            assert mock_ensure_dir.call_count == 3
    
    def test_initialization_gcs_enabled(self, mock_gcs_enabled_settings, mock_gcs_client):
        """Test StorageManager initialization with GCS enabled."""
        mock_client, mock_bucket = mock_gcs_client
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            assert storage_manager.gcs_client == mock_client
            assert storage_manager.bucket == mock_bucket
            mock_bucket.exists.assert_called_once()
    
    def test_initialization_gcs_bucket_not_exists(self, mock_gcs_enabled_settings):
        """Test initialization when GCS bucket doesn't exist."""
        with patch('google.cloud.storage.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.exists.return_value = False
            mock_client.bucket.return_value = mock_bucket
            mock_client_class.return_value = mock_client
            
            with patch('bulk_image_processor.storage.ensure_directory'):
                with pytest.raises(ValueError, match="Bucket .* does not exist"):
                    StorageManager(mock_gcs_enabled_settings)
    
    def test_get_local_output_dir(self, mock_gcs_disabled_settings):
        """Test getting local output directory."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            result = storage_manager.get_local_output_dir("virtual_try_on")
            expected = mock_gcs_disabled_settings.storage.local_output_dir / "virtual_try_on"
            
            assert result == expected
    
    def test_upload_file_gcs_disabled(self, mock_gcs_disabled_settings, tmp_path):
        """Test upload file when GCS is disabled."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            test_file = tmp_path / "test.txt"
            test_file.write_text("test content")
            
            with pytest.raises(ValueError, match="GCS upload is disabled"):
                storage_manager.upload_file(test_file, "path/to/file.txt")
    
    def test_upload_file_success(self, mock_gcs_enabled_settings, mock_gcs_client, tmp_path):
        """Test successful file upload to GCS."""
        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            test_file = tmp_path / "test.txt"
            test_file.write_text("test content")
            
            result = storage_manager.upload_file(
                test_file, 
                "uploads/test.txt",
                metadata={"source": "test"}
            )
            
            expected_uri = f"gs://{mock_gcs_enabled_settings.google_cloud.storage_bucket}/uploads/test.txt"
            assert result == expected_uri
            
            mock_bucket.blob.assert_called_once_with("uploads/test.txt")
            mock_blob.upload_from_filename.assert_called_once_with(str(test_file))
            assert mock_blob.metadata == {"source": "test"}
    
    def test_upload_file_no_bucket(self, mock_gcs_enabled_settings, tmp_path):
        """Test upload file when bucket is not initialized."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager.__new__(StorageManager)
            storage_manager.settings = mock_gcs_enabled_settings
            storage_manager.bucket = None
            
            test_file = tmp_path / "test.txt"
            test_file.write_text("test content")
            
            with pytest.raises(ValueError, match="GCS bucket not initialized"):
                storage_manager.upload_file(test_file, "path/to/file.txt")
    
    def test_download_file_success(self, mock_gcs_enabled_settings, mock_gcs_client, tmp_path):
        """Test successful file download from GCS."""
        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        
        with patch('bulk_image_processor.storage.ensure_directory') as mock_ensure_dir:
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            local_path = tmp_path / "downloaded.txt"
            
            # Create the file after download to avoid stat() error
            def mock_download(file_path):
                local_path.write_text("test content")
            mock_blob.download_to_filename.side_effect = mock_download
            
            storage_manager.download_file("remote/file.txt", local_path)
            
            mock_bucket.blob.assert_called_with("remote/file.txt")
            mock_blob.exists.assert_called_once()
            mock_ensure_dir.assert_called()
            mock_blob.download_to_filename.assert_called_once_with(str(local_path))
    
    def test_download_file_not_found(self, mock_gcs_enabled_settings, mock_gcs_client, tmp_path):
        """Test download file when file doesn't exist."""
        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        mock_bucket.name = "test-bucket"
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            local_path = tmp_path / "downloaded.txt"
            
            with pytest.raises(NotFound, match="File not found"):
                storage_manager.download_file("remote/file.txt", local_path)
    
    def test_download_file_no_bucket(self, mock_gcs_enabled_settings, tmp_path):
        """Test download file when bucket is not initialized."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager.__new__(StorageManager)
            storage_manager.settings = mock_gcs_enabled_settings
            storage_manager.bucket = None
            
            local_path = tmp_path / "downloaded.txt"
            
            with pytest.raises(ValueError, match="GCS bucket not initialized"):
                storage_manager.download_file("remote/file.txt", local_path)
    
    def test_list_files_success(self, mock_gcs_enabled_settings, mock_gcs_client):
        """Test successful file listing from GCS."""
        mock_client, mock_bucket = mock_gcs_client
        
        # Mock blobs
        mock_blob1 = MagicMock()
        mock_blob1.name = "file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = "file2.txt"
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            result = storage_manager.list_files("uploads/", max_results=10)
            
            assert result == ["file1.txt", "file2.txt"]
            mock_bucket.list_blobs.assert_called_once_with(
                prefix="uploads/",
                max_results=10
            )
    
    def test_list_files_no_bucket(self, mock_gcs_enabled_settings):
        """Test list files when bucket is not initialized."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager.__new__(StorageManager)
            storage_manager.settings = mock_gcs_enabled_settings
            storage_manager.bucket = None
            
            with pytest.raises(ValueError, match="GCS bucket not initialized"):
                storage_manager.list_files("uploads/")
    
    def test_save_processing_log_gcs_disabled(self, mock_gcs_disabled_settings, tmp_path):
        """Test saving processing log with GCS disabled."""
        with patch('bulk_image_processor.storage.ensure_directory'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump, \
             patch('time.strftime', return_value='20240101_120000'):
            
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            log_data = {"test": "data", "timestamp": "2024-01-01"}
            result = storage_manager.save_processing_log(
                log_data, "virtual_try_on", "record_001"
            )
            
            expected_path = Path("./logs/virtual_try_on_record_001_20240101_120000.json")
            assert result == expected_path
            
            mock_file.assert_called_once_with(expected_path, 'w')
            mock_json_dump.assert_called_once_with(log_data, mock_file(), indent=2, default=str)
    
    def test_save_processing_log_gcs_enabled(self, mock_gcs_enabled_settings, mock_gcs_client):
        """Test saving processing log with GCS enabled."""
        mock_client, mock_bucket = mock_gcs_client
        
        with patch('bulk_image_processor.storage.ensure_directory'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('time.strftime', return_value='20240101_120000'):
            
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            # Mock upload_file method
            storage_manager.upload_file = MagicMock()
            
            log_data = {"test": "data"}
            result = storage_manager.save_processing_log(
                log_data, "virtual_try_on", "record_001"
            )
            
            # Should have called upload_file
            storage_manager.upload_file.assert_called_once()
    
    def test_create_processing_summary_gcs_disabled(self, mock_gcs_disabled_settings):
        """Test creating processing summary with GCS disabled."""
        with patch('bulk_image_processor.storage.ensure_directory'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump, \
             patch('time.strftime', return_value='20240101_120000'):
            
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            summary_data = {"total_processed": 10, "successful": 8}
            result = storage_manager.create_processing_summary(
                summary_data, "bulk_processing"
            )
            
            expected_path = Path("./logs/bulk_processing_summary_20240101_120000.json")
            assert result == expected_path
            
            mock_file.assert_called_once_with(expected_path, 'w')
            mock_json_dump.assert_called_once_with(summary_data, mock_file(), indent=2, default=str)
    
    def test_create_processing_summary_gcs_enabled(self, mock_gcs_enabled_settings, mock_gcs_client):
        """Test creating processing summary with GCS enabled."""
        mock_client, mock_bucket = mock_gcs_client
        
        with patch('bulk_image_processor.storage.ensure_directory'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('time.strftime', return_value='20240101_120000'):
            
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            # Mock upload_file method
            storage_manager.upload_file = MagicMock()
            
            summary_data = {"total_processed": 10}
            result = storage_manager.create_processing_summary(
                summary_data, "bulk_processing"
            )
            
            # Should have called upload_file
            storage_manager.upload_file.assert_called_once()
    
    def test_cleanup_old_files_success(self, mock_gcs_disabled_settings, tmp_path):
        """Test successful cleanup of old files."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            # Create test files with different ages
            old_file = tmp_path / "old_file.txt"
            new_file = tmp_path / "new_file.txt"
            old_file.write_text("old content")
            new_file.write_text("new content")
            
            # Mock time to control file age calculation
            current_time = time.time()
            old_time = current_time - (8 * 24 * 60 * 60)  # 8 days old 
            new_time = current_time - (1 * 24 * 60 * 60)  # 1 day old
            
            # Mock os.stat to control file modification times
            original_stat = old_file.stat
            def mock_stat_old():
                stat_result = original_stat()
                stat_result.st_mtime = old_time
                return stat_result
            
            def mock_stat_new():
                stat_result = original_stat()
                stat_result.st_mtime = new_time 
                return stat_result
            
            with patch('time.time', return_value=current_time):
                # Manually set the modification times and run the cleanup
                import os
                os.utime(old_file, (old_time, old_time))
                os.utime(new_file, (new_time, new_time))
                
                result = storage_manager.cleanup_old_files(tmp_path, max_age_days=7)
                
                # Should have deleted the old file
                assert result == 1
                assert not old_file.exists()
                assert new_file.exists()
    
    def test_cleanup_old_files_directory_not_exists(self, mock_gcs_disabled_settings, tmp_path):
        """Test cleanup when directory doesn't exist."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            nonexistent_dir = tmp_path / "nonexistent"
            result = storage_manager.cleanup_old_files(nonexistent_dir)
            
            assert result == 0
    
    def test_get_storage_stats_gcs_disabled(self, mock_gcs_disabled_settings):
        """Test getting storage stats with GCS disabled."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            stats = storage_manager.get_storage_stats()
            
            assert 'local_storage' in stats
            assert 'gcs_storage' in stats
            assert stats['gcs_storage']['enabled'] is False
            assert stats['gcs_storage']['connected'] is False
    
    def test_get_storage_stats_gcs_enabled(self, mock_gcs_enabled_settings, mock_gcs_client):
        """Test getting storage stats with GCS enabled."""
        mock_client, mock_bucket = mock_gcs_client
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            stats = storage_manager.get_storage_stats()
            
            assert stats['gcs_storage']['enabled'] is True
            assert stats['gcs_storage']['connected'] is True
            assert stats['gcs_storage']['bucket'] == mock_gcs_enabled_settings.google_cloud.storage_bucket
    
    def test_get_storage_stats_with_local_size(self, mock_gcs_disabled_settings, tmp_path):
        """Test getting storage stats including local directory size."""
        mock_gcs_disabled_settings.storage.local_output_dir = tmp_path
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            # Create test files
            (tmp_path / "file1.txt").write_text("content1")
            (tmp_path / "file2.txt").write_text("content2")
            
            stats = storage_manager.get_storage_stats()
            
            assert 'size_bytes' in stats['local_storage']
            assert stats['local_storage']['size_bytes'] > 0
    
    def test_exception_handling_upload(self, mock_gcs_enabled_settings, mock_gcs_client, tmp_path):
        """Test exception handling during upload."""
        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_blob.upload_from_filename.side_effect = Exception("Upload failed")
        mock_bucket.blob.return_value = mock_blob
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            test_file = tmp_path / "test.txt"
            test_file.write_text("test content")
            
            with pytest.raises(Exception, match="Upload failed"):
                storage_manager.upload_file(test_file, "path/to/file.txt")
    
    def test_exception_handling_download(self, mock_gcs_enabled_settings, mock_gcs_client, tmp_path):
        """Test exception handling during download."""
        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_to_filename.side_effect = Exception("Download failed")
        mock_bucket.blob.return_value = mock_blob
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            local_path = tmp_path / "downloaded.txt"
            
            with pytest.raises(Exception, match="Download failed"):
                storage_manager.download_file("remote/file.txt", local_path)
    
    def test_exception_handling_list_files(self, mock_gcs_enabled_settings, mock_gcs_client):
        """Test exception handling during file listing."""
        mock_client, mock_bucket = mock_gcs_client
        mock_bucket.list_blobs.side_effect = Exception("List failed")
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            with pytest.raises(Exception, match="List failed"):
                storage_manager.list_files("uploads/")


class TestCreateStorageManager:
    """Test the create_storage_manager factory function."""
    
    def test_create_storage_manager(self, mock_gcs_disabled_settings):
        """Test creating a storage manager instance."""
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = create_storage_manager(mock_gcs_disabled_settings)
            
            assert isinstance(storage_manager, StorageManager)
            assert storage_manager.settings == mock_gcs_disabled_settings


class TestStorageManagerErrorScenarios:
    """Test error scenarios and edge cases."""
    
    def test_gcs_initialization_failure(self, mock_gcs_enabled_settings):
        """Test GCS initialization failure."""
        with patch('google.cloud.storage.Client', side_effect=Exception("GCS init failed")), \
             patch('bulk_image_processor.storage.ensure_directory'):
            
            with pytest.raises(Exception, match="GCS init failed"):
                StorageManager(mock_gcs_enabled_settings)
    
    def test_upload_file_warning_on_gcs_failure(self, mock_gcs_enabled_settings, mock_gcs_client):
        """Test that log upload failures are handled gracefully."""
        mock_client, mock_bucket = mock_gcs_client
        
        with patch('bulk_image_processor.storage.ensure_directory'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('time.strftime', return_value='20240101_120000'):
            
            storage_manager = StorageManager(mock_gcs_enabled_settings)
            
            # Mock upload_file to raise exception
            storage_manager.upload_file = MagicMock(side_effect=Exception("Upload failed"))
            
            # Should not raise exception, just log warning
            result = storage_manager.save_processing_log({"test": "data"}, "test", "001")
            
            assert result is not None
    
    def test_storage_stats_size_calculation_error(self, mock_gcs_disabled_settings, tmp_path):
        """Test error handling in storage stats size calculation."""
        mock_gcs_disabled_settings.storage.local_output_dir = tmp_path
        
        with patch('bulk_image_processor.storage.ensure_directory'):
            storage_manager = StorageManager(mock_gcs_disabled_settings)
            
            # Mock Path.rglob to raise exception globally
            with patch('pathlib.Path.rglob', side_effect=Exception("Access denied")):
                stats = storage_manager.get_storage_stats()
                
                # Should not crash, just not include size
                assert 'size_bytes' not in stats['local_storage']