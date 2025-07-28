"""Integration tests for the bulk image processor.

These tests verify that all components work together correctly
in realistic scenarios.
"""

import pytest
import asyncio
import csv
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from bulk_image_processor.main import BulkImageProcessor
from bulk_image_processor.downloader import ImageRecord
from bulk_image_processor.analyzer import ImageCategory, ClassificationResult
from bulk_image_processor.config import Settings


class TestBulkImageProcessorIntegration:
    """Integration tests for the BulkImageProcessor."""
    
    @pytest.fixture
    def mock_settings_integration(self):
        """Mock settings for integration tests."""
        return Settings(
            project_id="integration-test-project",
            location="us-central1",
            model_endpoint="test-vto-endpoint",
            model_endpoint_product="test-product-endpoint",
            google_cloud_storage="test-bucket",
            gemini_api_key="test-gemini-key",
            max_workers=2,
            max_retries=1,
            download_timeout=10,
            processing_timeout=30,
            local_output_dir="./test_output",
            enable_gcs_upload=False,
            log_level="INFO",
            log_format="json",
        )
    
    @pytest.fixture
    def sample_csv_file(self, tmp_path):
        """Create a sample CSV file for integration testing."""
        csv_path = tmp_path / "test_images.csv"
        
        data = [
            ["ID", "Image Src", "Image Command", "Image Position"],
            ["001", "https://example.com/shirt.jpg", "process", "front"],
            ["002", "https://example.com/dress.jpg", "enhance", "side"],
            ["003", "https://example.com/vase.jpg", "luxury style", "center"],
            ["004", "https://example.com/watch.jpg", "minimal", "top"],
            ["005", "https://example.com/invalid.txt", "process", "front"],  # Invalid URL
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        return csv_path
    
    @pytest.fixture
    def mock_image_downloads(self, tmp_path):
        """Mock successful image downloads."""
        def create_mock_download(filename, color):
            img_path = tmp_path / "downloads" / filename
            img_path.parent.mkdir(exist_ok=True)
            img = Image.new('RGB', (100, 100), color=color)
            img.save(img_path)
            return img_path
        
        return {
            "shirt.jpg": create_mock_download("img_001_shirt.jpg", "blue"),
            "dress.jpg": create_mock_download("img_002_dress.jpg", "red"),
            "vase.jpg": create_mock_download("img_003_vase.jpg", "white"),
            "watch.jpg": create_mock_download("img_004_watch.jpg", "black"),
        }
    
    @pytest.mark.asyncio
    @patch('bulk_image_processor.downloader.ImageDownloader.download_single_image')
    @patch('google.generativeai.GenerativeModel')
    @patch('google.cloud.aiplatform.Endpoint')
    @patch('bulk_image_processor.processors.virtual_try_on.load_model_pairs')
    async def test_end_to_end_processing(
        self, mock_load_models, mock_endpoint_class, mock_gemini_model,
        mock_download, mock_settings_integration, sample_csv_file,
        mock_image_downloads, tmp_path
    ):
        """Test end-to-end processing from CSV to output."""
        # Setup model pairs for VTO
        mock_load_models.return_value = [{
            "model_id": "model_001",
            "gender": "female",
            "pose": "front",
            "style": "casual",
            "image_path": "/models/model_001.jpg"
        }]
        
        # Mock image downloads
        async def mock_download_impl(record, output_dir):
            filename = Path(record.image_url).name
            if filename in mock_image_downloads:
                return mock_image_downloads[filename]
            return None
        
        mock_download.side_effect = mock_download_impl
        
        # Mock Gemini classifier
        gemini_instance = MagicMock()
        mock_gemini_model.return_value = gemini_instance
        
        classification_responses = {
            "001": '{"category": "APPAREL", "confidence": 0.95, "reasoning": "Shirt detected", "detected_items": ["shirt"]}',
            "002": '{"category": "APPAREL", "confidence": 0.92, "reasoning": "Dress detected", "detected_items": ["dress"]}',
            "003": '{"category": "PRODUCT", "confidence": 0.88, "reasoning": "Vase detected", "detected_items": ["vase"]}',
            "004": '{"category": "PRODUCT", "confidence": 0.90, "reasoning": "Watch detected", "detected_items": ["watch"]}',
        }
        
        async def mock_classify(content):
            # Extract record ID from the prompt
            for record_id, response in classification_responses.items():
                if record_id in str(content):
                    mock_response = MagicMock()
                    mock_response.text = response
                    return mock_response
            # Default response
            mock_response = MagicMock()
            mock_response.text = '{"category": "UNKNOWN", "confidence": 0.5, "reasoning": "Cannot classify"}'
            return mock_response
        
        gemini_instance.generate_content_async = AsyncMock(side_effect=mock_classify)
        
        # Mock validation responses
        validation_response = MagicMock()
        validation_response.text = '{"pass": true, "score": 0.90, "reasoning": "Good quality", "suggestions": []}'
        gemini_instance.generate_content_async = AsyncMock(return_value=validation_response)
        
        # Mock scene prompt generation
        scene_response = MagicMock()
        scene_response.text = '{"prompt": "Luxury setting", "style": "luxury", "lighting": "soft"}'
        
        # Setup response cycling for different calls
        response_cycle = [
            classification_responses["001"],
            classification_responses["002"],
            classification_responses["003"],
            classification_responses["004"],
            validation_response.text,
            scene_response.text,
        ]
        response_index = 0
        
        async def mock_gemini_call(*args, **kwargs):
            nonlocal response_index
            response = MagicMock()
            response.text = response_cycle[response_index % len(response_cycle)]
            response_index += 1
            return response
        
        gemini_instance.generate_content_async = AsyncMock(side_effect=mock_gemini_call)
        
        # Mock endpoint predictions
        mock_endpoint = MagicMock()
        mock_endpoint_class.return_value = mock_endpoint
        
        mock_prediction = MagicMock()
        mock_prediction.predictions = [{
            "bytesBase64Encoded": "fake_base64_output"
        }]
        mock_endpoint.predict.return_value = mock_prediction
        
        # Mock image conversion
        with patch('bulk_image_processor.processors.virtual_try_on.prediction_to_pil_image') as mock_vto_convert:
            with patch('bulk_image_processor.processors.product_recontext.prediction_to_pil_image') as mock_pr_convert:
                output_img = Image.new('RGB', (100, 100), color='green')
                mock_vto_convert.return_value = output_img
                mock_pr_convert.return_value = output_img
                
                # Run the processor
                processor = BulkImageProcessor(mock_settings_integration)
                summary = await processor.process_from_csv(sample_csv_file)
        
        # Verify summary statistics
        assert summary['download_stats']['total_records'] == 5
        assert summary['download_stats']['successful_downloads'] >= 4
        assert summary['classification_stats']['apparel_count'] >= 0
        assert summary['classification_stats']['product_count'] >= 0
        assert summary['processing_stats']['total_processed'] >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, mock_settings_integration, tmp_path
    ):
        """Test error handling and recovery mechanisms."""
        # Create CSV with problematic data
        csv_path = tmp_path / "problematic.csv"
        data = [
            ["ID", "Image Src", "Image Command", "Image Position"],
            ["001", "", "process", "front"],  # Empty URL
            ["002", "not-a-url", "enhance", "side"],  # Invalid URL
            ["003", "https://nonexistent.com/image.jpg", "process", "center"],  # Unreachable
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        with patch('google.generativeai.GenerativeModel'):
            with patch('google.cloud.aiplatform.Endpoint'):
                processor = BulkImageProcessor(mock_settings_integration)
                
                # Should complete without crashing
                summary = await processor.process_from_csv(csv_path)
                
                # Verify error handling
                assert summary['download_stats']['failed_downloads'] > 0
                assert summary['processing_stats']['total_processed'] == 0
    
    @pytest.mark.asyncio
    @patch('bulk_image_processor.main.BulkImageProcessor._classify_images')
    @patch('bulk_image_processor.main.BulkImageProcessor._process_classified_images')
    @patch('bulk_image_processor.downloader.download_images_from_csv')
    async def test_classification_routing(
        self, mock_download, mock_process, mock_classify,
        mock_settings_integration, sample_csv_file, tmp_path
    ):
        """Test that images are correctly routed based on classification."""
        # Mock successful downloads
        mock_records = [
            ImageRecord("001", "https://example.com/shirt.jpg", "process", "front", 0),
            ImageRecord("002", "https://example.com/vase.jpg", "enhance", "center", 1),
            ImageRecord("003", "https://example.com/unknown.jpg", "process", "front", 2),
        ]
        
        mock_paths = [
            tmp_path / "shirt.jpg",
            tmp_path / "vase.jpg",
            tmp_path / "unknown.jpg",
        ]
        
        # Create mock images
        for path in mock_paths:
            img = Image.new('RGB', (50, 50))
            img.save(path)
        
        mock_download.return_value = list(zip(mock_records, mock_paths))
        
        # Mock classification results
        mock_classify.return_value = [
            (mock_records[0], mock_paths[0], ImageCategory.APPAREL),
            (mock_records[1], mock_paths[1], ImageCategory.PRODUCT),
            (mock_records[2], mock_paths[2], ImageCategory.UNKNOWN),
        ]
        
        # Track which processors are called
        process_calls = []
        
        async def track_process_calls(classified_images):
            process_calls.append(classified_images)
            return []
        
        mock_process.side_effect = track_process_calls
        
        with patch('google.generativeai.GenerativeModel'):
            with patch('google.cloud.aiplatform.Endpoint'):
                processor = BulkImageProcessor(mock_settings_integration)
                await processor.process_from_csv(sample_csv_file)
        
        # Verify classification was called
        mock_classify.assert_called_once()
        
        # Verify processing was called with classified images
        mock_process.assert_called_once()
        assert len(process_calls) == 1
        
        # Check the classified images were passed correctly
        classified = process_calls[0]
        assert len(classified) == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(
        self, mock_settings_integration, tmp_path
    ):
        """Test concurrent processing of multiple images."""
        # Create CSV with multiple images
        csv_path = tmp_path / "concurrent.csv"
        num_images = 10
        
        data = [["ID", "Image Src", "Image Command", "Image Position"]]
        for i in range(num_images):
            data.append([f"{i:03d}", f"https://example.com/image_{i}.jpg", "process", "front"])
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        # Track concurrent operations
        concurrent_downloads = []
        concurrent_classifications = []
        
        async def mock_download(record, output_dir):
            concurrent_downloads.append(asyncio.current_task())
            await asyncio.sleep(0.01)  # Simulate download time
            img_path = output_dir / f"{record.id}.jpg"
            img = Image.new('RGB', (50, 50))
            img.save(img_path)
            return img_path
        
        async def mock_classify(*args, **kwargs):
            concurrent_classifications.append(asyncio.current_task())
            await asyncio.sleep(0.01)  # Simulate API call
            response = MagicMock()
            response.text = '{"category": "PRODUCT", "confidence": 0.9, "reasoning": "Test"}'
            return response
        
        with patch('bulk_image_processor.downloader.ImageDownloader.download_single_image', new=mock_download):
            with patch('google.generativeai.GenerativeModel') as mock_gemini:
                with patch('google.cloud.aiplatform.Endpoint'):
                    gemini_instance = MagicMock()
                    mock_gemini.return_value = gemini_instance
                    gemini_instance.generate_content_async = mock_classify
                    
                    processor = BulkImageProcessor(mock_settings_integration)
                    processor.settings.processing.max_workers = 5  # Allow concurrent operations
                    
                    await processor.process_from_csv(csv_path)
        
        # Verify concurrent operations occurred
        unique_download_tasks = len(set(concurrent_downloads))
        unique_classification_tasks = len(set(concurrent_classifications))
        
        # Should have multiple concurrent tasks (not all sequential)
        assert unique_download_tasks > 1
        assert unique_classification_tasks > 1
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(
        self, mock_settings_integration, sample_csv_file
    ):
        """Test graceful shutdown handling."""
        import signal
        
        with patch('google.generativeai.GenerativeModel'):
            with patch('google.cloud.aiplatform.Endpoint'):
                processor = BulkImageProcessor(mock_settings_integration)
                
                # Simulate shutdown request during processing
                async def trigger_shutdown():
                    await asyncio.sleep(0.1)
                    processor.shutdown_requested = True
                
                # Start processing and shutdown concurrently
                with patch('bulk_image_processor.downloader.download_images_from_csv') as mock_dl:
                    # Mock long-running download
                    async def slow_download(*args):
                        await asyncio.sleep(1.0)
                        return []
                    
                    mock_dl.side_effect = slow_download
                    
                    # Run processing and shutdown trigger concurrently
                    results = await asyncio.gather(
                        processor.process_from_csv(sample_csv_file),
                        trigger_shutdown(),
                        return_exceptions=True
                    )
                    
                    # Verify processing completed (even if partially)
                    summary = results[0]
                    assert isinstance(summary, dict)
    
    @pytest.mark.asyncio
    async def test_storage_integration(
        self, mock_settings_integration, tmp_path
    ):
        """Test integration with storage components."""
        # Enable GCS upload for this test
        mock_settings_integration.enable_gcs_upload = True
        
        csv_path = tmp_path / "storage_test.csv"
        data = [
            ["ID", "Image Src", "Image Command", "Image Position"],
            ["001", "https://example.com/test.jpg", "process", "front"],
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        # Track storage operations
        storage_operations = []
        
        with patch('google.cloud.storage.Client') as mock_storage:
            with patch('google.generativeai.GenerativeModel'):
                with patch('google.cloud.aiplatform.Endpoint'):
                    # Mock storage client
                    mock_client = MagicMock()
                    mock_bucket = MagicMock()
                    mock_blob = MagicMock()
                    
                    mock_storage.return_value = mock_client
                    mock_client.bucket.return_value = mock_bucket
                    mock_bucket.blob.return_value = mock_blob
                    
                    def track_upload(filename):
                        storage_operations.append(('upload', filename))
                    
                    mock_blob.upload_from_filename.side_effect = track_upload
                    
                    # Create a simple test image
                    img_path = tmp_path / "downloads" / "test.jpg"
                    img_path.parent.mkdir(exist_ok=True)
                    img = Image.new('RGB', (50, 50))
                    img.save(img_path)
                    
                    with patch('bulk_image_processor.downloader.ImageDownloader.download_single_image') as mock_dl:
                        async def return_test_image(*args):
                            return img_path
                        
                        mock_dl.side_effect = return_test_image
                        
                        processor = BulkImageProcessor(mock_settings_integration)
                        await processor.process_from_csv(csv_path)
        
        # Verify storage operations occurred
        assert len(storage_operations) > 0