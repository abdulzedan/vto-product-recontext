"""Tests for the main bulk image processor orchestrator."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from PIL import Image

from bulk_image_processor.main import BulkImageProcessor, main
from bulk_image_processor.analyzer import ImageCategory, ClassificationResult
from bulk_image_processor.downloader import ImageRecord
from bulk_image_processor.processors.base import ProcessingResult


class TestBulkImageProcessor:
    """Test BulkImageProcessor class."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock all external components."""
        with patch('bulk_image_processor.main.GeminiAnalyzer') as mock_analyzer_class, \
             patch('bulk_image_processor.main.VirtualTryOnProcessor') as mock_vto_class, \
             patch('bulk_image_processor.main.ProductRecontextProcessor') as mock_product_class, \
             patch('bulk_image_processor.main.StorageManager') as mock_storage_class, \
             patch('bulk_image_processor.main.setup_logging'):
            
            mock_analyzer = MagicMock()
            mock_vto = MagicMock() 
            mock_product = MagicMock()
            mock_storage = MagicMock()
            
            mock_analyzer_class.return_value = mock_analyzer
            mock_vto_class.return_value = mock_vto
            mock_product_class.return_value = mock_product  
            mock_storage_class.return_value = mock_storage
            
            yield {
                'analyzer': mock_analyzer,
                'vto_processor': mock_vto,
                'product_processor': mock_product,
                'storage_manager': mock_storage
            }
    
    def test_initialization(self, mock_settings, mock_components):
        """Test BulkImageProcessor initialization."""
        processor = BulkImageProcessor(mock_settings)
        
        assert processor.settings == mock_settings
        assert processor.analyzer is not None
        assert processor.vto_processor is not None
        assert processor.product_processor is not None
        assert processor.storage_manager is not None
        assert processor.shutdown_requested is False
    
    def test_initialization_with_default_settings(self, mock_components):
        """Test initialization with default settings."""
        with patch('bulk_image_processor.main.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_get_settings.return_value = mock_settings
            
            processor = BulkImageProcessor()
            
            assert processor.settings == mock_settings
            mock_get_settings.assert_called_once()
    
    def test_setup_signal_handlers(self, mock_settings, mock_components):
        """Test signal handler setup."""
        with patch('signal.signal') as mock_signal:
            processor = BulkImageProcessor(mock_settings)
            
            # Verify signal handlers were set up
            assert mock_signal.call_count == 2
            # Just check that signal.signal was called with signal numbers
            call_args = [call[0][0] for call in mock_signal.call_args_list]
            assert len(call_args) == 2  # Should have 2 signal handlers
    
    @pytest.mark.asyncio
    async def test_classify_single_image_success(self, mock_settings, mock_components):
        """Test successful single image classification."""
        processor = BulkImageProcessor(mock_settings)
        
        # Mock analyzer response
        classification_result = ClassificationResult(
            category=ImageCategory.APPAREL,
            confidence=0.9,
            reasoning="Detected clothing item",
            detected_items=["shirt"],
            metadata={"style": "casual"}
        )
        mock_components['analyzer'].classify_image = AsyncMock(return_value=classification_result)
        
        record = ImageRecord(
            id="test_001", 
            image_url="https://example.com/shirt.jpg",
            image_command="process", 
            image_position="front",
            row_index=0
        )
        image_path = Path("/fake/path.jpg")
        
        result = await processor._classify_single_image(record, image_path)
        
        assert result == ImageCategory.APPAREL
        mock_components['analyzer'].classify_image.assert_called_once_with(
            image_path,
            additional_context="Command: process, Position: front"
        )
    
    @pytest.mark.asyncio
    async def test_classify_single_image_failure(self, mock_settings, mock_components):
        """Test image classification failure."""
        processor = BulkImageProcessor(mock_settings)
        
        # Mock analyzer to raise exception
        mock_components['analyzer'].classify_image = AsyncMock(side_effect=Exception("Classification failed"))
        
        record = ImageRecord(
            id="test_001", 
            image_url="https://example.com/shirt.jpg",
            image_command="process", 
            image_position="front",
            row_index=0
        )
        image_path = Path("/fake/path.jpg")
        
        result = await processor._classify_single_image(record, image_path)
        
        assert result == ImageCategory.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_classify_images_parallel(self, mock_settings, mock_components):
        """Test parallel image classification."""
        processor = BulkImageProcessor(mock_settings)
        mock_settings.processing.max_workers = 2
        
        # Create test data
        records_and_paths = [
            (ImageRecord(id="001", image_url="https://example.com/image1.jpg", image_command="cmd1", image_position="pos1", row_index=0), Path("/path1.jpg")),
            (ImageRecord(id="002", image_url="https://example.com/image2.jpg", image_command="cmd2", image_position="pos2", row_index=1), Path("/path2.jpg"))
        ]
        
        # Mock classify_single_image to return different categories
        processor._classify_single_image = AsyncMock(side_effect=[ImageCategory.APPAREL, ImageCategory.PRODUCT])
        
        with patch('bulk_image_processor.main.ProgressTracker') as mock_progress:
            mock_tracker = MagicMock()
            mock_progress.return_value = mock_tracker
            
            result = await processor._classify_images(records_and_paths)
        
        assert len(result) == 2
        assert result[0][2] == ImageCategory.APPAREL
        assert result[1][2] == ImageCategory.PRODUCT
        
        # Verify progress tracking
        mock_progress.assert_called_once_with(2)
        assert mock_tracker.update.call_count == 2
        mock_tracker.log_progress.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_classified_images(self, mock_settings, mock_components):
        """Test processing of classified images."""
        processor = BulkImageProcessor(mock_settings)
        
        # Create classified images data
        classified_images = [
            (ImageRecord(id="001", image_url="https://example.com/apparel.jpg", image_command="cmd1", image_position="pos1", row_index=0), 
             Path("/apparel.jpg"), ImageCategory.APPAREL),
            (ImageRecord(id="002", image_url="https://example.com/product.jpg", image_command="cmd2", image_position="pos2", row_index=1), 
             Path("/product.jpg"), ImageCategory.PRODUCT),
            (ImageRecord(id="003", image_url="https://example.com/unknown.jpg", image_command="cmd3", image_position="pos3", row_index=2), 
             Path("/unknown.jpg"), ImageCategory.UNKNOWN)
        ]
        
        # Mock processor methods - each call should return results for its specific category
        processor._process_with_processor = AsyncMock(side_effect=[
            [ProcessingResult(record_id="001", success=True)],  # apparel results
            [ProcessingResult(record_id="002", success=True)],  # product results  
            [ProcessingResult(record_id="003", success=True)]   # unknown results
        ])
        
        result = await processor._process_classified_images(classified_images)
        
        assert len(result) == 3
        # Should have called _process_with_processor 3 times (apparel, product, unknown)
        assert processor._process_with_processor.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_with_processor(self, mock_settings, mock_components):
        """Test processing with a specific processor."""
        processor = BulkImageProcessor(mock_settings)
        mock_settings.processing.max_workers = 2
        
        # Create test images data
        images = [
            (ImageRecord(id="001", image_url="https://example.com/image1.jpg", image_command="cmd1", image_position="pos1", row_index=0), Path("/image1.jpg")),
            (ImageRecord(id="002", image_url="https://example.com/image2.jpg", image_command="cmd2", image_position="pos2", row_index=1), Path("/image2.jpg"))
        ]
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process_with_retry = AsyncMock(side_effect=[
            ProcessingResult(record_id="001", success=True),
            ProcessingResult(record_id="002", success=True)
        ])
        
        # Mock storage manager
        mock_components['storage_manager'].get_local_output_dir.return_value = Path("/output")
        
        with patch('bulk_image_processor.main.ProgressTracker') as mock_progress:
            mock_tracker = MagicMock()
            mock_progress.return_value = mock_tracker
            
            result = await processor._process_with_processor(images, mock_processor, "test_processor")
        
        assert len(result) == 2
        assert all(r.success for r in result)
        
        # Verify processor was called
        assert mock_processor.process_with_retry.call_count == 2
        mock_processor.log_processing_summary.assert_called_once()
        
        # Verify progress tracking
        mock_progress.assert_called_once_with(2)
        assert mock_tracker.update.call_count == 2
        mock_tracker.log_progress.assert_called_once()
    
    def test_create_summary(self, mock_settings, mock_components):
        """Test creation of processing summary."""
        processor = BulkImageProcessor(mock_settings)
        
        # Mock processor stats
        mock_components['vto_processor'].get_stats.return_value = {"total": 5, "success": 4}
        mock_components['product_processor'].get_stats.return_value = {"total": 3, "success": 3}
        mock_components['storage_manager'].get_storage_stats.return_value = {"uploads": 7}
        
        # Create test data
        download_results = [
            (ImageRecord(id="001", image_url="https://example.com/image1.jpg", image_command="cmd1", image_position="pos1", row_index=0), Path("/path1.jpg")),
            (ImageRecord(id="002", image_url="https://example.com/image2.jpg", image_command="cmd2", image_position="pos2", row_index=1), None)  # Failed download
        ]
        
        classified_images = [
            (ImageRecord(id="001", image_url="https://example.com/image1.jpg", image_command="cmd1", image_position="pos1", row_index=0), 
             Path("/path1.jpg"), ImageCategory.APPAREL)
        ]
        
        processing_results = [
            ProcessingResult(record_id="001", success=True, quality_score=0.9),
            ProcessingResult(record_id="002", success=False)
        ]
        
        import time
        start_time = time.time() - 100  # 100 seconds ago
        
        summary = processor._create_summary(start_time, download_results, classified_images, processing_results)
        
        # Verify summary structure
        assert 'timestamp' in summary
        assert 'total_processing_time' in summary
        assert summary['download_stats']['total_records'] == 2
        assert summary['download_stats']['successful_downloads'] == 1
        assert summary['download_stats']['failed_downloads'] == 1
        assert summary['classification_stats']['total_classified'] == 1
        assert summary['classification_stats']['apparel_count'] == 1
        assert summary['processing_stats']['total_processed'] == 2
        assert summary['processing_stats']['successful_processed'] == 1
        assert summary['quality_stats']['average_quality_score'] == 0.9
    
    def test_get_system_status(self, mock_settings, mock_components):
        """Test getting system status."""
        processor = BulkImageProcessor(mock_settings)
        
        # Mock stats
        mock_components['vto_processor'].get_stats.return_value = {"vto_stat": "value"}
        mock_components['product_processor'].get_stats.return_value = {"product_stat": "value"}
        mock_components['storage_manager'].get_storage_stats.return_value = {"storage_stat": "value"}
        
        status = processor.get_system_status()
        
        assert status['initialized'] is True
        assert 'settings' in status
        assert 'processors' in status
        assert 'storage' in status
        assert status['processors']['vto_stats'] == {"vto_stat": "value"}
        assert status['processors']['product_stats'] == {"product_stat": "value"}
        assert status['storage'] == {"storage_stat": "value"}
    
    def test_get_system_status_uninitialized(self, mock_settings):
        """Test system status when components are not initialized."""
        with patch('bulk_image_processor.main.setup_logging'), \
             patch('bulk_image_processor.main.GeminiAnalyzer', side_effect=Exception("Init failed")):
            
            try:
                processor = BulkImageProcessor(mock_settings)
            except Exception:
                # Create a processor with None components to test uninitialized state
                processor = BulkImageProcessor.__new__(BulkImageProcessor)
                processor.settings = mock_settings
                processor.analyzer = None
                processor.vto_processor = None
                processor.product_processor = None
                processor.storage_manager = None
                
                status = processor.get_system_status()
                
                assert status['initialized'] is False
                assert status['processors']['vto_stats'] is None
                assert status['processors']['product_stats'] is None
                assert status['storage'] is None


class TestMainFunction:
    """Test the main entry point function."""
    
    @pytest.mark.asyncio
    async def test_main_with_csv_path(self, tmp_path):
        """Test main function with provided CSV path."""
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("ID,Image Src,Image Command,Image Position\n001,url1,cmd1,pos1")
        
        with patch('bulk_image_processor.main.BulkImageProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_from_csv = AsyncMock(return_value={"summary": "test"})
            
            await main(csv_path)
            
            mock_processor_class.assert_called_once()
            mock_processor.process_from_csv.assert_called_once_with(csv_path)
    
    @pytest.mark.asyncio
    async def test_main_with_default_csv_path(self, tmp_path):
        """Test main function with default CSV path."""
        # Change to temp directory to test default path
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create default CSV path
            image_folder = tmp_path / "image_folder"
            image_folder.mkdir()
            csv_path = image_folder / "img_conversion_table.csv"
            csv_path.write_text("ID,Image Src,Image Command,Image Position\n001,url1,cmd1,pos1")
            
            with patch('bulk_image_processor.main.BulkImageProcessor') as mock_processor_class:
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor
                mock_processor.process_from_csv = AsyncMock(return_value={"summary": "test"})
                
                await main()
                
                mock_processor_class.assert_called_once()
                mock_processor.process_from_csv.assert_called_once()
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_main_csv_not_found(self, tmp_path):
        """Test main function when CSV file doesn't exist."""
        nonexistent_csv = tmp_path / "nonexistent.csv"
        
        with pytest.raises(SystemExit) as exc_info:
            with patch('bulk_image_processor.main.logger') as mock_logger:
                await main(nonexistent_csv)
        
        assert exc_info.value.code == 1
    
    @pytest.mark.asyncio
    async def test_main_processing_exception(self, tmp_path):
        """Test main function when processing raises an exception."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("ID,Image Src,Image Command,Image Position\n001,url1,cmd1,pos1")
        
        with patch('bulk_image_processor.main.BulkImageProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_from_csv = AsyncMock(side_effect=Exception("Processing failed"))
            
            with pytest.raises(SystemExit) as exc_info:
                await main(csv_path)
            
            assert exc_info.value.code == 1
    
    @pytest.mark.asyncio
    async def test_main_keyboard_interrupt(self, tmp_path):
        """Test main function handles KeyboardInterrupt."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("ID,Image Src,Image Command,Image Position\n001,url1,cmd1,pos1")
        
        with patch('bulk_image_processor.main.BulkImageProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_from_csv = AsyncMock(side_effect=KeyboardInterrupt())
            
            with pytest.raises(SystemExit) as exc_info:
                await main(csv_path)
            
            assert exc_info.value.code == 1


class TestSignalHandling:
    """Test signal handling functionality."""
    
    def test_signal_handler_sets_shutdown_flag(self, mock_settings):
        """Test that signal handler sets shutdown flag."""
        with patch('signal.signal') as mock_signal, \
             patch('bulk_image_processor.main.setup_logging'), \
             patch('bulk_image_processor.main.GeminiAnalyzer'), \
             patch('bulk_image_processor.main.VirtualTryOnProcessor'), \
             patch('bulk_image_processor.main.ProductRecontextProcessor'), \
             patch('bulk_image_processor.main.StorageManager'):
            
            processor = BulkImageProcessor(mock_settings)
            
            # Get the signal handler function
            signal_handler = mock_signal.call_args_list[0][0][1]
            
            # Call the handler
            signal_handler(2, None)  # SIGINT
            
            assert processor.shutdown_requested is True