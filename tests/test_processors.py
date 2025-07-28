"""Tests for image processors."""

import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from bulk_image_processor.processors.base import BaseProcessor, ProcessingResult
from bulk_image_processor.processors.virtual_try_on import VirtualTryOnProcessor
from bulk_image_processor.processors.product_recontext import ProductRecontextProcessor
from bulk_image_processor.downloader import ImageRecord
from bulk_image_processor.analyzer import ImageCategory, ClassificationResult
from bulk_image_processor.exceptions import ProcessingError, ImageValidationError


class TestProcessingResult:
    """Test ProcessingResult model."""
    
    def test_create_processing_result_success(self):
        """Test creating a successful processing result."""
        result = ProcessingResult(
            record_id="test_001",
            success=True,
            output_path=Path("/output/test.jpg"),
            gcs_path="gs://bucket/test.jpg",
            metadata={"model": "model_001"},
            processing_time=2.5,
            retry_count=0,
            quality_score=0.95,
            feedback="High quality output"
        )
        
        assert result.record_id == "test_001"
        assert result.success is True
        assert result.output_path == Path("/output/test.jpg")
        assert result.gcs_path == "gs://bucket/test.jpg"
        assert result.metadata["model"] == "model_001"
        assert result.processing_time == 2.5
        assert result.retry_count == 0
        assert result.quality_score == 0.95
        assert result.feedback == "High quality output"
    
    def test_create_processing_result_failure(self):
        """Test creating a failed processing result."""
        result = ProcessingResult(
            record_id="test_002",
            success=False,
            error_message="Processing failed: timeout",
            processing_time=30.0,
            retry_count=3
        )
        
        assert result.record_id == "test_002"
        assert result.success is False
        assert result.output_path is None
        assert result.error_message == "Processing failed: timeout"
        assert result.retry_count == 3


class TestBaseProcessor:
    """Test BaseProcessor abstract class."""
    
    class ConcreteProcessor(BaseProcessor):
        """Concrete implementation for testing."""
        
        async def process_image(self, record, image_path, output_dir):
            return ProcessingResult(
                record_id=record.id,
                success=True,
                output_path=output_dir / "output.jpg"
            )
        
        def get_processor_type(self):
            return "test_processor"
    
    def test_processor_initialization(self, mock_settings, mock_analyzer):
        """Test processor initialization."""
        processor = self.ConcreteProcessor(mock_settings, mock_analyzer)
        
        assert processor.settings == mock_settings
        assert processor.analyzer == mock_analyzer
        assert processor.processing_stats['total_processed'] == 0
        assert processor.processing_stats['successful'] == 0
        assert processor.processing_stats['failed'] == 0
    
    def test_create_output_directory(self, mock_settings, mock_analyzer, tmp_path):
        """Test creating output directory."""
        processor = self.ConcreteProcessor(mock_settings, mock_analyzer)
        record = ImageRecord(
            id="test_001",
            image_url="https://example.com/test.jpg",
            image_command="process",
            image_position="center",
            row_index=0
        )
        
        output_dir = processor.create_output_directory(tmp_path, record)
        
        assert output_dir.exists()
        assert output_dir.is_dir()
        assert "test_001" in str(output_dir)
    
    def test_prepare_metadata(self, mock_settings, mock_analyzer):
        """Test preparing metadata."""
        processor = self.ConcreteProcessor(mock_settings, mock_analyzer)
        record = ImageRecord(
            id="test_001",
            image_url="https://example.com/test.jpg",
            image_command="enhance",
            image_position="front",
            row_index=0
        )
        
        metadata = processor.prepare_metadata(
            record,
            additional_metadata={"custom_field": "value"}
        )
        
        assert metadata['record_id'] == "test_001"
        assert metadata['original_url'] == "https://example.com/test.jpg"
        assert metadata['image_command'] == "enhance"
        assert metadata['image_position'] == "front"
        assert metadata['processor_type'] == "test_processor"
        assert metadata['custom_field'] == "value"
        assert 'timestamp' in metadata
        assert 'settings' in metadata
    
    def test_update_stats(self, mock_settings, mock_analyzer):
        """Test updating processing statistics."""
        processor = self.ConcreteProcessor(mock_settings, mock_analyzer)
        
        # Update with success
        processor.update_stats(success=True, processing_time=1.5, retry_count=0)
        
        assert processor.processing_stats['total_processed'] == 1
        assert processor.processing_stats['successful'] == 1
        assert processor.processing_stats['failed'] == 0
        assert processor.processing_stats['total_processing_time'] == 1.5
        
        # Update with failure
        processor.update_stats(success=False, processing_time=2.0, retry_count=2)
        
        assert processor.processing_stats['total_processed'] == 2
        assert processor.processing_stats['successful'] == 1
        assert processor.processing_stats['failed'] == 1
        assert processor.processing_stats['total_processing_time'] == 3.5
        assert processor.processing_stats['retry_attempts'] == 2
    
    def test_get_stats(self, mock_settings, mock_analyzer):
        """Test getting statistics."""
        processor = self.ConcreteProcessor(mock_settings, mock_analyzer)
        
        # Add some stats
        processor.update_stats(success=True, processing_time=1.0)
        processor.update_stats(success=True, processing_time=2.0)
        processor.update_stats(success=False, processing_time=1.5)
        
        stats = processor.get_stats()
        
        assert stats['total_processed'] == 3
        assert stats['successful'] == 2
        assert stats['failed'] == 1
        assert stats['success_rate'] == pytest.approx(66.67, rel=0.01)
        assert stats['average_processing_time'] == 1.5
    
    @pytest.mark.asyncio
    async def test_process_with_retry_success(self, mock_settings, mock_analyzer, sample_image_record, tmp_path):
        """Test processing with retry - successful case."""
        processor = self.ConcreteProcessor(mock_settings, mock_analyzer)
        
        # Create a test image
        image_path = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
        
        result = await processor.process_with_retry(sample_image_record, image_path, tmp_path)
        
        assert result.success is True
        assert result.record_id == sample_image_record.id
        assert result.retry_count == 0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_process_with_retry_failure(self, mock_settings, mock_analyzer, sample_image_record, tmp_path):
        """Test processing with retry - failure case."""
        class FailingProcessor(BaseProcessor):
            async def process_image(self, record, image_path, output_dir):
                raise ProcessingError("Simulated processing failure")
            
            def get_processor_type(self):
                return "failing_processor"
        
        processor = FailingProcessor(mock_settings, mock_analyzer)
        mock_settings.processing.max_retries = 2
        
        with patch('asyncio.sleep'):  # Mock sleep to speed up test
            result = await processor.process_with_retry(
                sample_image_record,
                tmp_path / "nonexistent.jpg",
                tmp_path
            )
        
        assert result.success is False
        assert result.retry_count == 2
        assert "Failed after 3 attempts" in result.error_message
    
    def test_validate_image(self, mock_settings, mock_analyzer, tmp_path):
        """Test image validation."""
        processor = self.ConcreteProcessor(mock_settings, mock_analyzer)
        
        # Valid image
        valid_image = tmp_path / "valid.jpg"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(valid_image)
        
        assert processor.validate_image(valid_image) is True
        
        # Invalid image (corrupted file)
        invalid_image = tmp_path / "invalid.jpg"
        invalid_image.write_bytes(b"not an image")
        
        assert processor.validate_image(invalid_image) is False


class TestVirtualTryOnProcessor:
    """Test VirtualTryOnProcessor class."""
    
    @pytest.fixture
    def mock_model_pairs(self):
        """Mock model pairs data."""
        return [
            {
                "model_id": "model_001",
                "gender": "female",
                "pose": "front",
                "style": "casual",
                "image_path": "/models/model_001.jpg"
            },
            {
                "model_id": "model_002",
                "gender": "male",
                "pose": "front",
                "style": "formal",
                "image_path": "/models/model_002.jpg"
            }
        ]
    
    @patch('bulk_image_processor.processors.virtual_try_on.load_model_pairs')
    def test_vto_processor_initialization(self, mock_load_models, mock_settings, mock_analyzer, mock_model_pairs):
        """Test VTO processor initialization."""
        mock_load_models.return_value = mock_model_pairs
        
        processor = VirtualTryOnProcessor(mock_settings, mock_analyzer)
        
        assert processor.get_processor_type() == "virtual_try_on"
        assert len(processor.model_pairs) == 2
        assert processor.endpoint_client is not None
    
    @pytest.mark.asyncio
    @patch('bulk_image_processor.processors.virtual_try_on.load_model_pairs')
    @patch('google.cloud.aiplatform.Endpoint')
    async def test_process_apparel_image_success(
        self, mock_endpoint_class, mock_load_models, 
        mock_settings, mock_analyzer, mock_model_pairs, 
        sample_image_record, tmp_path
    ):
        """Test successful apparel image processing."""
        mock_load_models.return_value = mock_model_pairs
        
        # Mock endpoint
        mock_endpoint = MagicMock()
        mock_endpoint_class.return_value = mock_endpoint
        
        # Mock prediction response
        mock_prediction = MagicMock()
        mock_prediction.predictions = [{
            "bytesBase64Encoded": "fake_base64_image_data"
        }]
        mock_endpoint.predict.return_value = mock_prediction
        
        # Mock analyzer feedback
        mock_analyzer.validate_output = AsyncMock(return_value={
            "pass": True,
            "score": 0.92,
            "reasoning": "Good quality",
            "suggestions": []
        })
        
        # Create test image
        image_path = tmp_path / "shirt.jpg"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(image_path)
        
        processor = VirtualTryOnProcessor(mock_settings, mock_analyzer)
        
        with patch('bulk_image_processor.processors.virtual_try_on.prediction_to_pil_image') as mock_convert:
            mock_convert.return_value = img
            
            result = await processor.process_image(
                sample_image_record,
                image_path,
                tmp_path / "output"
            )
        
        assert result.success is True
        assert result.quality_score == 0.92
        assert "model_id" in result.metadata
        assert result.output_path is not None
    
    @pytest.mark.asyncio
    @patch('bulk_image_processor.processors.virtual_try_on.load_model_pairs')
    async def test_select_best_model_pair(
        self, mock_load_models, mock_settings, mock_analyzer, 
        mock_model_pairs, tmp_path
    ):
        """Test model pair selection."""
        mock_load_models.return_value = mock_model_pairs
        
        processor = VirtualTryOnProcessor(mock_settings, mock_analyzer)
        
        # Create test image
        image_path = tmp_path / "dress.jpg"
        img = Image.new('RGB', (100, 100), color='pink')
        img.save(image_path)
        
        # Mock analyzer to suggest female model
        mock_analyzer.analyze_apparel_for_model_selection = AsyncMock(
            return_value={
                "suggested_gender": "female",
                "suggested_pose": "front",
                "style_attributes": ["casual", "summer"]
            }
        )
        
        model_pair = await processor._select_best_model_pair(image_path)
        
        assert model_pair is not None
        assert model_pair["model_id"] == "model_001"
        assert model_pair["gender"] == "female"


class TestProductRecontextProcessor:
    """Test ProductRecontextProcessor class."""
    
    @patch('google.cloud.aiplatform.Endpoint')
    def test_product_processor_initialization(self, mock_endpoint_class, mock_settings, mock_analyzer):
        """Test product processor initialization."""
        mock_endpoint = MagicMock()
        mock_endpoint_class.return_value = mock_endpoint
        
        processor = ProductRecontextProcessor(mock_settings, mock_analyzer)
        
        assert processor.get_processor_type() == "product_recontext"
        assert processor.endpoint_client is not None
    
    @pytest.mark.asyncio
    @patch('google.cloud.aiplatform.Endpoint')
    async def test_process_product_image_success(
        self, mock_endpoint_class, mock_settings, mock_analyzer,
        sample_image_record, tmp_path
    ):
        """Test successful product image processing."""
        # Mock endpoint
        mock_endpoint = MagicMock()
        mock_endpoint_class.return_value = mock_endpoint
        
        # Mock prediction response
        mock_prediction = MagicMock()
        mock_prediction.predictions = [{
            "bytesBase64Encoded": "fake_base64_image_data"
        }]
        mock_endpoint.predict.return_value = mock_prediction
        
        # Mock analyzer scene generation
        mock_analyzer.generate_scene_prompt = AsyncMock(return_value={
            "prompt": "A luxury vase on marble pedestal",
            "style": "luxury",
            "lighting": "soft gallery lighting"
        })
        
        # Mock analyzer feedback
        mock_analyzer.validate_output = AsyncMock(return_value={
            "pass": True,
            "score": 0.88,
            "reasoning": "Good composition",
            "suggestions": []
        })
        
        # Create test image
        image_path = tmp_path / "vase.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(image_path)
        
        processor = ProductRecontextProcessor(mock_settings, mock_analyzer)
        
        with patch('bulk_image_processor.processors.product_recontext.prediction_to_pil_image') as mock_convert:
            mock_convert.return_value = img
            
            result = await processor.process_image(
                sample_image_record,
                image_path,
                tmp_path / "output"
            )
        
        assert result.success is True
        assert result.quality_score == 0.88
        assert "scene_prompt" in result.metadata
        assert "style" in result.metadata
        assert result.output_path is not None
    
    @pytest.mark.asyncio
    @patch('google.cloud.aiplatform.Endpoint')
    async def test_process_product_with_style_preference(
        self, mock_endpoint_class, mock_settings, mock_analyzer,
        tmp_path
    ):
        """Test processing with style preference from command."""
        mock_endpoint = MagicMock()
        mock_endpoint_class.return_value = mock_endpoint
        
        # Create record with style command
        record = ImageRecord(
            id="test_001",
            image_url="https://example.com/product.jpg",
            image_command="luxury style",
            image_position="center",
            row_index=0
        )
        
        # Mock analyzer to use the style preference
        mock_analyzer.generate_scene_prompt = AsyncMock(return_value={
            "prompt": "Luxury setting for product",
            "style": "luxury"
        })
        
        # Create test image
        image_path = tmp_path / "product.jpg"
        img = Image.new('RGB', (100, 100))
        img.save(image_path)
        
        processor = ProductRecontextProcessor(mock_settings, mock_analyzer)
        
        # Mock the process to check style preference is extracted
        with patch.object(processor, '_extract_style_preference') as mock_extract:
            mock_extract.return_value = "luxury"
            style = processor._extract_style_preference(record.image_command)
            
            assert style == "luxury"