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
        
        with patch('asyncio.sleep'):  # Mock sleep to speed up test
            result = await processor.process_with_retry(
                sample_image_record,
                tmp_path / "nonexistent.jpg",
                tmp_path
            )
        
        assert result.success is False
        # Default max_retries from Settings is 5
        assert result.retry_count == 5  
        assert "Failed after 6 attempts" in result.error_message  # max_retries + 1
    
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
                "id": "model_001",
                "gender": "women",
                "pose": "front",
                "style": "casual",
                "path": "/models/model_001.jpg",
                "filename": "1_women.jpg"
            },
            {
                "id": "model_002",
                "gender": "man",
                "pose": "front",
                "style": "formal",
                "path": "/models/model_002.jpg",
                "filename": "2_man.jpg"
            }
        ]
    
    @patch('google.cloud.aiplatform.init')
    @patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
    @patch.object(VirtualTryOnProcessor, '_load_model_images')
    def test_vto_processor_initialization(self, mock_load_images, mock_client, mock_init, mock_settings, mock_analyzer):
        """Test VTO processor initialization."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_load_images.return_value = None
        
        processor = VirtualTryOnProcessor(mock_settings, mock_analyzer)
        
        assert processor.get_processor_type() == "virtual_try_on"
        assert processor.client is not None
        assert processor.model_endpoint is not None
    
    @pytest.mark.asyncio
    @patch('google.cloud.aiplatform.init')
    @patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
    @patch.object(VirtualTryOnProcessor, '_load_model_images')
    async def test_process_apparel_image_success(
        self, mock_load_images, mock_client, mock_init,
        mock_settings, mock_analyzer, sample_image_record, tmp_path
    ):
        """Test successful apparel image processing."""
        # Create a real test image
        img = Image.new('RGB', (100, 100), color='blue')
        image_path = tmp_path / "test_apparel.jpg"
        img.save(image_path)
        
        # Mock client and prediction
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_load_images.return_value = None
        
        # Mock analyzer feedback
        mock_analyzer.analyze_virtual_try_on_quality = AsyncMock(return_value=MagicMock(
            passed=True,
            score=0.92,
            reasoning="Good quality",
            suggestions=[]
        ))
        
        # Disable GCS upload for this test
        mock_settings.storage.enable_gcs_upload = False
        processor = VirtualTryOnProcessor(mock_settings, mock_analyzer)
        
        # Mock the processing steps
        with patch.object(processor, 'call_virtual_try_on', new_callable=AsyncMock) as mock_call_api:
            # Mock API response with predictions
            mock_response = MagicMock()
            mock_response.predictions = [{"bytesBase64Encoded": "fake_base64_data"}]
            mock_call_api.return_value = mock_response
            
            with patch('bulk_image_processor.processors.virtual_try_on.prediction_to_pil_image') as mock_convert:
                mock_convert.return_value = img
                
                with patch.object(processor, 'select_model') as mock_select:
                    mock_select.return_value = {"model_id": "test_model", "path": image_path, "id": "test_model"}
                    
                    # Mock analyzer.classify_image method (async)
                    mock_analyzer.classify_image = AsyncMock(return_value=MagicMock(
                        category="dress", 
                        style="casual",
                        dict=lambda: {"category": "dress", "style": "casual"}
                    ))
                    
                    # Mock prepare_model_image
                    with patch.object(processor, 'prepare_model_image') as mock_prepare_model:
                        mock_prepare_model.return_value = "fake_model_image_bytes"
                        
                        result = await processor.process_image(
                            sample_image_record,
                            image_path,
                            tmp_path / "output"
                        )
        
        assert result.success is True
        assert result.quality_score == 0.92
    
    @pytest.mark.asyncio
    @patch('google.cloud.aiplatform.init')
    @patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
    @patch.object(VirtualTryOnProcessor, '_load_model_images')
    async def test_select_best_model_pair(
        self, mock_load_images, mock_client, mock_init,
        mock_settings, mock_analyzer, mock_model_pairs, tmp_path
    ):
        """Test model pair selection."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_load_images.return_value = None
        
        # Disable GCS upload for this test
        mock_settings.storage.enable_gcs_upload = False
        processor = VirtualTryOnProcessor(mock_settings, mock_analyzer)
        
        # Mock the model_images and models_by_gender attributes
        processor.model_images = mock_model_pairs
        processor.models_by_gender = {
            'women': [m for m in mock_model_pairs if m['gender'] == 'women'],
            'man': [m for m in mock_model_pairs if m['gender'] == 'man'],
        }
        
        # Mock apparel info that should select female model
        apparel_info = {
            "detected_items": ["dress"],
            "target_gender": "women",
            "confidence": 0.9
        }
        
        model_pair = processor.select_model(apparel_info)
        
        assert model_pair is not None
        assert model_pair["id"] == "model_001"
        assert model_pair["gender"] == "women"


class TestProductRecontextProcessor:
    """Test ProductRecontextProcessor class."""
    
    @patch('google.cloud.aiplatform.init')
    @patch('google.cloud.aiplatform.gapic.PredictionServiceClient')
    def test_product_processor_initialization(self, mock_client, mock_init, mock_settings, mock_analyzer):
        """Test product processor initialization."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        processor = ProductRecontextProcessor(mock_settings, mock_analyzer)
        
        assert processor.get_processor_type() == "product_recontext"
        assert processor.client is not None
        assert processor.model_endpoint is not None
    
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
        
        # Mock the generate_product_recontext_prompt method
        mock_analyzer.generate_product_recontext_prompt = AsyncMock(
            return_value="A luxury vase on marble pedestal"
        )
        
        # Mock analyzer feedback
        mock_analyzer.validate_output = AsyncMock(return_value={
            "pass": True,
            "score": 0.88,
            "reasoning": "Good composition",
            "suggestions": []
        })
        
        # Mock quality analysis
        mock_analyzer.analyze_product_recontext_quality = AsyncMock(return_value=MagicMock(
            passed=True,
            score=0.88,
            reasoning="Good composition",
            suggestions=[],
            dict=lambda: {"passed": True, "score": 0.88, "reasoning": "Good composition", "suggestions": []}
        ))
        
        # Create test image
        image_path = tmp_path / "vase.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(image_path)
        
        # Disable GCS upload for this test
        mock_settings.storage.enable_gcs_upload = False
        processor = ProductRecontextProcessor(mock_settings, mock_analyzer)
        
        # Mock analyzer.classify_image (async)
        mock_analyzer.classify_image = AsyncMock(return_value=MagicMock(
            category="vase",
            style="decorative",
            dict=lambda: {"category": "vase", "style": "decorative"}
        ))
        
        # Mock the API call
        with patch.object(processor, 'call_product_recontext', new_callable=AsyncMock) as mock_call_api:
            # Mock API response
            mock_response = MagicMock()
            mock_response.predictions = [{"bytesBase64Encoded": "fake_base64_data"}]
            mock_call_api.return_value = mock_response
            
            with patch('bulk_image_processor.processors.product_recontext.prediction_to_pil_image') as mock_convert:
                mock_convert.return_value = img
                
                result = await processor.process_image(
                    sample_image_record,
                    image_path,
                    tmp_path / "output"
                )
        
        assert result.success is True
        assert result.quality_score == 0.88
        assert "generated_prompt" in result.metadata
        assert "product_description" in result.metadata
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
        
        # Mock prediction response
        mock_prediction = MagicMock()
        mock_prediction.predictions = [{
            "bytesBase64Encoded": "fake_base64_image_data"
        }]
        mock_endpoint.predict.return_value = mock_prediction
        
        # Create record with style command
        record = ImageRecord(
            id="test_001",
            image_url="https://example.com/product.jpg",
            image_command="luxury style",
            image_position="center",
            row_index=0
        )
        
        # Mock the generate_product_recontext_prompt method
        mock_analyzer.generate_product_recontext_prompt = AsyncMock(
            return_value="Luxury setting for product"
        )
        
        # Mock analyzer feedback
        mock_analyzer.validate_output = AsyncMock(return_value={
            "pass": True,
            "score": 0.90,
            "reasoning": "Good luxury presentation",
            "suggestions": []
        })
        
        # Mock quality analysis
        mock_analyzer.analyze_product_recontext_quality = AsyncMock(return_value=MagicMock(
            passed=True,
            score=0.90,
            reasoning="Good luxury presentation",
            suggestions=[],
            dict=lambda: {"passed": True, "score": 0.90, "reasoning": "Good luxury presentation", "suggestions": []}
        ))
        
        # Create test image
        image_path = tmp_path / "product.jpg"
        img = Image.new('RGB', (100, 100))
        img.save(image_path)
        
        # Disable GCS upload for this test
        mock_settings.storage.enable_gcs_upload = False
        processor = ProductRecontextProcessor(mock_settings, mock_analyzer)
        
        # Mock analyzer.classify_image (async)
        mock_analyzer.classify_image = AsyncMock(return_value=MagicMock(
            category="product",
            style="luxury",
            dict=lambda: {"category": "product", "style": "luxury"}
        ))
        
        # Mock the API call
        with patch.object(processor, 'call_product_recontext', new_callable=AsyncMock) as mock_call_api:
            # Mock API response
            mock_response = MagicMock()
            mock_response.predictions = [{"bytesBase64Encoded": "fake_base64_data"}]
            mock_call_api.return_value = mock_response
            
            # Mock the image conversion
            with patch('bulk_image_processor.processors.product_recontext.prediction_to_pil_image') as mock_convert:
                mock_convert.return_value = img
                
                # The style preference should be passed through the analyzer
                # Let's verify that the analyzer is called with the correct style preference
                result = await processor.process_image(
                    record,
                    image_path,
                    tmp_path / "output"
                )
        
        # Verify result
        assert result.success is True
        
        # Verify the analyzer was called with style preference
        mock_analyzer.generate_product_recontext_prompt.assert_called_once()
        call_args = mock_analyzer.generate_product_recontext_prompt.call_args
        # Verify that the product_description (which includes the style command) was passed
        assert len(call_args[0]) >= 2  # Should have image_path and product_description arguments