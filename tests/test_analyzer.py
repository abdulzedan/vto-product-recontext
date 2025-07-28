"""Tests for the Gemini analyzer module."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import google.generativeai as genai

from bulk_image_processor.analyzer import GeminiAnalyzer, ImageCategory, ClassificationResult
from bulk_image_processor.exceptions import AnalysisError, ImageValidationError


class TestClassificationResult:
    """Test ClassificationResult model."""
    
    def test_create_classification_result(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            category=ImageCategory.APPAREL,
            confidence=0.95,
            reasoning="Detected clothing items",
            detected_items=["shirt", "pants"],
            metadata={"style": "casual"}
        )
        
        assert result.category == ImageCategory.APPAREL
        assert result.confidence == 0.95
        assert result.reasoning == "Detected clothing items"
        assert "shirt" in result.detected_items
        assert result.metadata["style"] == "casual"
    
    def test_classification_result_defaults(self):
        """Test classification result with defaults."""
        result = ClassificationResult(
            category=ImageCategory.PRODUCT,
            confidence=0.8,
            reasoning="Detected product"
        )
        
        assert result.detected_items == []
        assert result.metadata == {}


class TestGeminiAnalyzer:
    """Test GeminiAnalyzer class."""
    
    @pytest.fixture
    def mock_gemini_model(self):
        """Mock Gemini model for testing."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                instance = MagicMock()
                mock_model.return_value = instance
                yield instance
    
    def test_analyzer_initialization(self, mock_settings, mock_gemini_model):
        """Test analyzer initialization."""
        analyzer = GeminiAnalyzer(mock_settings)
        
        assert analyzer.settings == mock_settings
        assert analyzer.client is not None
    
    @pytest.mark.asyncio
    async def test_classify_image_apparel(self, mock_settings, mock_gemini_model, tmp_path):
        """Test classifying an apparel image."""
        # Create a mock image file
        image_path = tmp_path / "shirt.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "APPAREL",
            "confidence": 0.95,
            "reasoning": "The image shows a blue shirt",
            "detected_items": ["shirt"],
            "metadata": {"color": "blue", "style": "casual"}
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.classify_image(image_path)
        
        assert result.category == ImageCategory.APPAREL
        assert result.confidence == 0.95
        assert "shirt" in result.detected_items
        assert result.metadata["color"] == "blue"
        assert analyzer.stats['successful_classifications'] == 1
    
    @pytest.mark.asyncio
    async def test_classify_image_product(self, mock_settings, mock_gemini_model, tmp_path):
        """Test classifying a product image."""
        # Create a mock image file
        image_path = tmp_path / "vase.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "PRODUCT",
            "confidence": 0.88,
            "reasoning": "The image shows a decorative vase",
            "detected_items": ["vase"],
            "metadata": {"material": "ceramic", "use": "decoration"}
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.classify_image(image_path)
        
        assert result.category == ImageCategory.PRODUCT
        assert result.confidence == 0.88
        assert "vase" in result.detected_items
        assert result.metadata["material"] == "ceramic"
    
    @pytest.mark.asyncio
    async def test_classify_image_with_additional_context(self, mock_settings, mock_gemini_model, tmp_path):
        """Test classifying image with additional context."""
        image_path = tmp_path / "item.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "APPAREL",
            "confidence": 0.92,
            "reasoning": "Context suggests apparel processing",
            "detected_items": ["garment"],
            "metadata": {}
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.classify_image(
            image_path, 
            additional_context="Command: virtual-try-on"
        )
        
        # Verify the prompt includes additional context
        call_args = mock_gemini_model.generate_content_async.call_args
        assert "Additional context: Command: virtual-try-on" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_classify_image_invalid_json_response(self, mock_settings, mock_gemini_model, tmp_path):
        """Test handling invalid JSON response from Gemini."""
        image_path = tmp_path / "item.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        
        with pytest.raises(AnalysisError) as exc_info:
            await analyzer.classify_image(image_path)
        
        assert "Failed to parse classification response" in str(exc_info.value)
        assert analyzer.stats['failed_classifications'] == 1
    
    @pytest.mark.asyncio
    async def test_classify_image_missing_fields(self, mock_settings, mock_gemini_model, tmp_path):
        """Test handling response with missing required fields."""
        image_path = tmp_path / "item.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "UNKNOWN",
            "reasoning": "Cannot determine category"
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        
        # Should use default confidence value
        result = await analyzer.classify_image(image_path)
        assert result.category == ImageCategory.UNKNOWN
        assert result.confidence == 0.5  # default value
    
    @pytest.mark.asyncio
    async def test_classify_image_api_error(self, mock_settings, mock_gemini_model, tmp_path):
        """Test handling API errors."""
        image_path = tmp_path / "item.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        mock_gemini_model.generate_content_async = AsyncMock(
            side_effect=Exception("API quota exceeded")
        )
        
        analyzer = GeminiAnalyzer(mock_settings)
        
        with pytest.raises(AnalysisError) as exc_info:
            await analyzer.classify_image(image_path)
        
        assert "Failed to classify image" in str(exc_info.value)
        assert exc_info.value.context['api_error'] == "API quota exceeded"
    
    @pytest.mark.asyncio
    async def test_validate_output_pass(self, mock_settings, mock_gemini_model, tmp_path):
        """Test output validation with passing result."""
        output_path = tmp_path / "output.jpg"
        output_path.write_bytes(b"fake_output_data")
        
        mock_response = MagicMock()
        mock_response.text = """
        {
            "pass": true,
            "score": 0.92,
            "reasoning": "High quality output with proper composition",
            "suggestions": []
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.validate_output(
            output_path,
            processing_type="virtual_try_on"
        )
        
        assert result["pass"] is True
        assert result["score"] == 0.92
        assert result["reasoning"] == "High quality output with proper composition"
        assert analyzer.stats['successful_validations'] == 1
    
    @pytest.mark.asyncio
    async def test_validate_output_fail(self, mock_settings, mock_gemini_model, tmp_path):
        """Test output validation with failing result."""
        output_path = tmp_path / "output.jpg"
        output_path.write_bytes(b"fake_output_data")
        
        mock_response = MagicMock()
        mock_response.text = """
        {
            "pass": false,
            "score": 0.45,
            "reasoning": "Poor alignment and artifacts visible",
            "suggestions": ["Retry with better model alignment", "Check input image quality"]
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.validate_output(
            output_path,
            processing_type="virtual_try_on"
        )
        
        assert result["pass"] is False
        assert result["score"] == 0.45
        assert len(result["suggestions"]) == 2
        assert "Retry with better model alignment" in result["suggestions"]
    
    @pytest.mark.asyncio
    async def test_generate_scene_prompt(self, mock_settings, mock_gemini_model, tmp_path):
        """Test scene prompt generation."""
        image_path = tmp_path / "product.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        mock_response = MagicMock()
        mock_response.text = """
        {
            "prompt": "A luxury ceramic vase on a marble pedestal in a high-end gallery",
            "style": "luxury",
            "lighting": "soft gallery lighting",
            "background": "minimalist gallery space"
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.generate_scene_prompt(
            image_path,
            style_preference="luxury"
        )
        
        assert "prompt" in result
        assert "luxury ceramic vase" in result["prompt"]
        assert result["style"] == "luxury"
        assert result["lighting"] == "soft gallery lighting"
    
    def test_get_stats(self, mock_settings, mock_gemini_model):
        """Test getting analyzer statistics."""
        analyzer = GeminiAnalyzer(mock_settings)
        
        # Manually update some stats
        analyzer.stats['total_classifications'] = 10
        analyzer.stats['successful_classifications'] = 8
        analyzer.stats['failed_classifications'] = 2
        
        stats = analyzer.get_stats()
        
        assert stats['total_classifications'] == 10
        assert stats['successful_classifications'] == 8
        assert stats['failed_classifications'] == 2
        assert stats['classification_success_rate'] == 80.0
    
    @pytest.mark.asyncio
    async def test_classify_nonexistent_file(self, mock_settings, mock_gemini_model):
        """Test classifying a nonexistent file."""
        analyzer = GeminiAnalyzer(mock_settings)
        
        with pytest.raises(AnalysisError) as exc_info:
            await analyzer.classify_image(Path("/nonexistent/file.jpg"))
        
        assert "Failed to read image" in str(exc_info.value)