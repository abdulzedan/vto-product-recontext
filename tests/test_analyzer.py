"""Tests for the Gemini analyzer module."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import google.generativeai as genai
import pytest
from PIL import Image

from bulk_image_processor.analyzer import (
    ClassificationResult,
    FeedbackResult,
    GeminiAnalyzer,
    ImageCategory,
)
from bulk_image_processor.exceptions import AnalysisError


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_create_classification_result(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            category=ImageCategory.APPAREL,
            confidence=0.95,
            reasoning="Detected clothing items",
            detected_items=["shirt", "pants"],
            metadata={"style": "casual"},
        )

        assert result.category == ImageCategory.APPAREL
        assert result.confidence == 0.95
        assert result.reasoning == "Detected clothing items"
        assert "shirt" in result.detected_items
        assert result.metadata["style"] == "casual"

    def test_classification_result_defaults(self):
        """Test classification result with defaults."""
        result = ClassificationResult(
            category=ImageCategory.PRODUCT, confidence=0.8, reasoning="Detected product"
        )

        assert result.detected_items == []
        assert result.metadata == {}


class TestGeminiAnalyzer:
    """Test GeminiAnalyzer class."""

    @pytest.fixture
    def mock_gemini_model(self):
        """Mock Gemini model for testing."""
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel") as mock_model:
                instance = MagicMock()
                mock_model.return_value = instance
                yield instance

    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create a real test image file."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        image_path = tmp_path / "test_image.jpg"
        img.save(image_path)
        return image_path

    def test_analyzer_initialization(self, mock_settings, mock_gemini_model):
        """Test analyzer initialization."""
        analyzer = GeminiAnalyzer(mock_settings)

        assert analyzer.settings == mock_settings
        assert analyzer.client is not None

    @pytest.mark.asyncio
    async def test_classify_image_apparel(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test classifying an apparel image."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "apparel",
            "confidence": 0.95,
            "reasoning": "The image shows a blue shirt",
            "detected_items": ["shirt"],
            "metadata": {"color": "blue", "style": "casual"}
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.classify_image(test_image_path)

        assert result.category == ImageCategory.APPAREL
        assert result.confidence == 0.95
        assert "shirt" in result.detected_items
        assert result.metadata["color"] == "blue"

    @pytest.mark.asyncio
    async def test_classify_image_product(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test classifying a product image."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "product",
            "confidence": 0.88,
            "reasoning": "The image shows a decorative vase",
            "detected_items": ["vase"],
            "metadata": {"material": "ceramic", "use": "decoration"}
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.classify_image(test_image_path)

        assert result.category == ImageCategory.PRODUCT
        assert result.confidence == 0.88
        assert "vase" in result.detected_items
        assert result.metadata["material"] == "ceramic"

    @pytest.mark.asyncio
    async def test_classify_image_with_additional_context(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test classifying image with additional context."""
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "apparel",
            "confidence": 0.92,
            "reasoning": "Context suggests apparel processing",
            "detected_items": ["garment"],
            "metadata": {}
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.classify_image(
            test_image_path, additional_context="Command: virtual-try-on"
        )

        assert result.category == ImageCategory.APPAREL
        assert result.confidence == 0.92

        # Verify the prompt includes additional context
        call_args = mock_gemini_model.generate_content_async.call_args
        assert "Additional context: Command: virtual-try-on" in str(call_args)

    @pytest.mark.asyncio
    async def test_classify_image_invalid_json_response(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test handling invalid JSON response from Gemini."""
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.classify_image(test_image_path)

        # Should return unknown category with default values when JSON parsing fails
        assert result.category == ImageCategory.UNKNOWN
        assert result.confidence == 0.0
        assert "Failed to parse response" in result.reasoning

    @pytest.mark.asyncio
    async def test_classify_image_missing_fields(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test handling response with missing required fields."""
        mock_response = MagicMock()
        mock_response.text = """
        {
            "category": "unknown",
            "reasoning": "Cannot determine category"
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)

        # Should use default confidence value
        result = await analyzer.classify_image(test_image_path)
        assert result.category == ImageCategory.UNKNOWN
        assert result.confidence == 0.5  # default value

    @pytest.mark.asyncio
    async def test_classify_image_api_error(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test handling API errors."""
        mock_gemini_model.generate_content_async = AsyncMock(
            side_effect=Exception("API quota exceeded")
        )

        analyzer = GeminiAnalyzer(mock_settings)

        with pytest.raises(Exception, match="API quota exceeded"):
            await analyzer.classify_image(test_image_path)

    @pytest.mark.asyncio
    async def test_analyze_virtual_try_on_quality_pass(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test Virtual Try-On quality analysis with passing result."""
        mock_response = MagicMock()
        mock_response.text = """
        {
            "passed": true,
            "score": 0.92,
            "reasoning": "High quality output with proper composition",
            "suggestions": [],
            "metadata": {
                "fit_quality": "excellent",
                "color_accuracy": "high"
            }
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.analyze_virtual_try_on_quality(
            test_image_path,  # result image
            test_image_path,  # original apparel
            test_image_path,  # model image
        )

        assert result.passed is True
        assert result.score == 0.92
        assert result.reasoning == "High quality output with proper composition"
        assert result.metadata["fit_quality"] == "excellent"

    @pytest.mark.asyncio
    async def test_analyze_product_recontext_quality_fail(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test Product Recontext quality analysis with failing result."""
        mock_response = MagicMock()
        mock_response.text = """
        {
            "passed": false,
            "score": 0.45,
            "reasoning": "Poor alignment and artifacts visible",
            "suggestions": ["Retry with better model alignment", "Check input image quality"],
            "metadata": {
                "product_visibility": "poor",
                "retail_appeal": "low"
            }
        }
        """
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.analyze_product_recontext_quality(
            test_image_path,  # result image
            test_image_path,  # original product
            "A luxury product in elegant setting",  # generated prompt
        )

        assert result.passed is False
        assert result.score == 0.45
        assert len(result.suggestions) == 2
        assert "Retry with better model alignment" in result.suggestions

    @pytest.mark.asyncio
    async def test_generate_product_recontext_prompt(
        self, mock_settings, mock_gemini_model, test_image_path
    ):
        """Test product recontext prompt generation."""
        mock_response = MagicMock()
        mock_response.text = "A luxury ceramic vase on a marble pedestal in a high-end gallery with soft lighting"
        mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = GeminiAnalyzer(mock_settings)
        result = await analyzer.generate_product_recontext_prompt(
            test_image_path, product_description="luxury ceramic vase"
        )

        assert isinstance(result, str)
        assert "luxury ceramic vase" in result
        assert "gallery" in result

    @pytest.mark.asyncio
    async def test_classify_nonexistent_file(self, mock_settings, mock_gemini_model):
        """Test classifying a nonexistent file."""
        analyzer = GeminiAnalyzer(mock_settings)

        with pytest.raises(
            Exception
        ):  # PIL will raise an exception for nonexistent files
            await analyzer.classify_image(Path("/nonexistent/file.jpg"))
