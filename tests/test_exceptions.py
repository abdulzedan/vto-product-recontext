"""Tests for custom exceptions."""

import pytest

from bulk_image_processor.exceptions import (
    AnalysisError,
    BulkImageProcessorError,
    ConfigurationError,
    CSVParsingError,
    DownloadError,
    ImageValidationError,
    ProcessingError,
    RetryExhaustedError,
    StorageError,
    TimeoutError,
)


class TestBulkImageProcessorError:
    """Test base exception class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = BulkImageProcessorError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.context == {}

    def test_error_with_context(self):
        """Test error with context."""
        context = {"file": "test.jpg", "line": 42}
        error = BulkImageProcessorError("Test error", context)

        assert error.message == "Test error"
        assert error.context == context
        assert "file=test.jpg" in str(error)
        assert "line=42" in str(error)

    def test_error_inheritance(self):
        """Test that specific errors inherit from base."""
        assert issubclass(DownloadError, BulkImageProcessorError)
        assert issubclass(ProcessingError, BulkImageProcessorError)
        assert issubclass(AnalysisError, BulkImageProcessorError)


class TestDownloadError:
    """Test DownloadError class."""

    def test_download_error_with_url(self):
        """Test download error with URL."""
        error = DownloadError("Failed to download", url="https://example.com/image.jpg")

        assert error.message == "Failed to download"
        assert error.context["url"] == "https://example.com/image.jpg"
        assert "url=https://example.com/image.jpg" in str(error)

    def test_download_error_with_status_code(self):
        """Test download error with status code."""
        error = DownloadError(
            "HTTP error", url="https://example.com/image.jpg", status_code=404
        )

        assert error.context["url"] == "https://example.com/image.jpg"
        assert error.context["status_code"] == 404


class TestImageValidationError:
    """Test ImageValidationError class."""

    def test_validation_error_with_file_path(self):
        """Test validation error with file path."""
        error = ImageValidationError(
            "Invalid format", file_path="/path/to/image.jpg", validation_type="format"
        )

        assert error.message == "Invalid format"
        assert error.context["file_path"] == "/path/to/image.jpg"
        assert error.context["validation_type"] == "format"


class TestProcessingError:
    """Test ProcessingError class."""

    def test_processing_error_with_processor_type(self):
        """Test processing error with processor type."""
        error = ProcessingError(
            "Processing failed", processor_type="virtual_try_on", record_id="test_001"
        )

        assert error.message == "Processing failed"
        assert error.context["processor_type"] == "virtual_try_on"
        assert error.context["record_id"] == "test_001"


class TestAnalysisError:
    """Test AnalysisError class."""

    def test_analysis_error_with_api_error(self):
        """Test analysis error with API error."""
        error = AnalysisError(
            "Classification failed",
            analysis_type="classification",
            api_error="API quota exceeded",
        )

        assert error.message == "Classification failed"
        assert error.context["analysis_type"] == "classification"
        assert error.context["api_error"] == "API quota exceeded"


class TestStorageError:
    """Test StorageError class."""

    def test_storage_error_with_operation(self):
        """Test storage error with operation details."""
        error = StorageError("Upload failed", storage_type="gcs", operation="upload")

        assert error.message == "Upload failed"
        assert error.context["storage_type"] == "gcs"
        assert error.context["operation"] == "upload"


class TestRetryExhaustedError:
    """Test RetryExhaustedError class."""

    def test_retry_exhausted_with_attempts(self):
        """Test retry exhausted error with attempt count."""
        original_error = ValueError("Original error")
        error = RetryExhaustedError(
            "All retries failed", attempts=5, last_error=original_error
        )

        assert error.message == "All retries failed"
        assert error.context["attempts"] == 5
        assert error.context["last_error"] == "Original error"
        assert error.context["last_error_type"] == "ValueError"


class TestCSVParsingError:
    """Test CSVParsingError class."""

    def test_csv_parsing_error_with_line(self):
        """Test CSV parsing error with line number."""
        error = CSVParsingError(
            "Invalid CSV format", csv_path="/path/to/file.csv", line_number=42
        )

        assert error.message == "Invalid CSV format"
        assert error.context["csv_path"] == "/path/to/file.csv"
        assert error.context["line_number"] == 42


class TestTimeoutError:
    """Test TimeoutError class."""

    def test_timeout_error_with_operation(self):
        """Test timeout error with operation details."""
        error = TimeoutError(
            "Operation timed out", operation="download", timeout_seconds=30.0
        )

        assert error.message == "Operation timed out"
        assert error.context["operation"] == "download"
        assert error.context["timeout_seconds"] == 30.0


class TestErrorFormatting:
    """Test error message formatting."""

    def test_context_formatting(self):
        """Test that context is properly formatted in error messages."""
        error = BulkImageProcessorError(
            "Test error", {"key1": "value1", "key2": 42, "key3": True}
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert "key1=value1" in error_str
        assert "key2=42" in error_str
        assert "key3=True" in error_str

    def test_empty_context_formatting(self):
        """Test error formatting with empty context."""
        error = BulkImageProcessorError("Simple error", {})
        assert str(error) == "Simple error"

    def test_none_context_formatting(self):
        """Test error formatting with None context."""
        error = BulkImageProcessorError("Simple error", None)
        assert str(error) == "Simple error"
