"""Custom exceptions for the bulk image processor.

This module defines specific exception types for different error scenarios
that can occur during image processing, providing better error handling
and more informative error messages.
"""

from typing import Optional, Dict, Any


class BulkImageProcessorError(Exception):
    """Base exception for all bulk image processor errors.
    
    This is the base exception class that all other custom exceptions
    in this module inherit from. It provides common functionality for
    storing error context and formatting error messages.
    
    Attributes:
        message: The error message
        context: Optional dictionary containing additional error context
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Initialize the exception with a message and optional context.
        
        Args:
            message: The error message
            context: Optional dictionary containing additional error context
                    such as file paths, URLs, or processing parameters
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format the error message with context information.
        
        Returns:
            Formatted error message including context if available
        """
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ConfigurationError(BulkImageProcessorError):
    """Raised when there's an error in configuration settings.
    
    This exception is raised when:
    - Required configuration values are missing
    - Configuration values are invalid
    - Environment variables are not properly set
    - Configuration files cannot be loaded
    """
    pass


class DownloadError(BulkImageProcessorError):
    """Raised when image download fails.
    
    This exception is raised when:
    - Network connection fails
    - URL is invalid or unreachable
    - Download timeout occurs
    - Response status is not successful
    - Downloaded content is not a valid image
    """
    
    def __init__(self, message: str, url: Optional[str] = None, 
                 status_code: Optional[int] = None, **kwargs):
        """Initialize download error with URL and status code.
        
        Args:
            message: The error message
            url: The URL that failed to download
            status_code: HTTP status code if available
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if url:
            context['url'] = url
        if status_code:
            context['status_code'] = status_code
        super().__init__(message, context)


class ImageValidationError(BulkImageProcessorError):
    """Raised when image validation fails.
    
    This exception is raised when:
    - Image file is corrupted
    - Image format is not supported
    - Image dimensions exceed limits
    - Image file size exceeds limits
    - Security validation fails
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 validation_type: Optional[str] = None, **kwargs):
        """Initialize validation error with file path and validation type.
        
        Args:
            message: The error message
            file_path: Path to the image file that failed validation
            validation_type: Type of validation that failed (e.g., 'format', 'size')
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        if validation_type:
            context['validation_type'] = validation_type
        super().__init__(message, context)


class ProcessingError(BulkImageProcessorError):
    """Raised when image processing fails.
    
    This exception is raised when:
    - Virtual Try-On processing fails
    - Product Recontext processing fails
    - Model endpoint is unreachable
    - Processing timeout occurs
    - Output generation fails
    """
    
    def __init__(self, message: str, processor_type: Optional[str] = None,
                 record_id: Optional[str] = None, **kwargs):
        """Initialize processing error with processor type and record ID.
        
        Args:
            message: The error message
            processor_type: Type of processor that failed (e.g., 'virtual_try_on')
            record_id: ID of the record being processed
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if processor_type:
            context['processor_type'] = processor_type
        if record_id:
            context['record_id'] = record_id
        super().__init__(message, context)


class AnalysisError(BulkImageProcessorError):
    """Raised when Gemini analysis fails.
    
    This exception is raised when:
    - Gemini API is unreachable
    - API quota is exceeded
    - Image classification fails
    - Quality assessment fails
    - API returns invalid response
    """
    
    def __init__(self, message: str, analysis_type: Optional[str] = None,
                 api_error: Optional[str] = None, **kwargs):
        """Initialize analysis error with analysis type and API error.
        
        Args:
            message: The error message
            analysis_type: Type of analysis that failed (e.g., 'classification')
            api_error: Error message from the API if available
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if analysis_type:
            context['analysis_type'] = analysis_type
        if api_error:
            context['api_error'] = api_error
        super().__init__(message, context)


class StorageError(BulkImageProcessorError):
    """Raised when storage operations fail.
    
    This exception is raised when:
    - Local file system operations fail
    - Google Cloud Storage operations fail
    - Insufficient disk space
    - Permission denied errors
    - File already exists conflicts
    """
    
    def __init__(self, message: str, storage_type: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize storage error with storage type and operation.
        
        Args:
            message: The error message
            storage_type: Type of storage (e.g., 'local', 'gcs')
            operation: Operation that failed (e.g., 'upload', 'download')
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if storage_type:
            context['storage_type'] = storage_type
        if operation:
            context['operation'] = operation
        super().__init__(message, context)


class RetryExhaustedError(BulkImageProcessorError):
    """Raised when all retry attempts have been exhausted.
    
    This exception is raised when:
    - Maximum retry attempts reached
    - All retry attempts failed
    - Backoff period exceeded maximum
    
    This exception typically wraps the last error that occurred.
    """
    
    def __init__(self, message: str, attempts: Optional[int] = None,
                 last_error: Optional[Exception] = None, **kwargs):
        """Initialize retry exhausted error.
        
        Args:
            message: The error message
            attempts: Number of attempts made
            last_error: The last exception that occurred
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if attempts:
            context['attempts'] = attempts
        if last_error:
            context['last_error'] = str(last_error)
            context['last_error_type'] = type(last_error).__name__
        super().__init__(message, context)


class CSVParsingError(BulkImageProcessorError):
    """Raised when CSV file parsing fails.
    
    This exception is raised when:
    - CSV file is not found
    - CSV format is invalid
    - Required columns are missing
    - Data validation fails
    """
    
    def __init__(self, message: str, csv_path: Optional[str] = None,
                 line_number: Optional[int] = None, **kwargs):
        """Initialize CSV parsing error.
        
        Args:
            message: The error message
            csv_path: Path to the CSV file
            line_number: Line number where error occurred
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if csv_path:
            context['csv_path'] = csv_path
        if line_number:
            context['line_number'] = line_number
        super().__init__(message, context)


class TimeoutError(BulkImageProcessorError):
    """Raised when an operation times out.
    
    This exception is raised when:
    - Download timeout occurs
    - Processing timeout occurs
    - API request timeout occurs
    """
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 timeout_seconds: Optional[float] = None, **kwargs):
        """Initialize timeout error.
        
        Args:
            message: The error message
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            **kwargs: Additional context to pass to parent
        """
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        super().__init__(message, context)