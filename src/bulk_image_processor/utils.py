"""Utility functions for the bulk image processor."""

import asyncio
import base64
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import structlog
from PIL import Image
from google.cloud import storage
from google.cloud.aiplatform.gapic import PredictResponse

logger = structlog.get_logger(__name__)


def setup_logging(level: str = "INFO", format_type: str = "json") -> None:
    """Set up structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if format_type == "json"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
    )


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{prefix}_{timestamp}" if prefix else timestamp


def validate_image_url(url: str) -> bool:
    """Validate if a URL is a valid image URL."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Check if URL ends with common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        path_lower = parsed.path.lower()
        return any(path_lower.endswith(ext) for ext in image_extensions)
    except Exception:
        return False


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def prediction_to_pil_image(
    prediction: Dict[str, Any], 
    size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """Convert prediction response to PIL Image."""
    if "bytesBase64Encoded" not in prediction:
        raise ValueError("Prediction does not contain bytesBase64Encoded field")
    
    encoded_bytes_string = prediction["bytesBase64Encoded"]
    decoded_image_bytes = base64.b64decode(encoded_bytes_string)
    image_pil = Image.open(io.BytesIO(decoded_image_bytes))
    
    if size:
        image_pil.thumbnail(size)
    
    return image_pil


def prepare_image_for_api(image_path: Path, max_size: Tuple[int, int] = (1024, 1024)) -> str:
    """Prepare image file for API consumption."""
    with open(image_path, 'rb') as f:
        raw_image_bytes = f.read()
    
    # Process the image - convert to RGB and resize if needed
    image_pil = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
    original_size = image_pil.size
    
    # Apply thumbnail to maintain aspect ratio within max_size
    image_pil.thumbnail(max_size)
    
    logger.info(
        "Image prepared for API",
        original_size=original_size,
        processed_size=image_pil.size,
        path=str(image_path),
    )
    
    # Encode the processed image
    return encode_image_to_base64(image_pil)


def save_image_with_metadata(
    image: Image.Image,
    output_path: Path,
    metadata: Dict[str, Any],
    include_timestamp: bool = True,
) -> Tuple[Path, Path]:
    """Save image and metadata to specified path."""
    ensure_directory(output_path.parent)
    
    # Add timestamp to filename if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{output_path.stem}_{timestamp}"
        output_path = output_path.with_stem(stem)
    
    # Save image
    image.save(output_path)
    
    # Save metadata as JSON
    metadata_path = output_path.with_suffix('.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return output_path, metadata_path


def upload_to_gcs(
    local_path: Path,
    bucket_name: str,
    gcs_path: str,
    storage_client: Optional[storage.Client] = None,
) -> str:
    """Upload file to Google Cloud Storage."""
    if storage_client is None:
        storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    blob.upload_from_filename(str(local_path))
    
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    logger.info(
        "File uploaded to GCS",
        local_path=str(local_path),
        gcs_uri=gcs_uri,
    )
    
    return gcs_uri


def download_from_gcs(
    gcs_uri: str,
    local_path: Path,
    storage_client: Optional[storage.Client] = None,
) -> None:
    """Download file from Google Cloud Storage."""
    if storage_client is None:
        storage_client = storage.Client()
    
    # Parse GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    parts = gcs_uri[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
    
    bucket_name, object_name = parts
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    
    ensure_directory(local_path.parent)
    blob.download_to_filename(str(local_path))
    
    logger.info(
        "File downloaded from GCS",
        gcs_uri=gcs_uri,
        local_path=str(local_path),
    )


def retry_with_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Any:
    """Retry function with exponential backoff."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            logger.warning(
                "Function failed, retrying",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(e),
            )
            time.sleep(delay)
    
    raise last_exception


async def async_retry_with_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Any:
    """Async retry function with exponential backoff."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            logger.warning(
                "Async function failed, retrying",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(e),
            )
            await asyncio.sleep(delay)
    
    raise last_exception


def calculate_processing_stats(
    start_time: float,
    end_time: float,
    total_items: int,
    successful_items: int,
    failed_items: int,
) -> Dict[str, Any]:
    """Calculate processing statistics."""
    duration = end_time - start_time
    
    return {
        "duration_seconds": round(duration, 2),
        "total_items": total_items,
        "successful_items": successful_items,
        "failed_items": failed_items,
        "success_rate": round(successful_items / max(total_items, 1) * 100, 2),
        "items_per_second": round(total_items / max(duration, 0.001), 2),
        "average_time_per_item": round(duration / max(total_items, 1), 2),
    }


class ProgressTracker:
    """Track processing progress."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        
    def update(self, success: bool = True) -> None:
        """Update progress."""
        self.completed_items += 1
        if not success:
            self.failed_items += 1
        
        # Log progress every 10 items or 30 seconds
        current_time = time.time()
        if (
            self.completed_items % 10 == 0 
            or current_time - self.last_update > 30
        ):
            self.log_progress()
            self.last_update = current_time
    
    def log_progress(self) -> None:
        """Log current progress."""
        elapsed = time.time() - self.start_time
        rate = self.completed_items / max(elapsed, 0.001)
        remaining = self.total_items - self.completed_items
        eta = remaining / max(rate, 0.001)
        
        logger.info(
            "Processing progress",
            completed=self.completed_items,
            total=self.total_items,
            failed=self.failed_items,
            progress_percent=round(self.completed_items / self.total_items * 100, 1),
            rate_per_second=round(rate, 2),
            eta_seconds=round(eta, 1),
        )
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Get final processing statistics."""
        return calculate_processing_stats(
            start_time=self.start_time,
            end_time=time.time(),
            total_items=self.total_items,
            successful_items=self.completed_items - self.failed_items,
            failed_items=self.failed_items,
        )