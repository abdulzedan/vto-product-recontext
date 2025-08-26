"""Base processor class for image processing operations.""" 
#NOTE: take note of packaging imports  

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from ..analyzer import GeminiAnalyzer
from ..config import Settings
from ..downloader import ImageRecord
from ..utils import ensure_directory, generate_unique_id, save_image_with_metadata

logger = structlog.get_logger(__name__)


class ProcessingResult(BaseModel):
    """Result of image processing operation."""
    
    record_id: str
    success: bool
    output_path: Optional[Path] = None
    gcs_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0
    retry_count: int = 0
    quality_score: Optional[float] = None
    feedback: Optional[str] = None


class BaseProcessor(ABC):
    """Abstract base class for image processors."""
    
    def __init__(self, settings: Settings, analyzer: GeminiAnalyzer):
        self.settings = settings
        self.analyzer = analyzer
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'retry_attempts': 0,
            'total_processing_time': 0.0,
        }
    
    @abstractmethod
    async def process_image(
        self,
        record: ImageRecord,
        image_path: Path,
        output_dir: Path,
    ) -> ProcessingResult:
        """Process a single image."""
        pass
    
    @abstractmethod
    def get_processor_type(self) -> str:
        """Get the processor type name."""
        pass
    
    def create_output_directory(self, base_dir: Path, record: ImageRecord) -> Path:
        """Create a unique output directory for the record with microseconds to prevent race conditions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_name = f"{timestamp}_{record.id}"
        output_dir = base_dir / dir_name
        ensure_directory(output_dir)
        return output_dir
    
    def prepare_metadata(
        self,
        record: ImageRecord,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare metadata for the processing result."""
        metadata = {
            'record_id': record.id,
            'original_url': record.image_url,
            'image_command': record.image_command,
            'image_position': record.image_position,
            'processor_type': self.get_processor_type(),
            'timestamp': datetime.now().isoformat(),
            'settings': {
                'max_retries': self.settings.processing.max_retries,
                'processing_timeout': self.settings.processing.processing_timeout,
            },
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def update_stats(self, success: bool, processing_time: float, retry_count: int = 0) -> None:
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['retry_attempts'] += retry_count
        
        if success:
            self.processing_stats['successful'] += 1
        else:
            self.processing_stats['failed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_processed'] * 100
            stats['average_processing_time'] = stats['total_processing_time'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['average_processing_time'] = 0.0
        
        return stats
    
    async def process_with_retry(
        self,
        record: ImageRecord,
        image_path: Path,
        output_dir: Path,
    ) -> ProcessingResult:
        """Process image with retry logic."""
        max_retries = self.settings.processing.max_retries
        last_error = None
        
        for attempt in range(max_retries + 1):
            start_time = time.time()
            
            try:
                logger.info(
                    "Processing image",
                    record_id=record.id,
                    processor=self.get_processor_type(),
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                
                result = await self.process_image(record, image_path, output_dir)
                result.retry_count = attempt
                
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                
                self.update_stats(result.success, processing_time, attempt)
                
                if result.success:
                    logger.info(
                        "Image processing successful",
                        record_id=record.id,
                        processor=self.get_processor_type(),
                        attempt=attempt + 1,
                        processing_time=round(processing_time, 2),
                        quality_score=result.quality_score,
                    )
                    return result
                else:
                    logger.warning(
                        "Image processing failed",
                        record_id=record.id,
                        processor=self.get_processor_type(),
                        attempt=attempt + 1,
                        error=result.error_message,
                    )
                    
                    if attempt < max_retries:
                        # Add delay before retry
                        delay = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                        logger.info(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    
                    last_error = result.error_message
                    
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                
                logger.error(
                    "Image processing exception",
                    record_id=record.id,
                    processor=self.get_processor_type(),
                    attempt=attempt + 1,
                    error=error_msg,
                    processing_time=round(processing_time, 2),
                )
                
                if attempt < max_retries:
                    # Add delay before retry
                    delay = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                
                last_error = error_msg
        
        # All retries failed
        final_processing_time = time.time() - start_time
        self.update_stats(False, final_processing_time, max_retries)
        
        return ProcessingResult(
            record_id=record.id,
            success=False,
            error_message=f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
            processing_time=final_processing_time,
            retry_count=max_retries,
        )
    
    def validate_image(self, image_path: Path) -> bool:
        """Validate that the image file is valid and processable."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(
                "Image validation failed",
                path=str(image_path),
                error=str(e),
            )
            return False
    
    def log_processing_summary(self) -> None:
        """Log a summary of processing statistics."""
        stats = self.get_stats()
        
        logger.info(
            "Processing summary",
            processor=self.get_processor_type(),
            **stats,
        )


# Import asyncio for sleep
import asyncio