"""Main orchestrator for bulk image processing."""

import asyncio
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from tqdm import tqdm

from .analyzer import GeminiAnalyzer, ImageCategory
from .config import Settings, get_settings
from .downloader import ImageDownloader, ImageRecord, download_images_from_csv
from .processors import VirtualTryOnProcessor, ProductRecontextProcessor, ProcessingResult
from .storage import StorageManager
from .utils import ProgressTracker, setup_logging

logger = structlog.get_logger(__name__)


class BulkImageProcessor:
    """Main orchestrator for bulk image processing."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.analyzer = None
        self.vto_processor = None
        self.product_processor = None
        self.storage_manager = None
        self.shutdown_requested = False
        self._setup_components()
        self._setup_signal_handlers()
    
    def _setup_components(self) -> None:
        """Initialize all components."""
        try:
            # Setup logging
            setup_logging(
                level=self.settings.logging.level,
                format_type=self.settings.logging.format,
            )
            
            # Initialize components
            self.analyzer = GeminiAnalyzer(self.settings)
            self.vto_processor = VirtualTryOnProcessor(self.settings, self.analyzer)
            self.product_processor = ProductRecontextProcessor(self.settings, self.analyzer)
            self.storage_manager = StorageManager(self.settings)
            
            logger.info(
                "Components initialized successfully",
                project_id=self.settings.google_cloud.project_id,
                max_workers=self.settings.processing.max_workers,
            )
            
        except Exception as e:
            logger.error("Failed to initialize components", error=str(e))
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def process_from_csv(self, csv_path: Path) -> Dict[str, any]:
        """Process images from CSV file."""
        start_time = time.time()
        
        logger.info(
            "Starting bulk image processing",
            csv_path=str(csv_path),
            max_workers=self.settings.processing.max_workers,
        )
        
        try:
            # Step 1: Download images from CSV
            logger.info("Step 1: Downloading images from CSV")
            download_results = await download_images_from_csv(
                csv_path=csv_path,
                output_dir=Path("./downloads"),
                settings=self.settings,
            )
            
            # Filter successful downloads
            successful_downloads = [
                (record, path) for record, path in download_results
                if path is not None
            ]
            
            logger.info(
                "Image download completed",
                total_records=len(download_results),
                successful_downloads=len(successful_downloads),
                failed_downloads=len(download_results) - len(successful_downloads),
            )
            
            if not successful_downloads:
                logger.error("No images were successfully downloaded")
                return self._create_summary(start_time, [], [], [])
            
            # Step 2: Classify images
            logger.info("Step 2: Classifying images with Gemini")
            classified_images = await self._classify_images(successful_downloads)
            
            # Step 3: Process images based on classification
            logger.info("Step 3: Processing images")
            processing_results = await self._process_classified_images(classified_images)
            
            # Step 4: Generate summary
            summary = self._create_summary(
                start_time,
                download_results,
                classified_images,
                processing_results,
            )
            
            # Save summary
            self.storage_manager.create_processing_summary(
                summary,
                "bulk_processing",
            )
            
            logger.info(
                "Bulk processing completed",
                total_time=round(time.time() - start_time, 2),
                total_processed=len(processing_results),
                successful_processed=sum(1 for r in processing_results if r.success),
            )
            
            return summary
            
        except Exception as e:
            logger.error("Bulk processing failed", error=str(e))
            raise
    
    async def _classify_images(
        self,
        download_results: List[Tuple[ImageRecord, Path]],
    ) -> List[Tuple[ImageRecord, Path, ImageCategory]]:
        """Classify images using Gemini analyzer."""
        classified_images = []
        progress = ProgressTracker(len(download_results))
        
        # Use ThreadPoolExecutor for CPU-bound classification tasks
        with ThreadPoolExecutor(max_workers=self.settings.processing.max_workers) as executor:
            # Submit classification tasks
            future_to_record = {
                executor.submit(self._classify_single_image, record, path): (record, path)
                for record, path in download_results
            }
            
            # Process completed tasks
            for future in as_completed(future_to_record):
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping classification")
                    break
                
                record, path = future_to_record[future]
                
                try:
                    category = future.result()
                    classified_images.append((record, path, category))
                    progress.update(success=True)
                    
                except Exception as e:
                    logger.error(
                        "Classification failed",
                        record_id=record.id,
                        error=str(e),
                    )
                    # Default to unknown category on failure
                    classified_images.append((record, path, ImageCategory.UNKNOWN))
                    progress.update(success=False)
        
        progress.log_progress()
        return classified_images
    
    def _classify_single_image(self, record: ImageRecord, path: Path) -> ImageCategory:
        """Classify a single image."""
        try:
            result = self.analyzer.classify_image(
                path,
                additional_context=f"Command: {record.image_command}, Position: {record.image_position}",
            )
            
            logger.info(
                "Image classified",
                record_id=record.id,
                category=result.category,
                confidence=result.confidence,
            )
            
            return result.category
            
        except Exception as e:
            logger.error(
                "Image classification failed",
                record_id=record.id,
                error=str(e),
            )
            return ImageCategory.UNKNOWN
    
    async def _process_classified_images(
        self,
        classified_images: List[Tuple[ImageRecord, Path, ImageCategory]],
    ) -> List[ProcessingResult]:
        """Process classified images using appropriate processors."""
        # Separate images by category
        apparel_images = [
            (record, path) for record, path, category in classified_images
            if category == ImageCategory.APPAREL
        ]
        
        product_images = [
            (record, path) for record, path, category in classified_images
            if category == ImageCategory.PRODUCT
        ]
        
        unknown_images = [
            (record, path) for record, path, category in classified_images
            if category == ImageCategory.UNKNOWN
        ]
        
        logger.info(
            "Processing categories",
            apparel_count=len(apparel_images),
            product_count=len(product_images),
            unknown_count=len(unknown_images),
        )
        
        # Process each category
        results = []
        
        # Process apparel images with Virtual Try-On
        if apparel_images:
            logger.info("Processing apparel images with Virtual Try-On")
            vto_results = await self._process_with_processor(
                apparel_images,
                self.vto_processor,
                "virtual_try_on",
            )
            results.extend(vto_results)
        
        # Process product images with Product Recontext
        if product_images:
            logger.info("Processing product images with Product Recontext")
            product_results = await self._process_with_processor(
                product_images,
                self.product_processor,
                "product_recontext",
            )
            results.extend(product_results)
        
        # Handle unknown images (default to Product Recontext)
        if unknown_images:
            logger.info("Processing unknown images with Product Recontext (default)")
            unknown_results = await self._process_with_processor(
                unknown_images,
                self.product_processor,
                "product_recontext",
            )
            results.extend(unknown_results)
        
        return results
    
    async def _process_with_processor(
        self,
        images: List[Tuple[ImageRecord, Path]],
        processor,
        processor_name: str,
    ) -> List[ProcessingResult]:
        """Process images with specified processor."""
        output_dir = self.storage_manager.get_local_output_dir(processor_name)
        results = []
        progress = ProgressTracker(len(images))
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.settings.processing.max_workers)
        
        async def process_with_semaphore(record: ImageRecord, path: Path) -> ProcessingResult:
            async with semaphore:
                return await processor.process_with_retry(record, path, output_dir)
        
        # Process images concurrently
        tasks = [
            process_with_semaphore(record, path)
            for record, path in images
        ]
        
        # Wait for all tasks to complete
        for task in asyncio.as_completed(tasks):
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping processing")
                break
            
            try:
                result = await task
                results.append(result)
                progress.update(success=result.success)
                
            except Exception as e:
                logger.error("Processing task failed", error=str(e))
                progress.update(success=False)
        
        progress.log_progress()
        processor.log_processing_summary()
        
        return results
    
    def _create_summary(
        self,
        start_time: float,
        download_results: List[Tuple[ImageRecord, Optional[Path]]],
        classified_images: List[Tuple[ImageRecord, Path, ImageCategory]],
        processing_results: List[ProcessingResult],
    ) -> Dict[str, Any]:
        """Create processing summary."""
        end_time = time.time()
        total_time = end_time - start_time
        
        # Download statistics
        download_stats = {
            'total_records': len(download_results),
            'successful_downloads': sum(1 for _, path in download_results if path is not None),
            'failed_downloads': sum(1 for _, path in download_results if path is None),
        }
        
        # Classification statistics
        classification_stats = {
            'total_classified': len(classified_images),
            'apparel_count': sum(1 for _, _, cat in classified_images if cat == ImageCategory.APPAREL),
            'product_count': sum(1 for _, _, cat in classified_images if cat == ImageCategory.PRODUCT),
            'unknown_count': sum(1 for _, _, cat in classified_images if cat == ImageCategory.UNKNOWN),
        }
        
        # Processing statistics
        processing_stats = {
            'total_processed': len(processing_results),
            'successful_processed': sum(1 for r in processing_results if r.success),
            'failed_processed': sum(1 for r in processing_results if not r.success),
            'vto_processor_stats': self.vto_processor.get_stats(),
            'product_processor_stats': self.product_processor.get_stats(),
        }
        
        # Quality statistics
        quality_scores = [r.quality_score for r in processing_results if r.quality_score is not None]
        quality_stats = {
            'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'min_quality_score': min(quality_scores) if quality_scores else 0,
            'max_quality_score': max(quality_scores) if quality_scores else 0,
        }
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_processing_time': round(total_time, 2),
            'settings': {
                'max_workers': self.settings.processing.max_workers,
                'max_retries': self.settings.processing.max_retries,
                'project_id': self.settings.google_cloud.project_id,
            },
            'download_stats': download_stats,
            'classification_stats': classification_stats,
            'processing_stats': processing_stats,
            'quality_stats': quality_stats,
            'storage_stats': self.storage_manager.get_storage_stats(),
        }
        
        return summary
    
    def get_system_status(self) -> Dict[str, any]:
        """Get current system status."""
        return {
            'initialized': all([
                self.analyzer is not None,
                self.vto_processor is not None,
                self.product_processor is not None,
                self.storage_manager is not None,
            ]),
            'settings': {
                'project_id': self.settings.google_cloud.project_id,
                'location': self.settings.google_cloud.location,
                'max_workers': self.settings.processing.max_workers,
                'gcs_enabled': self.settings.storage.enable_gcs_upload,
            },
            'processors': {
                'vto_stats': self.vto_processor.get_stats() if self.vto_processor else None,
                'product_stats': self.product_processor.get_stats() if self.product_processor else None,
            },
            'storage': self.storage_manager.get_storage_stats() if self.storage_manager else None,
        }


async def main(csv_path: Optional[Path] = None) -> None:
    """Main entry point for the bulk image processor."""
    if csv_path is None:
        csv_path = Path("./image_folder/img_conversion_table.csv")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        processor = BulkImageProcessor()
        summary = await processor.process_from_csv(csv_path)
        
        logger.info(
            "Processing completed successfully",
            summary=summary,
        )
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Processing failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())