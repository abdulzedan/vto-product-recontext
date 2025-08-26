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
        """Process images from CSV file with run tracking."""
        from .utils import get_next_run_id, create_run_directory, create_run_manifest, generate_run_statistics
        
        start_time = time.time()
        
        # Create new run directory
        run_id = get_next_run_id()
        run_path = create_run_directory(run_id)
        
        # Create manifest
        manifest = create_run_manifest(run_id, run_path, csv_path, self.settings)
        
        logger.info(
            "Starting bulk image processing",
            run_id=run_id,
            run_path=str(run_path),
            csv_path=str(csv_path),
            max_workers=self.settings.processing.max_workers,
        )
        
        # Store run info for later use
        self.current_run_id = run_id
        self.current_run_path = run_path
        
        try:
            # Step 1: Download images from CSV
            logger.info("Step 1: Downloading images from CSV")
            downloads_dir = run_path / "downloads"
            download_results = await download_images_from_csv(
                csv_path=csv_path,
                output_dir=downloads_dir,
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
            
            # Step 3: Process images based on classification (using run-specific output)
            logger.info("Step 3: Processing images")
            processing_results = await self._process_classified_images(classified_images, run_path)
            
            # Step 4: Generate run statistics
            total_duration = time.time() - start_time
            logger.info("Step 4: Generating run statistics")
            
            # Convert processing results to dict format for statistics
            results_for_stats = []
            for result in processing_results:
                results_for_stats.append({
                    'success': result.success,
                    'processor_type': getattr(result, 'processor_type', 'unknown'),
                    'quality_score': getattr(result, 'quality_score', 0.0),
                    'processing_time': getattr(result, 'processing_time', 0.0),
                    'error_message': getattr(result, 'error_message', ''),
                    'attempts': getattr(result, 'retry_count', 1),
                })
            
            statistics = generate_run_statistics(run_path, results_for_stats, total_duration)
            
            # Step 5: Generate summary (for backward compatibility)
            summary = self._create_summary(
                start_time,
                download_results,
                classified_images,
                processing_results,
            )
            summary['run_id'] = run_id
            summary['run_path'] = str(run_path)
            summary['statistics'] = statistics
            
            # Save summary in run directory
            self.storage_manager.create_processing_summary(
                summary,
                "bulk_processing",
                run_path,
            )
            
            # Create results CSV in img_conversion_table format
            self.storage_manager.create_results_csv(
                processing_results,
                run_path,
                run_id,
            )
            
            # Step 6: Upload entire run directory to GCS
            if self.settings.storage.enable_gcs_upload:
                logger.info("Step 6: Uploading complete run directory to GCS")
                await self._upload_run_directory_to_gcs(run_path)
            
            logger.info(
                "Bulk processing completed",
                run_id=run_id,
                total_time=round(total_duration, 2),
                total_processed=len(processing_results),
                successful_processed=sum(1 for r in processing_results if r.success),
                success_rate=f"{statistics['results']['success_rate']:.1%}",
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
        progress = ProgressTracker(len(download_results))
        
        # Use asyncio semaphore for concurrent classification tasks
        semaphore = asyncio.Semaphore(self.settings.processing.max_workers)
        
        async def classify_with_semaphore(record: ImageRecord, path: Path) -> Tuple[ImageRecord, Path, ImageCategory]:
            async with semaphore:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, skipping classification")
                    return (record, path, ImageCategory.UNKNOWN)
                
                try:
                    category = await self._classify_single_image(record, path)
                    progress.update(success=True)
                    return (record, path, category)
                except Exception as e:
                    logger.error(
                        "Classification failed",
                        record_id=record.id,
                        error=str(e),
                    )
                    progress.update(success=False)
                    # Default to unknown category on failure
                    return (record, path, ImageCategory.UNKNOWN)
        
        # Create classification tasks
        tasks = [
            classify_with_semaphore(record, path)
            for record, path in download_results
        ]
        
        # Use asyncio.gather for true parallel processing
        logger.info(f"Starting parallel classification of {len(tasks)} images")
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Filter out any None results
        classified_images = [r for r in results if r is not None]
        
        progress.log_progress()
        return classified_images
    
    async def _classify_single_image(self, record: ImageRecord, path: Path) -> ImageCategory:
        """Classify a single image."""
        try:
            result = await self.analyzer.classify_image(
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
        run_path: Path = None,
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
        
        # Use run-specific output directory or default
        if run_path is None:
            run_path = Path("./output")
        
        # Process apparel images with Virtual Try-On
        if apparel_images:
            logger.info("Processing apparel images with Virtual Try-On")
            vto_output_dir = run_path / "virtual_try_on"
            vto_results = await self._process_with_processor(
                apparel_images,
                self.vto_processor,
                "virtual_try_on",
                vto_output_dir,
            )
            results.extend(vto_results)
        
        # Process product images with Product Recontext
        if product_images:
            logger.info("Processing product images with Product Recontext")
            product_output_dir = run_path / "product_recontext"
            product_results = await self._process_with_processor(
                product_images,
                self.product_processor,
                "product_recontext",
                product_output_dir,
            )
            results.extend(product_results)
        
        # Handle unknown images (default to Product Recontext)
        if unknown_images:
            logger.info("Processing unknown images with Product Recontext (default)")
            unknown_output_dir = run_path / "product_recontext"
            unknown_results = await self._process_with_processor(
                unknown_images,
                self.product_processor,
                "product_recontext",
                unknown_output_dir,
            )
            results.extend(unknown_results)
        
        return results
    
    async def _process_with_processor(
        self,
        images: List[Tuple[ImageRecord, Path]],
        processor,
        processor_name: str,
        output_dir: Path = None,
    ) -> List[ProcessingResult]:
        """Process images with specified processor."""
        if output_dir is None:
            output_dir = self.storage_manager.get_local_output_dir(processor_name)
        progress = ProgressTracker(len(images))
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.settings.processing.max_workers)
        
        async def process_with_semaphore(record: ImageRecord, path: Path) -> ProcessingResult:
            async with semaphore:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, skipping processing")
                    return ProcessingResult(
                        record_id=record.id,
                        success=False,
                        error_message="Shutdown requested",
                    )
                
                try:
                    result = await processor.process_with_retry(record, path, output_dir)
                    progress.update(success=result.success)
                    return result
                except Exception as e:
                    logger.error("Processing task failed", record_id=record.id, error=str(e))
                    progress.update(success=False)
                    return ProcessingResult(
                        record_id=record.id,
                        success=False,
                        error_message=str(e),
                    )
        
        # Process images concurrently using gather
        tasks = [
            process_with_semaphore(record, path)
            for record, path in images
        ]
        
        logger.info(f"Starting parallel processing of {len(tasks)} images with {processor_name}")
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
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
    
    async def _upload_run_directory_to_gcs(self, run_path: Path) -> None:
        """Upload entire run directory to GCS preserving the same structure."""
        try:
            logger.info("Starting GCS upload", run_path=str(run_path))
            from google.cloud import storage
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.settings.google_cloud.storage_bucket)
            logger.info("GCS client and bucket initialized", bucket_name=self.settings.google_cloud.storage_bucket)
            
            # Extract the relative path from 'output/' onwards
            # run_path is like: output/2025/08/13/run_002
            base_output_dir = Path("./output")
            relative_run_path = run_path.relative_to(base_output_dir)
            
            # Collect all files to upload
            files_to_upload = []
            total_size = 0
            
            logger.info("Starting file enumeration", base_dir=str(base_output_dir))
            for local_file_path in run_path.rglob('*'):
                if local_file_path.is_file():
                    # Calculate relative path from the run directory
                    relative_file_path = local_file_path.relative_to(base_output_dir)
                    
                    # Create GCS path: output/2025/08/13/run_002/virtual_try_on/...
                    gcs_path = f"output/{relative_file_path}"
                    
                    files_to_upload.append((local_file_path, gcs_path))
                    total_size += local_file_path.stat().st_size
            
            logger.info(f"Found {len(files_to_upload)} files to upload, total size: {total_size / (1024*1024):.2f} MB")
            
            # Create upload function for concurrent uploads
            async def upload_file_async(local_path: Path, gcs_path: str, semaphore: asyncio.Semaphore) -> bool:
                async with semaphore:
                    try:
                        # Run blocking upload in executor
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: bucket.blob(gcs_path).upload_from_filename(str(local_path))
                        )
                        return True
                    except Exception as e:
                        logger.error(f"Failed to upload {local_path} to {gcs_path}: {e}")
                        return False
            
            # Use semaphore to limit concurrent uploads (avoid overwhelming GCS)
            semaphore = asyncio.Semaphore(min(10, self.settings.processing.max_workers))
            
            # Create upload tasks
            upload_tasks = [
                upload_file_async(local_path, gcs_path, semaphore)
                for local_path, gcs_path in files_to_upload
            ]
            
            # Execute all uploads concurrently
            logger.info(f"Starting concurrent upload of {len(upload_tasks)} files")
            results = await asyncio.gather(*upload_tasks, return_exceptions=False)
            
            # Count successful uploads
            uploaded_files = sum(1 for r in results if r)
            failed_files = len(results) - uploaded_files
            
            # Calculate total size in MB
            total_size_mb = total_size / (1024 * 1024)
            
            logger.info(
                "Complete run directory uploaded to GCS",
                run_path=str(run_path),
                gcs_base_path=f"gs://{self.settings.google_cloud.storage_bucket}/output/{relative_run_path}",
                uploaded_files=uploaded_files,
                failed_files=failed_files,
                total_size_mb=round(total_size_mb, 2),
            )
            
        except Exception as e:
            logger.error(
                "Failed to upload run directory to GCS",
                run_path=str(run_path),
                error=str(e),
            )
    
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