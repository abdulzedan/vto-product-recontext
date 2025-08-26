"""Pipeline-optimized main orchestrator for bulk image processing."""

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from asyncio import Queue

import structlog
from tqdm import tqdm

from .analyzer import GeminiAnalyzer, ImageCategory
from .config import Settings, get_settings
from .downloader import ImageDownloader, ImageRecord
from .processors import VirtualTryOnProcessor, ProductRecontextProcessor, ProcessingResult
from .storage import StorageManager
from .utils import (
    ProgressTracker, 
    setup_logging,
    get_next_run_id,
    create_run_directory,
    create_run_manifest,
    generate_run_statistics
)

logger = structlog.get_logger(__name__)


class PipelineBulkImageProcessor:
    """Pipeline-optimized orchestrator for bulk image processing."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.analyzer = None
        self.vto_processor = None
        self.product_processor = None
        self.storage_manager = None
        self.shutdown_requested = False
        self._setup_components()
        self._setup_signal_handlers()
        
        # Pipeline statistics
        self.pipeline_stats = {
            'downloads_completed': 0,
            'classifications_completed': 0,
            'processing_completed': 0,
            'first_result_time': None,
            'pipeline_start_time': None,
        }
    
    def _setup_components(self) -> None:
        """Initialize all components."""
        try:
            setup_logging(
                level=self.settings.logging.level,
                format_type=self.settings.logging.format,
            )
            
            self.analyzer = GeminiAnalyzer(self.settings)
            self.vto_processor = VirtualTryOnProcessor(self.settings, self.analyzer)
            self.product_processor = ProductRecontextProcessor(self.settings, self.analyzer)
            self.storage_manager = StorageManager(self.settings)
            
            logger.info(
                "Pipeline components initialized successfully",
                project_id=self.settings.google_cloud.project_id,
                max_workers=self.settings.processing.max_workers,
                pipeline_mode=True,
            )
            
        except Exception as e:
            logger.error("Failed to initialize pipeline components", error=str(e))
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def process_from_csv(self, csv_path: Path) -> Dict[str, any]:
        """Process images from CSV using pipeline approach."""
        start_time = time.time()
        self.pipeline_stats['pipeline_start_time'] = start_time
        
        # Create new run directory
        run_id = get_next_run_id()
        run_path = create_run_directory(run_id)
        
        # Create manifest
        manifest = create_run_manifest(run_id, run_path, csv_path, self.settings)
        
        logger.info(
            "Starting PIPELINE bulk image processing",
            run_id=run_id,
            run_path=str(run_path),
            csv_path=str(csv_path),
            max_workers=self.settings.processing.max_workers,
            pipeline_mode=True,
        )
        
        self.current_run_id = run_id
        self.current_run_path = run_path
        
        try:
            # Load CSV to get total count
            async with ImageDownloader(self.settings) as downloader:
                records = downloader.load_csv(csv_path)
                total_images = len(records)
            
            logger.info(f"Processing {total_images} images in pipeline mode")
            
            # Create processing queues with appropriate sizes
            download_queue = Queue(maxsize=min(total_images, 50))
            classification_queue = Queue(maxsize=min(total_images, 50))
            # Separate queues for each processor type to enable true parallelism
            vto_queue = Queue(maxsize=min(total_images, 50))
            product_queue = Queue(maxsize=min(total_images, 50))
            
            # Create lists to collect results
            all_downloads = []
            all_classifications = []
            all_results = []
            
            # Start pipeline stages concurrently
            tasks = [
                asyncio.create_task(
                    self._download_stage(csv_path, run_path, download_queue, all_downloads)
                ),
                asyncio.create_task(
                    self._classification_stage(download_queue, vto_queue, product_queue, all_classifications)
                ),
                asyncio.create_task(
                    self._vto_processing_stage(vto_queue, run_path, all_results)
                ),
                asyncio.create_task(
                    self._product_processing_stage(product_queue, run_path, all_results)
                ),
                asyncio.create_task(
                    self._progress_monitor(total_images)
                ),
            ]
            
            # Wait for all stages to complete
            await asyncio.gather(*tasks)
            
            # Generate statistics and summary
            total_duration = time.time() - start_time
            
            # Convert results for statistics
            results_for_stats = []
            for result in all_results:
                results_for_stats.append({
                    'success': result.success,
                    'processor_type': getattr(result, 'processor_type', 'unknown'),
                    'quality_score': getattr(result, 'quality_score', 0.0),
                    'processing_time': getattr(result, 'processing_time', 0.0),
                    'error_message': getattr(result, 'error_message', ''),
                    'attempts': getattr(result, 'retry_count', 1),
                })
            
            statistics = generate_run_statistics(run_path, results_for_stats, total_duration)
            
            # Calculate pipeline-specific metrics
            if self.pipeline_stats['first_result_time']:
                time_to_first_result = self.pipeline_stats['first_result_time'] - start_time
            else:
                time_to_first_result = None
            
            # Check for failed downloads report
            failed_downloads_info = None
            failed_downloads_path = run_path / "failed_downloads.json"
            if failed_downloads_path.exists():
                import json
                with open(failed_downloads_path, 'r') as f:
                    failed_downloads_info = json.load(f)
            
            summary = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_processing_time': round(total_duration, 2),
                'run_id': run_id,
                'run_path': str(run_path),
                'statistics': statistics,
                'pipeline_mode': True,
                'pipeline_metrics': {
                    'time_to_first_result': round(time_to_first_result, 2) if time_to_first_result else None,
                    'downloads_completed': self.pipeline_stats['downloads_completed'],
                    'classifications_completed': self.pipeline_stats['classifications_completed'],
                    'processing_completed': self.pipeline_stats['processing_completed'],
                },
                'failed_downloads': failed_downloads_info,
                'total_processed': len(all_results),
                'successful_processed': sum(1 for r in all_results if r.success),
            }
            
            # Save summary
            self.storage_manager.create_processing_summary(
                summary,
                "pipeline_processing",
                run_path,
            )
            
            # Create results CSV in img_conversion_table format
            self.storage_manager.create_results_csv(
                all_results,
                run_path,
                run_id,
            )
            
            # Upload to GCS if enabled
            if self.settings.storage.enable_gcs_upload:
                logger.info("Uploading complete run directory to GCS")
                await self._upload_run_directory_to_gcs(run_path)
            
            logger.info(
                "Pipeline processing completed",
                run_id=run_id,
                total_time=round(total_duration, 2),
                total_processed=len(all_results),
                time_to_first_result=round(time_to_first_result, 2) if time_to_first_result else "N/A",
            )
            
            return summary
            
        except Exception as e:
            logger.error("Pipeline processing failed", error=str(e))
            raise
    
    async def _download_stage(self, csv_path: Path, run_path: Path, output_queue: Queue, results_list: List):
        """Download stage - downloads images and immediately queues them."""
        logger.info("Starting download stage")
        downloads_dir = run_path / "downloads"
        
        # Track failed downloads
        failed_downloads = []
        
        async with ImageDownloader(self.settings) as downloader:
            records = downloader.load_csv(csv_path)
            total_records = len(records)
            
            # Limit concurrent downloads to avoid overwhelming the network
            semaphore = asyncio.Semaphore(min(self.settings.processing.max_workers, 30))
            
            async def download_and_queue(record: ImageRecord):
                if self.shutdown_requested:
                    return
                    
                async with semaphore:
                    try:
                        path = await downloader.download_single_image(record, downloads_dir)
                        if path:
                            await output_queue.put((record, path))
                            results_list.append((record, path))
                            self.pipeline_stats['downloads_completed'] += 1
                            logger.debug(f"Downloaded and queued: {record.id}")
                        else:
                            # Download returned None - track as failed
                            failed_downloads.append({
                                'record_id': record.id,
                                'url': record.image_url,
                                'error': 'Download returned None'
                            })
                            logger.warning(f"Download returned None for {record.id}")
                    except Exception as e:
                        # Track failed download with error details
                        error_msg = str(e)
                        if '404' in error_msg or 'Not Found' in error_msg:
                            error_type = 'Image not found (404)'
                        elif 'timeout' in error_msg.lower():
                            error_type = 'Download timeout'
                        elif '403' in error_msg:
                            error_type = 'Access forbidden (403)'
                        else:
                            error_type = 'Download error'
                        
                        failed_downloads.append({
                            'record_id': record.id,
                            'url': record.image_url,
                            'error': error_type,
                            'details': error_msg
                        })
                        logger.error(f"Download failed for {record.id}: {error_type} - {e}")
            
            # Start all downloads concurrently
            tasks = [download_and_queue(record) for record in records]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save failed downloads report if any
            if failed_downloads:
                failed_downloads_path = run_path / "failed_downloads.json"
                import json
                with open(failed_downloads_path, 'w') as f:
                    json.dump({
                        'total_failed': len(failed_downloads),
                        'total_attempted': total_records,
                        'failures': failed_downloads
                    }, f, indent=2)
                
                logger.warning(
                    f"Download stage completed with failures: "
                    f"{self.pipeline_stats['downloads_completed']}/{total_records} succeeded, "
                    f"{len(failed_downloads)} failed. See {failed_downloads_path}"
                )
            else:
                logger.info(f"Download stage completed: {self.pipeline_stats['downloads_completed']}/{total_records} images")
            
            # Signal end of downloads to all classification workers
            classification_workers = min(15, self.settings.processing.max_workers)
            for _ in range(classification_workers):
                await output_queue.put(None)
    
    async def _classification_stage(self, input_queue: Queue, vto_queue: Queue, product_queue: Queue, results_list: List):
        """Classification stage - classifies images as they arrive from downloads."""
        logger.info("Starting classification stage")
        
        # Limit concurrent classifications to avoid API rate limits
        semaphore = asyncio.Semaphore(min(self.settings.processing.max_workers, 25))
        
        async def classify_worker():
            while not self.shutdown_requested:
                try:
                    # Use timeout to avoid indefinite waiting
                    item = await asyncio.wait_for(input_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                    
                if item is None:
                    # End signal received - don't propagate, let all workers exit
                    break
                
                record, path = item
                async with semaphore:
                    try:
                        result = await self.analyzer.classify_image(
                            path,
                            additional_context=f"Command: {record.image_command}, Position: {record.image_position}",
                        )
                        
                        category = result.category
                        apparel_info = {
                            'detected_items': result.detected_items,
                            'target_gender': result.target_gender,
                            'confidence': result.confidence,
                            'reasoning': result.reasoning,
                            'metadata': result.metadata,
                        }
                        
                        # Route to appropriate processor queue
                        if category == ImageCategory.APPAREL:
                            await vto_queue.put((record, path, category, apparel_info))
                        else:
                            # PRODUCT and UNKNOWN go to product_queue
                            await product_queue.put((record, path, category, apparel_info))
                        
                        results_list.append((record, path, category))
                        self.pipeline_stats['classifications_completed'] += 1
                        
                        logger.info(
                            f"Classified: {record.id} as {category.value}",
                            confidence=result.confidence,
                            items=result.detected_items,
                        )
                    except Exception as e:
                        logger.error(f"Classification failed for {record.id}: {e}")
                        # Put with UNKNOWN category on failure (goes to product_queue)
                        await product_queue.put((record, path, ImageCategory.UNKNOWN, {}))
                        self.pipeline_stats['classifications_completed'] += 1
        
        # Start multiple classification workers
        num_workers = min(15, self.settings.processing.max_workers)
        workers = [asyncio.create_task(classify_worker()) for _ in range(num_workers)]
        await asyncio.gather(*workers)
        
        # Signal end to both processing queues
        vto_workers = min(10, self.settings.processing.max_workers // 2)
        product_workers = min(15, self.settings.processing.max_workers)
        
        # Send None signals to both queues to ensure all workers exit
        for _ in range(vto_workers):
            await vto_queue.put(None)
        for _ in range(product_workers):
            await product_queue.put(None)
            
        logger.info(f"Classification stage completed: {self.pipeline_stats['classifications_completed']} images")
    
    async def _vto_processing_stage(self, vto_queue: Queue, run_path: Path, results_list: List):
        """VTO processing stage - processes apparel images through Virtual Try-On."""
        logger.info("Starting VTO processing stage")
        
        vto_semaphore = asyncio.Semaphore(min(self.settings.processing.max_workers // 2, 15))
        
        async def vto_worker():
            while not self.shutdown_requested:
                try:
                    # Use timeout to avoid indefinite waiting
                    item = await asyncio.wait_for(vto_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                    
                if item is None:
                    # End signal received
                    break
                
                record, path, category, apparel_info = item
                
                try:
                    async with vto_semaphore:
                        vto_output_dir = run_path / "virtual_try_on"
                        result = await self.vto_processor.process_with_retry(
                            record, path, vto_output_dir
                        )
                        results_list.append(result)
                        self.pipeline_stats['processing_completed'] += 1
                        
                        # Track time to first result
                        if self.pipeline_stats['first_result_time'] is None:
                            self.pipeline_stats['first_result_time'] = time.time()
                        
                        logger.info(
                            f"VTO processed: {record.id}",
                            success=result.success,
                            quality_score=result.quality_score,
                        )
                        
                except Exception as e:
                    logger.error(f"VTO processing failed for {record.id}: {e}")
                    # Create failed result
                    result = ProcessingResult(
                        record_id=record.id,
                        success=False,
                        error_message=str(e),
                    )
                    results_list.append(result)
                    self.pipeline_stats['processing_completed'] += 1
        
        # Start multiple VTO workers
        num_workers = min(10, self.settings.processing.max_workers // 2)
        workers = [asyncio.create_task(vto_worker()) for _ in range(num_workers)]
        await asyncio.gather(*workers)
        logger.info(f"VTO processing stage completed")
    
    async def _product_processing_stage(self, product_queue: Queue, run_path: Path, results_list: List):
        """Product processing stage - processes product/unknown images through Product Recontext."""
        logger.info("Starting Product Recontext processing stage")
        
        product_semaphore = asyncio.Semaphore(min(self.settings.processing.max_workers, 25))
        
        async def product_worker():
            while not self.shutdown_requested:
                try:
                    # Use timeout to avoid indefinite waiting
                    item = await asyncio.wait_for(product_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                    
                if item is None:
                    # End signal received
                    break
                
                record, path, category, apparel_info = item
                
                try:
                    async with product_semaphore:
                        product_output_dir = run_path / "product_recontext"
                        result = await self.product_processor.process_with_retry(
                            record, path, product_output_dir
                        )
                        results_list.append(result)
                        self.pipeline_stats['processing_completed'] += 1
                        
                        # Track time to first result
                        if self.pipeline_stats['first_result_time'] is None:
                            self.pipeline_stats['first_result_time'] = time.time()
                        
                        if category == ImageCategory.PRODUCT:
                            logger.info(
                                f"Product Recontext processed: {record.id}",
                                success=result.success,
                                quality_score=result.quality_score,
                            )
                        else:
                            logger.info(
                                f"Unknown category processed with Product Recontext: {record.id}",
                                success=result.success,
                            )
                            
                except Exception as e:
                    logger.error(f"Product processing failed for {record.id}: {e}")
                    # Create failed result
                    result = ProcessingResult(
                        record_id=record.id,
                        success=False,
                        error_message=str(e),
                    )
                    results_list.append(result)
                    self.pipeline_stats['processing_completed'] += 1
        
        # Start multiple Product workers
        num_workers = min(15, self.settings.processing.max_workers)
        workers = [asyncio.create_task(product_worker()) for _ in range(num_workers)]
        await asyncio.gather(*workers)
        logger.info(f"Product Recontext processing stage completed")
    
    async def _progress_monitor(self, total_images: int):
        """Monitor and log pipeline progress."""
        stall_counter = 0
        last_progress = {
            'downloads': 0,
            'classifications': 0,
            'processing': 0,
        }
        
        while not self.shutdown_requested:
            await asyncio.sleep(10)  # Log progress every 10 seconds
            
            current_progress = {
                'downloads': self.pipeline_stats['downloads_completed'],
                'classifications': self.pipeline_stats['classifications_completed'],
                'processing': self.pipeline_stats['processing_completed'],
            }
            
            logger.info(
                "Pipeline progress",
                downloads=f"{current_progress['downloads']}/{total_images}",
                classifications=f"{current_progress['classifications']}/{total_images}",
                processing=f"{current_progress['processing']}/{total_images}",
                pipeline_mode=True,
            )
            
            # Check if we're making progress
            if current_progress == last_progress:
                stall_counter += 1
                if stall_counter >= 6:  # No progress for 60 seconds
                    # Check if downloads are complete and we've processed everything we can
                    if (current_progress['downloads'] < total_images and 
                        current_progress['processing'] == current_progress['downloads']):
                        logger.warning(
                            "Pipeline stalled - all downloaded items processed but waiting for remaining downloads. "
                            f"Downloaded: {current_progress['downloads']}/{total_images}"
                        )
                        # If downloads are done but less than total, some failed
                        # Exit if everything that could be processed has been
                        if current_progress['processing'] == current_progress['classifications'] == current_progress['downloads']:
                            logger.info("All available items processed. Exiting pipeline.")
                            break
            else:
                stall_counter = 0
                last_progress = current_progress.copy()
            
            # Stop monitoring when all downloaded items are processed
            # (not waiting for total_images since some might fail to download)
            if (current_progress['downloads'] == current_progress['classifications'] == current_progress['processing'] and
                stall_counter >= 3):  # Give it 30 seconds to ensure everything is done
                logger.info(
                    f"Pipeline complete: processed {current_progress['processing']} items "
                    f"(of {total_images} attempted)"
                )
                break
    
    async def _upload_run_directory_to_gcs(self, run_path: Path) -> None:
        """Upload entire run directory to GCS."""
        try:
            logger.info("Starting GCS upload", run_path=str(run_path))
            from google.cloud import storage
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.settings.google_cloud.storage_bucket)
            
            # Extract the relative path from 'output/' onwards
            base_output_dir = Path("./output")
            relative_run_path = run_path.relative_to(base_output_dir)
            
            # Collect all files to upload
            files_to_upload = []
            total_size = 0
            
            for local_file_path in run_path.rglob('*'):
                if local_file_path.is_file():
                    relative_file_path = local_file_path.relative_to(base_output_dir)
                    gcs_path = f"output/{relative_file_path}"
                    files_to_upload.append((local_file_path, gcs_path))
                    total_size += local_file_path.stat().st_size
            
            logger.info(f"Found {len(files_to_upload)} files to upload, total size: {total_size / (1024*1024):.2f} MB")
            
            # Upload files concurrently
            async def upload_file_async(local_path: Path, gcs_path: str, semaphore: asyncio.Semaphore) -> bool:
                async with semaphore:
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: bucket.blob(gcs_path).upload_from_filename(str(local_path))
                        )
                        return True
                    except Exception as e:
                        logger.error(f"Failed to upload {local_path} to {gcs_path}: {e}")
                        return False
            
            # Limit concurrent uploads
            semaphore = asyncio.Semaphore(min(20, self.settings.processing.max_workers))
            
            upload_tasks = [
                upload_file_async(local_path, gcs_path, semaphore)
                for local_path, gcs_path in files_to_upload
            ]
            
            results = await asyncio.gather(*upload_tasks, return_exceptions=False)
            
            uploaded_files = sum(1 for r in results if r)
            failed_files = len(results) - uploaded_files
            
            logger.info(
                "Complete run directory uploaded to GCS",
                run_path=str(run_path),
                gcs_base_path=f"gs://{self.settings.google_cloud.storage_bucket}/output/{relative_run_path}",
                uploaded_files=uploaded_files,
                failed_files=failed_files,
                total_size_mb=round(total_size / (1024 * 1024), 2),
            )
            
        except Exception as e:
            logger.error(
                "Failed to upload run directory to GCS",
                run_path=str(run_path),
                error=str(e),
            )


async def main_pipeline(csv_path: Optional[Path] = None) -> None:
    """Main entry point for pipeline processor."""
    if csv_path is None:
        csv_path = Path("./image_folder/img_conversion_table.csv")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        processor = PipelineBulkImageProcessor()
        summary = await processor.process_from_csv(csv_path)
        
        logger.info(
            "Pipeline processing completed successfully",
            summary=summary,
        )
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Processing failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_pipeline())