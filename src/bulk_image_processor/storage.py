"""Storage management for local and Google Cloud Storage."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from google.cloud import storage
from google.cloud.exceptions import NotFound

from .config import Settings
from .utils import ensure_directory

logger = structlog.get_logger(__name__)


class StorageManager:
    """Manages storage operations for local and GCS."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.gcs_client = None
        self.bucket = None
        self._setup_storage()

    def _setup_storage(self) -> None:
        """Initialize storage clients."""
        # Setup local storage - only create the base output directory
        # Processor-specific directories are created on-demand with timestamped structure
        ensure_directory(self.settings.storage.local_output_dir)

        # Setup GCS if enabled
        if self.settings.storage.enable_gcs_upload:
            try:
                self.gcs_client = storage.Client(
                    project=self.settings.google_cloud.project_id
                )
                self.bucket = self.gcs_client.bucket(
                    self.settings.google_cloud.storage_bucket
                )

                # Verify bucket exists
                if not self.bucket.exists():
                    logger.error(
                        "GCS bucket does not exist",
                        bucket=self.settings.google_cloud.storage_bucket,
                    )
                    raise ValueError(
                        f"Bucket {self.settings.google_cloud.storage_bucket} does not exist"
                    )

                logger.info(
                    "GCS storage initialized",
                    bucket=self.settings.google_cloud.storage_bucket,
                )

            except Exception as e:
                logger.error("Failed to initialize GCS storage", error=str(e))
                raise

        logger.info(
            "Storage manager initialized",
            local_dir=str(self.settings.storage.local_output_dir),
            gcs_enabled=self.settings.storage.enable_gcs_upload,
        )

    def get_local_output_dir(self, processor_type: str) -> Path:
        """Get local output directory for processor type."""
        return self.settings.storage.local_output_dir / processor_type

    def upload_file(
        self,
        local_path: Path,
        gcs_path: str,
        metadata: Optional[Dict[str, str]] = None,
        make_public: bool = True,
    ) -> str:
        """Upload file to GCS and optionally make it publicly accessible."""
        if not self.settings.storage.enable_gcs_upload:
            raise ValueError("GCS upload is disabled")

        if not self.bucket:
            raise ValueError("GCS bucket not initialized")

        try:
            blob = self.bucket.blob(gcs_path)

            if metadata:
                blob.metadata = metadata

            blob.upload_from_filename(str(local_path))

            # Make the blob publicly accessible if requested
            # NOTE: Disabled due to uniform bucket-level access
            # if make_public:
            #     blob.make_public()

            gcs_uri = f"gs://{self.settings.google_cloud.storage_bucket}/{gcs_path}"

            logger.info(
                "File uploaded to GCS",
                local_path=str(local_path),
                gcs_uri=gcs_uri,
                size=local_path.stat().st_size,
                public=make_public,
            )

            return gcs_uri

        except Exception as e:
            logger.error(
                "Failed to upload file to GCS",
                local_path=str(local_path),
                gcs_path=gcs_path,
                error=str(e),
            )
            raise

    def download_file(
        self,
        gcs_path: str,
        local_path: Path,
    ) -> None:
        """Download file from GCS."""
        if not self.bucket:
            raise ValueError("GCS bucket not initialized")

        try:
            blob = self.bucket.blob(gcs_path)

            if not blob.exists():
                raise NotFound(f"File not found: gs://{self.bucket.name}/{gcs_path}")

            ensure_directory(local_path.parent)
            blob.download_to_filename(str(local_path))

            logger.info(
                "File downloaded from GCS",
                gcs_path=gcs_path,
                local_path=str(local_path),
                size=local_path.stat().st_size,
            )

        except Exception as e:
            logger.error(
                "Failed to download file from GCS",
                gcs_path=gcs_path,
                local_path=str(local_path),
                error=str(e),
            )
            raise

    def list_files(
        self,
        gcs_prefix: str,
        max_results: Optional[int] = None,
    ) -> List[str]:
        """List files in GCS with given prefix."""
        if not self.bucket:
            raise ValueError("GCS bucket not initialized")

        try:
            blobs = self.bucket.list_blobs(
                prefix=gcs_prefix,
                max_results=max_results,
            )

            file_paths = [blob.name for blob in blobs]

            logger.info(
                "Files listed from GCS",
                prefix=gcs_prefix,
                count=len(file_paths),
            )

            return file_paths

        except Exception as e:
            logger.error(
                "Failed to list files from GCS",
                prefix=gcs_prefix,
                error=str(e),
            )
            raise

    def save_processing_log(
        self,
        log_data: Dict[str, Any],
        processor_type: str,
        record_id: str,
    ) -> Path:
        """Save processing log locally and optionally to GCS."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{processor_type}_{record_id}_{timestamp}.json"

        # Save locally
        local_log_dir = Path("./logs")
        ensure_directory(local_log_dir)
        local_log_path = local_log_dir / log_filename

        with open(local_log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        # Upload to GCS if enabled
        if self.settings.storage.enable_gcs_upload:
            try:
                gcs_log_path = f"{self.settings.storage.gcs_logs_path}/{log_filename}"
                self.upload_file(local_log_path, gcs_log_path)
            except Exception as e:
                logger.warning(
                    "Failed to upload log to GCS",
                    log_file=str(local_log_path),
                    error=str(e),
                )

        return local_log_path

    def create_processing_summary(
        self,
        summary_data: Dict[str, Any],
        processor_type: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Create processing summary file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_filename = f"{processor_type}_summary_{timestamp}.json"

        # Save in the run directory if provided, otherwise use logs directory
        if output_dir:
            local_summary_path = output_dir / summary_filename
        else:
            local_summary_dir = Path("./logs")
            ensure_directory(local_summary_dir)
            local_summary_path = local_summary_dir / summary_filename

        with open(local_summary_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        # Upload to GCS if enabled
        if self.settings.storage.enable_gcs_upload:
            try:
                gcs_summary_path = f"{self.settings.storage.gcs_logs_path}/summaries/{summary_filename}"
                self.upload_file(local_summary_path, gcs_summary_path)
            except Exception as e:
                logger.warning(
                    "Failed to upload summary to GCS",
                    summary_file=str(local_summary_path),
                    error=str(e),
                )

        logger.info(
            "Processing summary created",
            summary_file=str(local_summary_path),
            processor_type=processor_type,
        )

        return local_summary_path

    def create_results_csv(
        self,
        processing_results: List[Any],
        output_dir: Path,
        run_id: str,
    ) -> Path:
        """Create CSV file with processing results in img_conversion_table format."""
        import csv

        csv_filename = f"results_{run_id}.csv"
        csv_path = output_dir / csv_filename

        # Create CSV with img_conversion_table format
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["ID", "Image Src", "Image Command", "Image Position"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for result in processing_results:
                if result.success:
                    # Prefer GCS path if available, otherwise use local path
                    if result.gcs_path:
                        # Generate public URL for GCS path
                        gcs_path_clean = result.gcs_path.replace(
                            "gs://" + self.settings.google_cloud.storage_bucket + "/",
                            "",
                        )
                        image_url = f"https://storage.googleapis.com/{self.settings.google_cloud.storage_bucket}/{gcs_path_clean}"
                    elif result.output_path:
                        # Fall back to local file path (for debugging/testing)
                        image_url = str(result.output_path)
                    else:
                        # Skip results without any output path
                        continue

                    row = {
                        "ID": result.record_id,
                        "Image Src": image_url,
                        "Image Command": "MERGE",
                        "Image Position": "1",
                    }
                    writer.writerow(row)

        # Upload to GCS if enabled
        if self.settings.storage.enable_gcs_upload:
            try:
                gcs_csv_path = f"results/{csv_filename}"
                self.upload_file(csv_path, gcs_csv_path)
                logger.info(
                    "Results CSV uploaded to GCS",
                    local_path=str(csv_path),
                    gcs_path=gcs_csv_path,
                )
            except Exception as e:
                logger.warning(
                    "Failed to upload results CSV to GCS",
                    csv_file=str(csv_path),
                    error=str(e),
                )

        logger.info(
            "Results CSV created",
            csv_file=str(csv_path),
            num_results=len(processing_results),
        )

        return csv_path

    def cleanup_old_files(
        self,
        directory: Path,
        max_age_days: int = 7,
    ) -> int:
        """Clean up old files from local directory."""
        if not directory.exists():
            return 0

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        deleted_count = 0

        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        deleted_count += 1

            logger.info(
                "Old files cleaned up",
                directory=str(directory),
                deleted_count=deleted_count,
                max_age_days=max_age_days,
            )

        except Exception as e:
            logger.error(
                "Failed to clean up old files",
                directory=str(directory),
                error=str(e),
            )

        return deleted_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "local_storage": {
                "base_dir": str(self.settings.storage.local_output_dir),
                "exists": self.settings.storage.local_output_dir.exists(),
            },
            "gcs_storage": {
                "enabled": self.settings.storage.enable_gcs_upload,
                "bucket": self.settings.google_cloud.storage_bucket,
                "connected": self.bucket is not None,
            },
        }

        # Add local directory sizes if they exist
        if self.settings.storage.local_output_dir.exists():
            try:

                def get_dir_size(path: Path) -> int:
                    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

                stats["local_storage"]["size_bytes"] = get_dir_size(
                    self.settings.storage.local_output_dir
                )

            except Exception as e:
                logger.warning("Failed to calculate local storage size", error=str(e))

        return stats


def create_storage_manager(settings: Settings) -> StorageManager:
    """Create a storage manager instance."""
    return StorageManager(settings)
