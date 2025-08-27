"""Asynchronous image downloader with retry logic."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import aiohttp
import pandas as pd
import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings
from .exceptions import (
    CSVParsingError,
    DownloadError,
    ImageValidationError,
    TimeoutError,
)
from .utils import ensure_directory, generate_unique_id, validate_image_url

logger = structlog.get_logger(__name__)


class ImageRecord:
    """Represents a single image record from CSV."""

    def __init__(
        self,
        id: Union[str, int],
        image_url: str,
        image_command: str,
        image_position: str,
        row_index: int,
    ):
        self.id = str(id)
        self.image_url = image_url.strip()
        self.image_command = image_command.strip() if image_command else ""
        self.image_position = image_position.strip() if image_position else ""
        self.row_index = row_index
        self.unique_id = generate_unique_id(f"img_{self.id}")

        # Validate URL
        if not validate_image_url(self.image_url):
            raise DownloadError(
                f"Invalid image URL format",
                url=self.image_url,
                context={"record_id": self.id},
            )

    def __repr__(self) -> str:
        return f"ImageRecord(id={self.id}, url={self.image_url[:50]}...)"


class ImageDownloader:
    """Asynchronous image downloader with retry logic."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: Optional[aiohttp.ClientSession] = None
        self.download_stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(
            total=self.settings.processing.download_timeout,
            connect=10,
        )

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def load_csv(self, csv_path: Path) -> List[ImageRecord]:
        """Load image records from CSV file."""
        logger.info("Loading CSV file", path=str(csv_path))

        try:
            # First, try to read without specifying dtype to detect format
            df = pd.read_csv(csv_path)
            logger.info(
                "CSV loaded successfully", rows=len(df), columns=list(df.columns)
            )
        except Exception as e:
            logger.error("Failed to load CSV", error=str(e))
            raise CSVParsingError(
                f"Failed to load CSV file: {str(e)}", csv_path=str(csv_path)
            )

        # Detect CSV format based on columns
        is_accessories_format = False
        if "Image URL" in df.columns and len(df.columns) == 1:
            # Accessories format: only has Image URL column
            is_accessories_format = True
            logger.info("Detected accessories CSV format (Image URL only)")
        else:
            # Standard format: check for required columns
            required_columns = ["ID", "Image Src", "Image Command", "Image Position"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise CSVParsingError(
                    f"Missing required columns in CSV",
                    csv_path=str(csv_path),
                    context={
                        "missing_columns": missing_columns,
                        "found_columns": list(df.columns),
                    },
                )

        # Parse records based on format
        records = []
        failed_records = []

        for index, row in df.iterrows():
            try:
                if is_accessories_format:
                    # Accessories format handling
                    if pd.isna(row["Image URL"]) or not str(row["Image URL"]).strip():
                        logger.warning("Skipping row with empty URL", row_index=index)
                        continue

                    # Generate ID from index for accessories
                    record = ImageRecord(
                        id=f"accessory_{index + 1}",  # Generate ID
                        image_url=str(row["Image URL"]),
                        image_command="",  # No command for accessories
                        image_position="",  # No position for accessories
                        row_index=index,
                    )
                else:
                    # Standard format handling
                    if pd.isna(row["Image Src"]) or not str(row["Image Src"]).strip():
                        logger.warning("Skipping row with empty URL", row_index=index)
                        continue

                    record = ImageRecord(
                        id=row["ID"],
                        image_url=str(row["Image Src"]),
                        image_command=(
                            str(row["Image Command"])
                            if pd.notna(row["Image Command"])
                            else ""
                        ),
                        image_position=(
                            str(row["Image Position"])
                            if pd.notna(row["Image Position"])
                            else ""
                        ),
                        row_index=index,
                    )

                records.append(record)

            except Exception as e:
                logger.warning(
                    "Failed to parse record",
                    row_index=index,
                    error=str(e),
                )
                failed_records.append(index)

        logger.info(
            "CSV parsing completed",
            csv_format="accessories" if is_accessories_format else "standard",
            total_rows=len(df),
            valid_records=len(records),
            failed_records=len(failed_records),
        )

        return records

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def download_single_image(
        self,
        record: ImageRecord,
        output_dir: Path,
    ) -> Optional[Path]:
        """Download a single image with retry logic."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        start_time = time.time()

        try:
            logger.info(
                "Starting image download",
                record_id=record.id,
                url=record.image_url,
            )

            # Generate output filename
            parsed_url = urlparse(record.image_url)
            extension = Path(parsed_url.path).suffix or ".jpg"
            output_path = output_dir / f"{record.unique_id}{extension}"

            # Check if already downloaded
            if output_path.exists():
                logger.info("Image already exists, skipping", path=str(output_path))
                self.download_stats["skipped"] += 1
                return output_path

            # Download image
            async with self.session.get(record.image_url) as response:
                if response.status != 200:
                    raise DownloadError(
                        f"HTTP error: {response.reason}",
                        url=record.image_url,
                        status_code=response.status,
                    )

                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    logger.warning(
                        "Unexpected content type",
                        content_type=content_type,
                        record_id=record.id,
                    )

                # Read image data
                image_data = await response.read()

                # Validate image data
                if len(image_data) == 0:
                    raise DownloadError(
                        "Empty image data received",
                        url=record.image_url,
                        context={"record_id": record.id},
                    )

                # Validate image with PIL
                try:
                    image = Image.open(io.BytesIO(image_data))
                    image.verify()  # Verify image integrity

                    # Reopen for processing (verify closes the file)
                    image = Image.open(io.BytesIO(image_data))

                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Ensure output directory exists
                    ensure_directory(output_path.parent)

                    # Save image
                    image.save(output_path, "JPEG", quality=95)

                except Exception as e:
                    raise ImageValidationError(
                        f"Invalid image data: {str(e)}",
                        file_path=str(output_path),
                        validation_type="format",
                        context={"record_id": record.id, "url": record.image_url},
                    )

                download_time = time.time() - start_time
                file_size = len(image_data)

                logger.info(
                    "Image downloaded successfully",
                    record_id=record.id,
                    output_path=str(output_path),
                    file_size=file_size,
                    download_time=round(download_time, 2),
                    image_size=f"{image.width}x{image.height}",
                )

                self.download_stats["successful"] += 1
                return output_path

        except Exception as e:
            download_time = time.time() - start_time
            logger.error(
                "Image download failed",
                record_id=record.id,
                url=record.image_url,
                error=str(e),
                download_time=round(download_time, 2),
            )
            self.download_stats["failed"] += 1
            if isinstance(e, (DownloadError, ImageValidationError)):
                raise
            else:
                raise DownloadError(
                    f"Failed to download image: {str(e)}",
                    url=record.image_url,
                    context={"record_id": record.id, "error_type": type(e).__name__},
                )

    async def download_images(
        self,
        records: List[ImageRecord],
        output_dir: Path,
    ) -> List[Tuple[ImageRecord, Optional[Path]]]:
        """Download multiple images concurrently."""
        ensure_directory(output_dir)

        self.download_stats = {
            "total": len(records),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }

        logger.info(
            "Starting batch image download",
            total_images=len(records),
            output_dir=str(output_dir),
            max_workers=self.settings.processing.max_workers,
        )

        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(self.settings.processing.max_workers)

        async def download_with_semaphore(
            record: ImageRecord,
        ) -> Tuple[ImageRecord, Optional[Path]]:
            async with semaphore:
                try:
                    path = await self.download_single_image(record, output_dir)
                    return record, path
                except Exception as e:
                    logger.error(
                        "Failed to download image",
                        record_id=record.id,
                        error=str(e),
                    )
                    return record, None

        # Execute downloads concurrently
        start_time = time.time()
        results = await asyncio.gather(
            *[download_with_semaphore(record) for record in records],
            return_exceptions=False,
        )

        total_time = time.time() - start_time

        # Log final statistics
        logger.info(
            "Batch download completed",
            total_time=round(total_time, 2),
            **self.download_stats,
            success_rate=round(
                self.download_stats["successful"] / max(len(records), 1) * 100, 2
            ),
            avg_time_per_image=round(total_time / max(len(records), 1), 2),
        )

        return results

    def get_download_stats(self) -> Dict[str, int]:
        """Get current download statistics."""
        return self.download_stats.copy()


async def download_images_from_csv(
    csv_path: Path,
    output_dir: Path,
    settings: Settings,
) -> List[Tuple[ImageRecord, Optional[Path]]]:
    """Download images from CSV file."""
    async with ImageDownloader(settings) as downloader:
        records = downloader.load_csv(csv_path)
        return await downloader.download_images(records, output_dir)


# Import missing module
import io
