"""Tests for the image downloader module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bulk_image_processor.downloader import ImageDownloader, ImageRecord
from bulk_image_processor.exceptions import (
    CSVParsingError,
    DownloadError,
    ImageValidationError,
)


class TestImageRecord:
    """Test ImageRecord class."""

    def test_create_image_record(self):
        """Test creating an image record."""
        record = ImageRecord(
            id="test_001",
            image_url="https://example.com/test.jpg",
            image_command="process",
            image_position="center",
            row_index=0,
        )

        assert record.id == "test_001"
        assert record.image_url == "https://example.com/test.jpg"
        assert record.image_command == "process"
        assert record.image_position == "center"
        assert record.row_index == 0
        assert record.unique_id.startswith("img_test_001_")

    def test_invalid_url(self):
        """Test that invalid URLs raise DownloadError."""
        with pytest.raises(DownloadError, match="Invalid image URL format"):
            ImageRecord(
                id="test_001",
                image_url="not-a-valid-url",
                image_command="process",
                image_position="center",
                row_index=0,
            )


class TestImageDownloader:
    """Test ImageDownloader class."""

    @pytest.mark.asyncio
    async def test_load_csv(self, mock_settings, temp_csv_file):
        """Test loading CSV file."""
        async with ImageDownloader(mock_settings) as downloader:
            records = downloader.load_csv(temp_csv_file)

            assert len(records) == 3
            assert records[0].id == "001"
            assert records[0].image_url == "https://example.com/shirt.jpg"
            assert records[1].id == "002"
            assert records[2].id == "003"

    @pytest.mark.asyncio
    async def test_load_csv_missing_columns(self, mock_settings, tmp_path):
        """Test loading CSV with missing columns."""
        csv_path = tmp_path / "invalid.csv"
        with open(csv_path, "w") as f:
            f.write("ID,Image Src\n001,https://example.com/test.jpg\n")

        async with ImageDownloader(mock_settings) as downloader:
            with pytest.raises(CSVParsingError, match="Missing required columns"):
                downloader.load_csv(csv_path)

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_download_single_image_success(
        self, mock_get, mock_settings, sample_image_record, tmp_path
    ):
        """Test successful image download."""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "image/jpeg"}
        mock_response.read.return_value = b"fake_image_data"
        mock_get.return_value.__aenter__.return_value = mock_response

        # Mock PIL Image
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image.verify = MagicMock()
            mock_image.mode = "RGB"
            mock_image.convert.return_value = mock_image
            mock_image.save = MagicMock()
            mock_image.width = 100
            mock_image.height = 100
            mock_image_open.return_value = mock_image

            async with ImageDownloader(mock_settings) as downloader:
                result = await downloader.download_single_image(
                    sample_image_record, tmp_path
                )

                assert result is not None
                assert result.name.startswith("img_test_001_")
                assert result.suffix == ".jpg"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_download_single_image_http_error(
        self, mock_get, mock_settings, sample_image_record, tmp_path
    ):
        """Test image download with HTTP error."""
        # Mock HTTP error response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.reason = "Not Found"
        mock_get.return_value.__aenter__.return_value = mock_response

        async with ImageDownloader(mock_settings) as downloader:
            with pytest.raises(DownloadError) as exc_info:
                await downloader.download_single_image(sample_image_record, tmp_path)

            assert "HTTP error" in str(exc_info.value)
            assert exc_info.value.context["status_code"] == 404

    @pytest.mark.asyncio
    async def test_download_stats(self, mock_settings):
        """Test download statistics tracking."""
        async with ImageDownloader(mock_settings) as downloader:
            stats = downloader.get_download_stats()

            assert stats["total"] == 0
            assert stats["successful"] == 0
            assert stats["failed"] == 0
            assert stats["skipped"] == 0
