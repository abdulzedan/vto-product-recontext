"""Product Recontext processor implementation."""

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictResponse
from PIL import Image

from ..analyzer import GeminiAnalyzer
from ..config import Settings
from ..downloader import ImageRecord
from ..utils import (
    encode_image_to_base64,
    prediction_to_pil_image,
    save_image_with_metadata,
)
from .base import BaseProcessor, ProcessingResult

logger = structlog.get_logger(__name__)


class ProductRecontextProcessor(BaseProcessor):
    """Processor for Product Recontext operations."""

    def __init__(self, settings: Settings, analyzer: GeminiAnalyzer):
        super().__init__(settings, analyzer)
        self.client = None
        self.model_endpoint = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Initialize Vertex AI client."""
        try:
            aiplatform.init(
                project=self.settings.google_cloud.project_id,
                location=self.settings.google_cloud.location,
            )

            api_regional_endpoint = (
                f"{self.settings.google_cloud.location}-aiplatform.googleapis.com"
            )
            client_options = {"api_endpoint": api_regional_endpoint}
            self.client = aiplatform.gapic.PredictionServiceClient(
                client_options=client_options
            )

            self.model_endpoint = (
                f"projects/{self.settings.google_cloud.project_id}/"
                f"locations/{self.settings.google_cloud.location}/"
                f"publishers/google/models/{self.settings.google_cloud.model_endpoint_product}"
            )

            logger.info(
                "Product Recontext client initialized",
                project=self.settings.google_cloud.project_id,
                location=self.settings.google_cloud.location,
                endpoint=self.model_endpoint,
            )

        except Exception as e:
            logger.error("Failed to initialize Product Recontext client", error=str(e))
            raise

    def get_processor_type(self) -> str:
        """Get processor type name."""
        return "product_recontext"

    async def call_product_recontext(
        self,
        image_bytes: str,
        prompt: str,
        product_description: Optional[str] = None,
        sample_count: int = 1,
        disable_prompt_enhancement: bool = False,
        safety_setting: str = "block_low_and_above",
        person_generation: str = "allow_adult",
    ) -> PredictResponse:
        """Call Product Recontext API."""
        instance = {
            "productImages": [{"image": {"bytesBase64Encoded": image_bytes}}],
            "prompt": prompt,
        }

        if product_description:
            instance["productImages"][0]["productConfig"] = {
                "productDescription": product_description
            }

        parameters = {
            "sampleCount": sample_count,
            "safetySetting": safety_setting,
            "personGeneration": person_generation,
        }

        if disable_prompt_enhancement:
            parameters["enhancePrompt"] = False

        try:
            start_time = time.time()

            response = self.client.predict(
                endpoint=self.model_endpoint,
                instances=[instance],
                parameters=parameters,
            )

            api_time = time.time() - start_time

            logger.info(
                "Product Recontext API call completed",
                api_time=round(api_time, 2),
                sample_count=sample_count,
                prompt_length=len(prompt),
            )

            return response

        except Exception as e:
            logger.error(
                "Product Recontext API call failed",
                error=str(e),
                prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
            )
            raise

    async def process_image(
        self,
        record: ImageRecord,
        image_path: Path,
        output_dir: Path,
    ) -> ProcessingResult:
        """Process a single image for Product Recontext."""
        try:
            # Validate input image
            if not self.validate_image(image_path):
                return ProcessingResult(
                    record_id=record.id,
                    success=False,
                    error_message="Invalid input image",
                )

            # Create output directory for this record
            record_output_dir = self.create_output_directory(output_dir, record)

            # Prepare product image
            product_image_bytes = self._prepare_product_image(image_path)

            # Analyze product to understand what it is
            product_analysis = await self.analyzer.classify_image(image_path)

            # Generate compelling prompt for high-end retail
            product_description = self._extract_product_description(
                record, product_analysis
            )
            generated_prompt = await self.analyzer.generate_product_recontext_prompt(
                image_path,
                product_description,
            )

            logger.info(
                "Generated product recontext prompt",
                record_id=record.id,
                prompt=generated_prompt,
                product_description=product_description,
            )

            # Call Product Recontext API
            response = await self.call_product_recontext(
                image_bytes=product_image_bytes,
                prompt=generated_prompt,
                product_description=product_description,
                sample_count=1,
                disable_prompt_enhancement=False,
            )

            # Process API response
            predictions = list(response.predictions)
            if not predictions or "bytesBase64Encoded" not in predictions[0]:
                return ProcessingResult(
                    record_id=record.id,
                    success=False,
                    error_message="No valid predictions returned from API",
                )

            # Convert prediction to image
            result_image = prediction_to_pil_image(predictions[0])

            # Save result image
            result_path = record_output_dir / "result.jpg"
            result_image.save(result_path, "JPEG", quality=95)

            # Save generated prompt
            prompt_path = record_output_dir / "prompt.txt"
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(generated_prompt)

            # Analyze quality using Gemini
            quality_feedback = None
            quality_score = 0.8  # Placeholder score

            try:
                quality_feedback = (
                    await self.analyzer.analyze_product_recontext_quality(
                        result_image=result_image,
                        original_product=image_path,
                        generated_prompt=generated_prompt,
                    )
                )
                quality_score = quality_feedback.score

                logger.info(
                    "Product Recontext quality analysis completed",
                    record_id=record.id,
                    passed=quality_feedback.passed,
                    score=quality_score,
                )

            except Exception as e:
                logger.warning(
                    "Quality analysis failed",
                    record_id=record.id,
                    error=str(e),
                )

            # Prepare metadata
            metadata = self.prepare_metadata(
                record,
                {
                    "product_analysis": product_analysis.dict(),
                    "generated_prompt": generated_prompt,
                    "product_description": product_description,
                    "api_response_time": time.time(),
                    "quality_score": quality_score,
                    "quality_feedback": (
                        quality_feedback.dict() if quality_feedback else None
                    ),
                },
            )

            # Save metadata
            metadata_path = record_output_dir / "metadata.json"
            import json

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Upload to GCS if enabled
            gcs_path = None
            if self.settings.storage.enable_gcs_upload:
                gcs_path = await self._upload_to_gcs(
                    result_path,
                    prompt_path,
                    metadata_path,
                    record,
                    image_path,
                )

            return ProcessingResult(
                record_id=record.id,
                success=True,
                output_path=result_path,
                gcs_path=gcs_path,
                metadata=metadata,
                quality_score=quality_score,
                feedback=str(quality_feedback) if quality_feedback else None,
            )

        except Exception as e:
            logger.error(
                "Product Recontext processing failed",
                record_id=record.id,
                error=str(e),
            )
            return ProcessingResult(
                record_id=record.id,
                success=False,
                error_message=str(e),
            )

    def _prepare_product_image(self, image_path: Path) -> str:
        """Prepare product image for API call."""
        with open(image_path, "rb") as f:
            raw_image_bytes = f.read()

        # Process image - convert to RGB and resize
        image_pil = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
        original_size = image_pil.size

        # Apply thumbnail to maintain aspect ratio within 1024x1024
        image_pil.thumbnail((1024, 1024))

        logger.info(
            "Product image prepared",
            original_size=original_size,
            processed_size=image_pil.size,
        )

        # Encode to base64
        buffer = io.BytesIO()
        image_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_product_description(
        self,
        record: ImageRecord,
        product_analysis: Any,
    ) -> Optional[str]:
        """Extract product description from record and analysis."""
        descriptions = []

        # Use image command as description if available
        if record.image_command:
            descriptions.append(record.image_command)

        # Use detected items from analysis
        if (
            hasattr(product_analysis, "detected_items")
            and product_analysis.detected_items
        ):
            items_str = ", ".join(product_analysis.detected_items)
            descriptions.append(f"Product contains: {items_str}")

        # Use metadata if available
        if hasattr(product_analysis, "metadata") and product_analysis.metadata:
            metadata = product_analysis.metadata
            if metadata.get("primary_color"):
                descriptions.append(f"Primary color: {metadata['primary_color']}")
            if metadata.get("material"):
                descriptions.append(f"Material: {metadata['material']}")
            if metadata.get("style"):
                descriptions.append(f"Style: {metadata['style']}")

        return ". ".join(descriptions) if descriptions else None

    async def _upload_to_gcs(
        self,
        result_path: Path,
        prompt_path: Path,
        metadata_path: Path,
        record: ImageRecord,
        product_image_path: Path,
    ) -> Optional[str]:
        """Upload results to Google Cloud Storage."""
        try:
            from google.cloud import storage

            storage_client = storage.Client()
            bucket = storage_client.bucket(self.settings.google_cloud.storage_bucket)

            # Generate GCS paths
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_path = f"{self.settings.storage.gcs_product_recontext_path}/outputs/{timestamp}_{record.id}"
            inputs_path = f"{self.settings.storage.gcs_product_recontext_path}/inputs/{timestamp}_{record.id}"

            # Upload input product image
            product_input_blob = bucket.blob(f"{inputs_path}/product_image.jpg")
            product_input_blob.upload_from_filename(str(product_image_path))

            # Upload result image
            result_blob = bucket.blob(f"{base_path}/result.jpg")
            result_blob.upload_from_filename(str(result_path))

            # Make only the result image publicly accessible
            try:
                result_blob.make_public()
                logger.info(
                    f"Result image made publicly accessible: {result_blob.name}"
                )
            except Exception as e:
                logger.warning(f"Failed to make result image public: {e}")

            # Upload prompt
            prompt_blob = bucket.blob(f"{base_path}/prompt.txt")
            prompt_blob.upload_from_filename(str(prompt_path))

            # Upload metadata
            metadata_blob = bucket.blob(f"{base_path}/metadata.json")
            metadata_blob.upload_from_filename(str(metadata_path))

            gcs_uri = f"gs://{self.settings.google_cloud.storage_bucket}/{base_path}/result.jpg"

            logger.info(
                "Results and inputs uploaded to GCS",
                record_id=record.id,
                gcs_uri=gcs_uri,
                inputs_path=f"gs://{self.settings.google_cloud.storage_bucket}/{inputs_path}/",
            )

            return gcs_uri

        except Exception as e:
            logger.error(
                "Failed to upload to GCS",
                record_id=record.id,
                error=str(e),
            )
            return None
