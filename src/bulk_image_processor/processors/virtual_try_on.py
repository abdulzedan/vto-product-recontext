"""Virtual Try-On processor implementation."""

import asyncio
import base64
import io
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictResponse
from PIL import Image

from ..config import Settings
from ..analyzer import GeminiAnalyzer
from ..downloader import ImageRecord
from ..utils import encode_image_to_base64, prediction_to_pil_image, save_image_with_metadata
from .base import BaseProcessor, ProcessingResult

logger = structlog.get_logger(__name__)


class VirtualTryOnProcessor(BaseProcessor):
    """Processor for Virtual Try-On operations."""
    
    def __init__(self, settings: Settings, analyzer: GeminiAnalyzer):
        super().__init__(settings, analyzer)
        self.client = None
        self.model_endpoint = None
        self.model_images = []
        self._setup_client()
        self._load_model_images()
    
    def _setup_client(self) -> None:
        """Initialize Vertex AI client."""
        try:
            aiplatform.init(
                project=self.settings.google_cloud.project_id,
                location=self.settings.google_cloud.location,
            )
            
            api_regional_endpoint = f"{self.settings.google_cloud.location}-aiplatform.googleapis.com"
            client_options = {"api_endpoint": api_regional_endpoint}
            self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
            
            self.model_endpoint = (
                f"projects/{self.settings.google_cloud.project_id}/"
                f"locations/{self.settings.google_cloud.location}/"
                f"publishers/google/models/{self.settings.google_cloud.model_endpoint}"
            )
            
            logger.info(
                "Virtual Try-On client initialized",
                project=self.settings.google_cloud.project_id,
                location=self.settings.google_cloud.location,
                endpoint=self.model_endpoint,
            )
            
        except Exception as e:
            logger.error("Failed to initialize Virtual Try-On client", error=str(e))
            raise
    
    def _load_model_images(self) -> None:
        """Load predefined model images for Virtual Try-On."""
        models_dir = Path("image_folder/image_models")
        
        if not models_dir.exists():
            logger.error(
                "Model images directory not found",
                path=str(models_dir),
            )
            self.model_images = []
            return
        
        # Load all JPG files from the models directory
        model_files = list(models_dir.glob("*.jpg"))
        
        if not model_files:
            logger.error(
                "No model images found",
                path=str(models_dir),
            )
            self.model_images = []
            return
        
        # Create model info for each image file based on naming convention: number_gender.jpg
        self.model_images = []
        
        for model_file in sorted(model_files):
            try:
                # Parse filename: e.g., "1_women.jpg" -> ("1", "women")
                filename_parts = model_file.stem.split('_')
                if len(filename_parts) != 2:
                    logger.warning(
                        "Skipping model file with unexpected naming format",
                        filename=model_file.name,
                        expected_format="number_gender.jpg",
                    )
                    continue
                
                model_number, gender = filename_parts
                
                # Validate gender
                if gender not in ['man', 'women']:
                    logger.warning(
                        "Skipping model file with invalid gender",
                        filename=model_file.name,
                        gender=gender,
                        expected_genders=['man', 'women'],
                    )
                    continue
                
                # Create model info
                model_info = {
                    'id': f'model_{model_number}_{gender}',
                    'description': f'Professional {gender} model {model_number}',
                    'path': model_file,
                    'gender': gender,
                    'number': model_number,
                    'filename': model_file.name,
                }
                self.model_images.append(model_info)
                
            except Exception as e:
                logger.warning(
                    "Failed to parse model file",
                    filename=model_file.name,
                    error=str(e),
                )
                continue
        
        # Group models by gender for easy access
        self.models_by_gender = {
            'women': [m for m in self.model_images if m['gender'] == 'women'],
            'man': [m for m in self.model_images if m['gender'] == 'man'],
        }
        
        logger.info(
            "Model images loaded",
            total_count=len(self.model_images),
            women_models=len(self.models_by_gender['women']),
            man_models=len(self.models_by_gender['man']),
            models=[m['filename'] for m in self.model_images],
        )
    
    def get_processor_type(self) -> str:
        """Get processor type name."""
        return "virtual_try_on"
    
    def select_model(self, apparel_info: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate model based on apparel characteristics and gender."""
        detected_items = apparel_info.get('detected_items', [])
        target_gender = apparel_info.get('target_gender')
        
        # First try to use the gender detected by Gemini
        if target_gender and target_gender in self.models_by_gender:
            suitable_models = self.models_by_gender[target_gender]
            if suitable_models:
                selected_model = random.choice(suitable_models)
                logger.info(
                    "Model selected by gender",
                    model_id=selected_model['id'],
                    gender=target_gender,
                    detected_items=detected_items,
                )
                return selected_model
        
        # Fallback to heuristic-based selection if gender detection failed
        if any(item.lower() in ['dress', 'blouse', 'skirt', 'heels'] for item in detected_items):
            preferred_gender = 'women'
        elif any(item.lower() in ['suit', 'tie', 'shirt'] for item in detected_items):
            preferred_gender = 'man'
        else:
            # Default to any available model
            preferred_gender = None
        
        # Try to select from preferred gender
        if preferred_gender and preferred_gender in self.models_by_gender:
            suitable_models = self.models_by_gender[preferred_gender]
            if suitable_models:
                selected_model = random.choice(suitable_models)
                logger.info(
                    "Model selected by heuristic",
                    model_id=selected_model['id'],
                    gender=preferred_gender,
                    detected_items=detected_items,
                )
                return selected_model
        
        # Final fallback to any available model
        if self.model_images:
            selected_model = random.choice(self.model_images)
            logger.info(
                "Model selected (fallback)",
                model_id=selected_model['id'],
                gender=selected_model.get('gender', 'unknown'),
                detected_items=detected_items,
            )
            return selected_model
        
        # This should never happen if models are loaded properly
        logger.error("No models available for selection")
        raise ValueError("No models available for selection")
    
    def prepare_model_image(self, model_info: Dict[str, Any]) -> str:
        """Prepare model image for API call."""
        model_path = model_info.get('path')
        
        if not model_path or not model_path.exists():
            logger.error(
                "Model image file not found",
                model_id=model_info['id'],
                path=str(model_path),
            )
            return ""
        
        try:
            # Load model image
            with open(model_path, 'rb') as f:
                raw_image_bytes = f.read()
            
            # Process image - convert to RGB and ensure proper size
            image_pil = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
            original_size = image_pil.size
            
            # Don't downscale person image to preserve original resolution
            # This follows the pattern from the original main.py
            logger.info(
                "Model image prepared",
                model_id=model_info['id'],
                filename=model_info['filename'],
                original_size=original_size,
                processed_size=image_pil.size,
            )
            
            # Encode the image to base64
            buffer = io.BytesIO()
            image_pil.save(buffer, format='PNG')
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return encoded_image
            
        except Exception as e:
            logger.error(
                "Failed to prepare model image",
                model_id=model_info['id'],
                path=str(model_path),
                error=str(e),
            )
            return ""
    
    async def call_virtual_try_on(
        self,
        person_image_bytes: str,
        product_image_bytes: str,
        sample_count: int = 1,
        safety_setting: str = "block_low_and_above",
        person_generation: str = "allow_adult",
    ) -> PredictResponse:
        """Call Virtual Try-On API."""
        instances = [{
            "personImage": {"image": {"bytesBase64Encoded": person_image_bytes}},
            "productImages": [{"image": {"bytesBase64Encoded": product_image_bytes}}],
        }]
        
        parameters = {
            "sampleCount": sample_count,
            "safetySetting": safety_setting,
            "personGeneration": person_generation,
        }
        
        try:
            start_time = time.time()
            
            response = self.client.predict(
                endpoint=self.model_endpoint,
                instances=instances,
                parameters=parameters,
            )
            
            api_time = time.time() - start_time
            
            logger.info(
                "Virtual Try-On API call completed",
                api_time=round(api_time, 2),
                sample_count=sample_count,
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Virtual Try-On API call failed",
                error=str(e),
            )
            raise
    
    async def process_image(
        self,
        record: ImageRecord,
        image_path: Path,
        output_dir: Path,
    ) -> ProcessingResult:
        """Process a single image for Virtual Try-On."""
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
            
            # Prepare apparel image
            apparel_image_bytes = self._prepare_apparel_image(image_path)
            
            # Analyze apparel to select appropriate model
            apparel_analysis = self.analyzer.classify_image(image_path)
            
            # Select model based on apparel characteristics and gender
            apparel_info = {
                'detected_items': apparel_analysis.detected_items,
                'target_gender': apparel_analysis.target_gender,
                'confidence': apparel_analysis.confidence,
                'reasoning': apparel_analysis.reasoning,
                'metadata': apparel_analysis.metadata,
            }
            selected_model = self.select_model(apparel_info)
            
            # Prepare model image
            model_image_bytes = self.prepare_model_image(selected_model)
            
            if not model_image_bytes:
                logger.error(
                    "Failed to prepare model image, skipping Virtual Try-On",
                    record_id=record.id,
                    selected_model=selected_model['id'],
                )
                return ProcessingResult(
                    record_id=record.id,
                    success=False,
                    error_message=f"Failed to prepare model image: {selected_model['id']}",
                )
            
            # Call Virtual Try-On API
            response = await self.call_virtual_try_on(
                person_image_bytes=model_image_bytes,
                product_image_bytes=apparel_image_bytes,
                sample_count=1,
            )
            
            # Process API response
            predictions = list(response.predictions)
            if not predictions or 'bytesBase64Encoded' not in predictions[0]:
                return ProcessingResult(
                    record_id=record.id,
                    success=False,
                    error_message="No valid predictions returned from API",
                )
            
            # Convert prediction to image
            result_image = prediction_to_pil_image(predictions[0])
            
            # Save result image
            result_path = record_output_dir / "result.jpg"
            result_image.save(result_path, 'JPEG', quality=95)
            
            # Save copy of selected model image for reference
            model_copy_path = record_output_dir / "selected_model.jpg"
            try:
                model_image = Image.open(selected_model['path'])
                model_image.save(model_copy_path, 'JPEG', quality=95)
            except Exception as e:
                logger.warning(
                    "Failed to save model image copy",
                    record_id=record.id,
                    model_id=selected_model['id'],
                    error=str(e),
                )
            
            # Analyze quality using Gemini
            quality_feedback = None
            quality_score = 0.8  # Default score
            
            try:
                # Use the actual model image path for quality analysis
                model_image_path = selected_model['path']
                
                quality_feedback = self.analyzer.analyze_virtual_try_on_quality(
                    result_image=result_image,
                    original_apparel=image_path,
                    model_image=model_image_path,
                )
                quality_score = quality_feedback.score
                
                logger.info(
                    "Virtual Try-On quality analysis completed",
                    record_id=record.id,
                    passed=quality_feedback.passed,
                    score=quality_score,
                    model_used=selected_model['id'],
                )
                
            except Exception as e:
                logger.warning(
                    "Quality analysis failed",
                    record_id=record.id,
                    error=str(e),
                )
            
            # Prepare metadata
            metadata = self.prepare_metadata(record, {
                'apparel_analysis': apparel_analysis.dict(),
                'selected_model': selected_model,
                'api_response_time': time.time(),
                'quality_score': quality_score,
                'quality_feedback': quality_feedback,
            })
            
            # Save metadata
            metadata_path = record_output_dir / "metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Upload to GCS if enabled
            gcs_path = None
            if self.settings.storage.enable_gcs_upload:
                gcs_path = await self._upload_to_gcs(
                    result_path,
                    metadata_path,
                    model_copy_path,
                    record,
                    image_path,
                    selected_model,
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
                "Virtual Try-On processing failed",
                record_id=record.id,
                error=str(e),
            )
            return ProcessingResult(
                record_id=record.id,
                success=False,
                error_message=str(e),
            )
    
    def _prepare_apparel_image(self, image_path: Path) -> str:
        """Prepare apparel image for API call."""
        with open(image_path, 'rb') as f:
            raw_image_bytes = f.read()
        
        # Process image - convert to RGB and resize
        image_pil = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
        original_size = image_pil.size
        
        # Apply thumbnail to maintain aspect ratio within 1024x1024
        image_pil.thumbnail((1024, 1024))
        
        logger.info(
            "Apparel image prepared",
            original_size=original_size,
            processed_size=image_pil.size,
        )
        
        # Encode to base64
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def _upload_to_gcs(
        self,
        result_path: Path,
        metadata_path: Path,
        model_copy_path: Path,
        record: ImageRecord,
        apparel_image_path: Path,
        selected_model: Dict[str, Any],
    ) -> Optional[str]:
        """Upload results to Google Cloud Storage."""
        try:
            from google.cloud import storage
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.settings.google_cloud.storage_bucket)
            
            # Generate GCS paths
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_path = f"{self.settings.storage.gcs_virtual_try_on_path}/outputs/{timestamp}_{record.id}"
            inputs_path = f"{self.settings.storage.gcs_virtual_try_on_path}/inputs/{timestamp}_{record.id}"
            
            # Upload input images (apparel and model)
            apparel_input_blob = bucket.blob(f"{inputs_path}/apparel_image.jpg")
            apparel_input_blob.upload_from_filename(str(apparel_image_path))
            
            # Upload model image to inputs
            model_input_blob = bucket.blob(f"{inputs_path}/model_image.jpg")
            model_input_blob.upload_from_filename(str(selected_model['path']))
            
            # Upload result image
            result_blob = bucket.blob(f"{base_path}/result.jpg")
            result_blob.upload_from_filename(str(result_path))
            
            # Upload selected model image copy to outputs
            if model_copy_path.exists():
                model_blob = bucket.blob(f"{base_path}/selected_model.jpg")
                model_blob.upload_from_filename(str(model_copy_path))
            
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