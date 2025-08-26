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
        """Load predefined model images for Virtual Try-On with fashion coordination."""
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
        
        # Create model info for each image file based on naming convention:
        # Format: {number}_{gender}_{bottom_color}_{bottom_type}_{top_color}_{top_item}.jpg
        # Or: {number}_{gender}_{dress_color}_dress.jpg
        self.model_images = []
        
        for model_file in sorted(model_files):
            try:
                # Parse filename: e.g., "1_woman_black_trousers_white_shirt.jpg"
                filename_parts = model_file.stem.split('_')
                
                if len(filename_parts) < 4:
                    logger.warning(
                        "Skipping model file with unexpected naming format",
                        filename=model_file.name,
                        expected_format="number_gender_outfit_details.jpg",
                    )
                    continue
                
                model_number = filename_parts[0]
                gender = filename_parts[1]
                
                # Validate gender
                if gender not in ['man', 'woman']:
                    logger.warning(
                        "Skipping model file with invalid gender",
                        filename=model_file.name,
                        gender=gender,
                        expected_genders=['man', 'woman'],
                    )
                    continue
                
                # Parse outfit details
                outfit_details = self._parse_outfit_details(filename_parts[2:])
                
                # Create model info
                model_info = {
                    'id': f'model_{model_number}_{gender}',
                    'description': f'Professional {gender} model {model_number} wearing {outfit_details["description"]}',
                    'path': model_file,
                    'gender': gender,
                    'number': model_number,
                    'filename': model_file.name,
                    'outfit': outfit_details,
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
            'woman': [m for m in self.model_images if m['gender'] == 'woman'],
            'man': [m for m in self.model_images if m['gender'] == 'man'],
        }
        
        # Also create lookup by outfit characteristics for fashion coordination
        self.models_by_outfit = {}
        for model in self.model_images:
            outfit = model['outfit']
            gender = model['gender']
            
            # Create keys for different outfit characteristics
            keys = [
                f"{gender}_{outfit.get('bottom_color', 'unknown')}_bottoms",
                f"{gender}_{outfit.get('top_color', 'unknown')}_tops",
                f"{gender}_{outfit.get('style', 'unknown')}_style",
            ]
            
            for key in keys:
                if key not in self.models_by_outfit:
                    self.models_by_outfit[key] = []
                self.models_by_outfit[key].append(model)
        
        logger.info(
            "Model images loaded with fashion coordination",
            total_count=len(self.model_images),
            woman_models=len(self.models_by_gender['woman']),
            man_models=len(self.models_by_gender['man']),
            models=[m['filename'] for m in self.model_images],
            outfit_categories=len(self.models_by_outfit),
        )
    
    def _parse_outfit_details(self, parts: List[str]) -> Dict[str, Any]:
        """Parse outfit details from filename parts.
        
        Examples:
        - ['black', 'trousers', 'white', 'shirt'] -> black trousers, white shirt
        - ['grey', 'dress'] -> grey dress
        - ['blue', 'jeans', 'white', 'tshirt'] -> blue jeans, white t-shirt
        """
        if len(parts) == 2 and parts[1] == 'dress':
            # Handle dress format: color_dress
            return {
                'type': 'dress',
                'dress_color': parts[0],
                'style': 'dress',
                'description': f'{parts[0]} dress',
                'bottom_color': parts[0],
                'bottom_type': 'dress',
                'top_color': parts[0],
                'top_type': 'dress',
            }
        elif len(parts) >= 4:
            # Handle full outfit format: bottom_color_bottom_type_top_color_top_item
            return {
                'type': 'separate_pieces',
                'bottom_color': parts[0],
                'bottom_type': parts[1],
                'top_color': parts[2], 
                'top_type': parts[3],
                'style': f'{parts[1]}_{parts[3]}',
                'description': f'{parts[0]} {parts[1]}, {parts[2]} {parts[3]}',
            }
        else:
            # Fallback for unexpected formats
            return {
                'type': 'unknown',
                'style': 'casual',
                'description': ' '.join(parts),
                'bottom_color': 'neutral',
                'bottom_type': 'unknown',
                'top_color': 'neutral', 
                'top_type': 'unknown',
            }
    
    def get_processor_type(self) -> str:
        """Get processor type name."""
        return "virtual_try_on"
    
    async def select_model_with_fashion_coordination(
        self, 
        apparel_info: Dict[str, Any], 
        apparel_image_path: Path,
        exclude_models: List[str] = None
    ) -> Dict[str, Any]:
        """Select model using Gemini-powered fashion coordination."""
        try:
            # For unisex items, provide all models to Gemini for selection
            target_gender = apparel_info.get('target_gender')
            if target_gender == 'unisex':
                logger.info(
                    "Unisex item detected, providing all models to Gemini for selection",
                    detected_items=apparel_info.get('detected_items', [])
                )
                # Use all models for unisex items
                available_models = self.model_images
            else:
                # Use models matching the target gender
                available_models = self.model_images
            
            # Get fashion coordination recommendation from Gemini
            recommendation = await self.analyzer.recommend_fashion_coordination(
                apparel_image_path, apparel_info, available_models, exclude_models
            )
            
            # Find the recommended model
            recommended_model = None
            for model in self.model_images:
                if model['id'] == recommendation['recommended_model_id']:
                    recommended_model = model
                    break
            
            if recommended_model:
                # Add fashion reasoning to model for metadata
                recommended_model['fashion_reasoning'] = {
                    'reasoning': recommendation['reasoning'],
                    'coordination_score': recommendation.get('coordination_score', 0),
                    'color_analysis': recommendation.get('color_analysis', ''),
                    'style_analysis': recommendation.get('style_analysis', ''),
                    'fashion_tips': recommendation.get('fashion_tips', ''),
                }
                
                logger.info(
                    "Model selected with fashion coordination",
                    model_id=recommended_model['id'],
                    reasoning=recommendation['reasoning'],
                    coordination_score=recommendation.get('coordination_score', 0),
                )
                return recommended_model
            else:
                logger.warning(
                    "Recommended model not found, falling back to legacy selection",
                    recommended_id=recommendation['recommended_model_id'],
                )
                # Fallback to legacy selection
                return self.select_model(apparel_info, exclude_models)
                
        except Exception as e:
            logger.warning(
                "Fashion coordination failed, falling back to legacy selection",
                error=str(e),
            )
            # Fallback to legacy selection
            return self.select_model(apparel_info, exclude_models)
    
    def select_model(self, apparel_info: Dict[str, Any], exclude_models: List[str] = None) -> Dict[str, Any]:
        """Select appropriate model based on apparel characteristics and gender (legacy method).
        
        Args:
            apparel_info: Information about the apparel item
            exclude_models: List of model IDs to exclude from selection (for retries)
        """
        detected_items = apparel_info.get('detected_items', [])
        target_gender = apparel_info.get('target_gender')
        exclude_models = exclude_models or []
        
        # Handle unisex items by selecting from all available models
        if target_gender == 'unisex':
            logger.info(
                "Unisex item detected, selecting from all available models",
                original_gender=target_gender,
                detected_items=detected_items
            )
            # Combine all models from both genders for unisex items
            all_models = self.models_by_gender.get('woman', []) + self.models_by_gender.get('man', [])
            suitable_models = [m for m in all_models if m['id'] not in exclude_models]
            if suitable_models:
                selected_model = random.choice(suitable_models)
                logger.info(
                    "Model selected for unisex item",
                    model_id=selected_model['id'],
                    model_gender=selected_model['gender'],
                    detected_items=detected_items,
                    excluded_count=len(exclude_models),
                )
                return selected_model
        
        # First try to use the gender detected by Gemini
        if target_gender and target_gender in self.models_by_gender:
            suitable_models = [m for m in self.models_by_gender[target_gender] 
                             if m['id'] not in exclude_models]
            if suitable_models:
                selected_model = random.choice(suitable_models)
                logger.info(
                    "Model selected by gender",
                    model_id=selected_model['id'],
                    gender=target_gender,
                    detected_items=detected_items,
                    excluded_count=len(exclude_models),
                )
                return selected_model
        
        # Fallback to heuristic-based selection if gender detection failed
        if any(item.lower() in ['dress', 'blouse', 'skirt', 'heels'] for item in detected_items):
            preferred_gender = 'woman'
        elif any(item.lower() in ['suit', 'tie', 'shirt'] for item in detected_items):
            preferred_gender = 'man'
        else:
            # Default to any available model
            preferred_gender = None
        
        # Try to select from preferred gender
        if preferred_gender and preferred_gender in self.models_by_gender:
            suitable_models = [m for m in self.models_by_gender[preferred_gender]
                             if m['id'] not in exclude_models]
            if suitable_models:
                selected_model = random.choice(suitable_models)
                logger.info(
                    "Model selected by heuristic",
                    model_id=selected_model['id'],
                    gender=preferred_gender,
                    detected_items=detected_items,
                    excluded_count=len(exclude_models),
                )
                return selected_model
        
        # CRITICAL: Do NOT fallback to different gender models
        # If we can't find a model of the correct gender, fail the selection
        # This maintains gender consistency across retries
        
        logger.error(
            "No more models available for selection while maintaining gender consistency", 
            target_gender=target_gender,
            excluded_models=exclude_models,
            available_genders=list(self.models_by_gender.keys()),
            total_models_per_gender={g: len(models) for g, models in self.models_by_gender.items()},
        )
        
        raise ValueError(
            f"All models of gender '{target_gender}' have been tried. "
            f"Excluded: {exclude_models}. "
            f"Gender consistency maintained - will not try different gender models."
        )
    
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
            
            # Use executor to avoid blocking the async event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.predict(
                    endpoint=self.model_endpoint,
                    instances=instances,
                    parameters=parameters,
                )
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
        """Process a single image for Virtual Try-On with API retry and model retry logic."""
        # Create a single output directory for this record that will be reused across retries
        record_output_dir = self.create_output_directory(output_dir, record)
        
        # Track which models we've already tried and retry counts per model
        tried_models = []
        model_retry_counts = {}
        max_api_retries_per_model = 2  # Retry same model up to 2 times for API issues
        max_model_attempts = min(len(self.model_images), 3)  # Try up to 3 different models
        
        total_attempts = 0
        max_total_attempts = max_model_attempts * (max_api_retries_per_model + 1)
        
        # Track the best result so far (even if not perfect)
        best_result = None
        best_score = 0.0
        
        while total_attempts < max_total_attempts:
            total_attempts += 1
            
            result = await self._process_with_model(
                record, image_path, record_output_dir, tried_models
            )
            
            # Track the best result even if not perfect
            if result.success and hasattr(result, 'quality_score'):
                current_score = result.quality_score
                if current_score > best_score:
                    best_score = current_score
                    best_result = result
            
            # Check if garment was successfully applied
            if result.success and hasattr(result, 'metadata') and result.metadata:
                quality_feedback = result.metadata.get('quality_feedback')
                current_model = result.metadata.get('selected_model', {}).get('id', 'unknown')
                
                if quality_feedback and hasattr(quality_feedback, 'garment_applied'):
                    # Check if garment was applied AND length is accurate
                    garment_applied = quality_feedback.garment_applied
                    length_accurate = getattr(quality_feedback, 'length_accurate', True)
                    
                    if not garment_applied:
                        # Garment not applied - could be API issue, try same model again first
                        retry_count = model_retry_counts.get(current_model, 0)
                        
                        if retry_count < max_api_retries_per_model:
                            # Retry same model (API might have failed)
                            model_retry_counts[current_model] = retry_count + 1
                            
                            # Save failed attempt IMAGE and metadata
                            failed_attempt_file = record_output_dir / f"failed_attempt_{total_attempts}.json"
                            failed_attempt_image = record_output_dir / f"failed_attempt_{total_attempts}.jpg"
                            
                            # Save the failed result image
                            if hasattr(result, '_temp_result_data'):
                                temp_data = result._temp_result_data
                                failed_image = temp_data['result_image']
                                failed_image.save(failed_attempt_image, 'JPEG', quality=95)
                            
                            failed_attempt_data = {
                                'total_attempt': total_attempts,
                                'retry_attempt': retry_count + 1,
                                'model_used': current_model,
                                'quality_feedback': str(quality_feedback),
                                'quality_score': getattr(result, 'quality_score', 0.0),
                                'failure_reason': 'garment_not_applied',
                                'gemini_interpretation': quality_feedback.reasoning if hasattr(quality_feedback, 'reasoning') else 'No reasoning provided',
                                'failed_image': str(failed_attempt_image) if failed_attempt_image.exists() else None,
                            }
                            
                            import json
                            with open(failed_attempt_file, 'w') as f:
                                json.dump(failed_attempt_data, f, indent=2, default=str)
                            
                            logger.warning(
                                "Garment not applied - retrying same model (API issue)",
                                record_id=record.id,
                                model_id=current_model,
                                retry_attempt=retry_count + 1,
                                max_api_retries=max_api_retries_per_model,
                                total_attempt=total_attempts,
                                failure_saved=str(failed_attempt_file),
                                failed_image_saved=str(failed_attempt_image),
                            )
                            
                            # Continue to retry same model
                            continue
                        else:
                            # Exhausted retries for this model, switch to different model
                            tried_models.append(current_model)
                            
                            # Save the final failed attempt for this model
                            failed_attempt_file = record_output_dir / f"failed_attempt_{total_attempts}.json"
                            failed_attempt_image = record_output_dir / f"failed_attempt_{total_attempts}.jpg"
                            
                            # Save the failed result image
                            if hasattr(result, '_temp_result_data'):
                                temp_data = result._temp_result_data
                                failed_image = temp_data['result_image']
                                failed_image.save(failed_attempt_image, 'JPEG', quality=95)
                            
                            failed_attempt_data = {
                                'total_attempt': total_attempts,
                                'model_used': current_model,
                                'api_retries_exhausted': retry_count + 1,
                                'quality_feedback': str(quality_feedback),
                                'quality_score': getattr(result, 'quality_score', 0.0),
                                'failure_reason': 'garment_not_applied_switching_model',
                                'gemini_interpretation': quality_feedback.reasoning if hasattr(quality_feedback, 'reasoning') else 'No reasoning provided',
                                'failed_image': str(failed_attempt_image) if failed_attempt_image.exists() else None,
                            }
                            
                            import json
                            with open(failed_attempt_file, 'w') as f:
                                json.dump(failed_attempt_data, f, indent=2, default=str)
                            
                            logger.warning(
                                "Garment not applied after all retries, switching to different model",
                                record_id=record.id,
                                failed_model=current_model,
                                api_retries=retry_count + 1,
                                switching_to_new_model=True,
                                failure_details_saved=str(failed_attempt_file),
                                failed_image_saved=str(failed_attempt_image),
                            )
                            
                            # Continue with new model
                            continue
                    
                    elif garment_applied and not length_accurate:
                        # Garment applied but length is wrong - switch to different model immediately
                        # (No point retrying same model for length issues)
                        tried_models.append(current_model)
                        
                        # Save failed attempt due to incorrect length
                        failed_attempt_file = record_output_dir / f"failed_attempt_{total_attempts}.json"
                        failed_attempt_image = record_output_dir / f"failed_attempt_{total_attempts}.jpg"
                        
                        # Save the failed result image
                        if hasattr(result, '_temp_result_data'):
                            temp_data = result._temp_result_data
                            failed_image = temp_data['result_image']
                            failed_image.save(failed_attempt_image, 'JPEG', quality=95)
                        
                        failed_attempt_data = {
                            'total_attempt': total_attempts,
                            'model_used': current_model,
                            'quality_feedback': str(quality_feedback),
                            'quality_score': getattr(result, 'quality_score', 0.0),
                            'failure_reason': 'incorrect_garment_length',
                            'length_issue': getattr(quality_feedback, 'length_issue', 'Length mismatch detected'),
                            'gemini_interpretation': quality_feedback.reasoning if hasattr(quality_feedback, 'reasoning') else 'No reasoning provided',
                            'failed_image': str(failed_attempt_image) if failed_attempt_image.exists() else None,
                        }
                        
                        import json
                        with open(failed_attempt_file, 'w') as f:
                            json.dump(failed_attempt_data, f, indent=2, default=str)
                        
                        logger.warning(
                            "Garment applied but length incorrect - switching to different model",
                            record_id=record.id,
                            model_id=current_model,
                            length_issue=getattr(quality_feedback, 'length_issue', 'Length mismatch'),
                            quality_score=getattr(result, 'quality_score', 0.0),
                            total_attempt=total_attempts,
                            failure_saved=str(failed_attempt_file),
                            failed_image_saved=str(failed_attempt_image),
                        )
                        
                        # Continue with new model
                        continue
                    
                    # If garment was applied AND length is accurate, accept as final result
                    # Save the successful result
                    if hasattr(result, '_temp_result_data'):
                        await self._save_successful_result(result, record, image_path)
                    
                    logger.info(
                        "Garment successfully applied with correct length, accepting result",
                        record_id=record.id,
                        model_id=current_model,
                        quality_score=quality_feedback.score,
                        length_accurate=length_accurate,
                        total_attempts=total_attempts,
                    )
                    
                    return result
            
            # If we get here, either success or non-retry failure
            if result.success and hasattr(result, '_temp_result_data'):
                await self._save_successful_result(result, record, image_path)
            
            return result
        
        # Exhausted all model attempts - use best result if available
        if best_result is not None:
            logger.warning(
                "All retry attempts exhausted, using best result with low confidence",
                record_id=record.id,
                best_score=best_score,
                tried_models=tried_models,
            )
            
            # Mark result as low confidence
            if hasattr(best_result, '_temp_result_data'):
                temp_data = best_result._temp_result_data
                # Save as result_maybe.jpg instead of result.jpg
                result_maybe_path = temp_data['result_path'].parent / "result_maybe.jpg"
                temp_data['result_path'] = result_maybe_path
                best_result._temp_result_data = temp_data
                
                # Add low confidence flag to metadata
                if best_result.metadata:
                    best_result.metadata['low_confidence'] = True
                    best_result.metadata['confidence_reason'] = 'Failed quality checks but best available'
                
                # Save the best result even if not perfect
                await self._save_successful_result(best_result, record, image_path)
                
            return best_result
        
        # No result at all - this should rarely happen
        return ProcessingResult(
            record_id=record.id,
            success=False,
            error_message=f"Failed after trying {len(tried_models)} different models with retries",
        )
    
    async def _save_successful_result(self, result: ProcessingResult, record: ImageRecord, image_path: Path) -> None:
        """Save the successful result files and upload to GCS."""
        temp_data = result._temp_result_data
        result_image = temp_data['result_image']
        result_path = temp_data['result_path']
        model_copy_path = temp_data['model_copy_path']
        selected_model = temp_data['selected_model']
        
        try:
            # Save final result image
            result_image.save(result_path, 'JPEG', quality=95)
            
            # Save copy of selected model image for reference
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
            
            # Save metadata
            metadata_path = result_path.parent / "metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(result.metadata, f, indent=2, default=str)
            
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
            
            # Update result with final paths
            result.output_path = result_path
            result.gcs_path = gcs_path
            
            logger.info(
                "Final result saved successfully",
                record_id=record.id,
                result_path=str(result_path),
                quality_score=result.quality_score,
            )
            
        except Exception as e:
            logger.error(
                "Failed to save successful result",
                record_id=record.id,
                error=str(e),
            )
            result.success = False
            result.error_message = f"Failed to save result: {str(e)}"
    
    async def _process_with_model(
        self,
        record: ImageRecord,
        image_path: Path,
        record_output_dir: Path,  # Now receiving the pre-created directory
        exclude_models: List[str] = None,
    ) -> ProcessingResult:
        """Process a single image with a specific model selection."""
        try:
            # Validate input image
            if not self.validate_image(image_path):
                return ProcessingResult(
                    record_id=record.id,
                    success=False,
                    error_message="Invalid input image",
                )
            
            # Use the provided output directory (created once in parent method)
            
            # Prepare apparel image
            apparel_image_bytes = self._prepare_apparel_image(image_path)
            
            # Analyze apparel to select appropriate model
            apparel_analysis = await self.analyzer.classify_image(image_path)
            
            # Select model based on apparel characteristics and gender
            apparel_info = {
                'detected_items': apparel_analysis.detected_items,
                'target_gender': apparel_analysis.target_gender,
                'confidence': apparel_analysis.confidence,
                'reasoning': apparel_analysis.reasoning,
                'metadata': apparel_analysis.metadata,
            }
            # Use fashion coordination for model selection
            selected_model = await self.select_model_with_fashion_coordination(
                apparel_info, image_path, exclude_models=exclude_models
            )
            
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
            
            # Don't save final result image yet - wait for retry logic
            # But store it temporarily for potential saving
            
            # Analyze quality using Gemini
            quality_feedback = None
            quality_score = 0.8  # Default score
            
            try:
                # Use the actual model image path for quality analysis
                model_image_path = selected_model['path']
                
                quality_feedback = await self.analyzer.analyze_virtual_try_on_quality(
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
            
            # Only save result if it's successful (will be saved later after retry logic)
            # For now, just store the result_image and paths for potential saving
            temp_result_data = {
                'result_image': result_image,
                'result_path': record_output_dir / "result.jpg",
                'model_copy_path': record_output_dir / "selected_model.jpg",
                'selected_model': selected_model,
            }
            
            # Prepare result for return (but don't save files yet)
            temp_metadata = self.prepare_metadata(record, {
                'apparel_analysis': apparel_analysis.model_dump(),
                'selected_model': selected_model,
                'api_response_time': time.time(),
                'quality_score': quality_score,
                'quality_feedback': quality_feedback,
                'fashion_coordination': getattr(selected_model, 'fashion_reasoning', None),
            })
            
            # Return result with temp data for retry logic processing
            result = ProcessingResult(
                record_id=record.id,
                success=True,
                output_path=None,  # Will be set after retry logic
                gcs_path=None,     # Will be set after retry logic
                metadata=temp_metadata,
                quality_score=quality_score,
                feedback=str(quality_feedback) if quality_feedback else None,
            )
            
            # Store temp result data for saving later
            result._temp_result_data = temp_result_data
            return result
            
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
            
            # Note: Public access needs to be configured at bucket/folder level 
            # due to uniform bucket-level access being enabled.
            # The GCS URI will be publicly accessible if bucket IAM is configured correctly.
            
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