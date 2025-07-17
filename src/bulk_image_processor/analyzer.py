"""Gemini-based image analyzer for classification and feedback."""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import google.generativeai as genai
import structlog
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings
from .utils import async_retry_with_backoff, encode_image_to_base64

logger = structlog.get_logger(__name__)


class ImageCategory(str, Enum):
    """Image classification categories."""
    APPAREL = "apparel"
    PRODUCT = "product"
    UNKNOWN = "unknown"


class ClassificationResult(BaseModel):
    """Result of image classification."""
    category: ImageCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    detected_items: List[str] = Field(default_factory=list)
    target_gender: Optional[str] = None  # "women", "men", or None for unisex/unknown
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackResult(BaseModel):
    """Result of image quality feedback."""
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GeminiAnalyzer:
    """Gemini-based image analyzer for classification and feedback."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Initialize Gemini client."""
        try:
            genai.configure(api_key=self.settings.gemini.api_key)
            
            # Configure safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Initialize model
            self.client = genai.GenerativeModel(
                model_name=self.settings.gemini.model_name,
                safety_settings=safety_settings,
            )
            
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Gemini client", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def classify_image(
        self,
        image: Union[Image.Image, Path, str],
        additional_context: Optional[str] = None,
    ) -> ClassificationResult:
        """Classify an image as apparel or product."""
        start_time = time.time()
        
        try:
            # Prepare image for analysis
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            
            # Create classification prompt
            prompt = self._create_classification_prompt(additional_context)
            
            # Generate response
            response = self.client.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=self.settings.gemini.temperature,
                    max_output_tokens=self.settings.gemini.max_output_tokens,
                ),
            )
            
            # Parse response
            result = self._parse_classification_response(response.text)
            
            analysis_time = time.time() - start_time
            
            logger.info(
                "Image classification completed",
                category=result.category,
                confidence=result.confidence,
                analysis_time=round(analysis_time, 2),
                detected_items=result.detected_items,
            )
            
            return result
            
        except Exception as e:
            analysis_time = time.time() - start_time
            logger.error(
                "Image classification failed",
                error=str(e),
                analysis_time=round(analysis_time, 2),
            )
            raise
    
    def _create_classification_prompt(self, additional_context: Optional[str] = None) -> str:
        """Create prompt for image classification."""
        base_prompt = """
        Analyze this image and classify it as either "apparel" or "product".
        
        CLASSIFICATION CRITERIA:
        - "apparel": Clothing items, accessories, shoes, bags, jewelry, or anything that can be worn on a person
        - "product": All other items including furniture, electronics, home goods, food, tools, etc.
        
        IMPORTANT GUIDELINES:
        - Focus on the main subject of the image
        - If multiple items are present, classify based on the primary/dominant item
        - Shoes, bags, and jewelry should be classified as "apparel"
        - Consider the context and typical use of the item
        
        GENDER DETECTION FOR APPAREL:
        If the item is classified as "apparel", also determine the target gender:
        - "women": Women's clothing, feminine styles, dresses, skirts, blouses, women's shoes, women's accessories
        - "men": Men's clothing, masculine styles, suits, ties, men's shirts, men's shoes, men's accessories
        - "unisex": Items that can be worn by any gender (some t-shirts, jeans, sneakers, etc.)
        
        Please respond with a JSON object containing:
        {
            "category": "apparel" or "product",
            "confidence": float between 0.0 and 1.0,
            "reasoning": "detailed explanation of your classification decision",
            "detected_items": ["list", "of", "items", "detected", "in", "image"],
            "target_gender": "women", "men", "unisex", or null (only for apparel items),
            "metadata": {
                "primary_color": "dominant color if applicable",
                "style": "style description if applicable",
                "material": "material type if identifiable"
            }
        }
        """
        
        if additional_context:
            base_prompt += f"\n\nAdditional context: {additional_context}"
        
        return base_prompt
    
    def _parse_classification_response(self, response_text: str) -> ClassificationResult:
        """Parse classification response from Gemini."""
        try:
            # Clean response text
            response_text = response_text.strip()
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            response_data = json.loads(json_str)
            
            # Validate and create result
            category = response_data.get("category", "unknown").lower()
            if category not in ["apparel", "product"]:
                category = "unknown"
            
            return ClassificationResult(
                category=ImageCategory(category),
                confidence=min(max(float(response_data.get("confidence", 0.5)), 0.0), 1.0),
                reasoning=response_data.get("reasoning", "No reasoning provided"),
                detected_items=response_data.get("detected_items", []),
                target_gender=response_data.get("target_gender"),
                metadata=response_data.get("metadata", {}),
            )
            
        except Exception as e:
            logger.warning("Failed to parse classification response", error=str(e))
            return ClassificationResult(
                category=ImageCategory.UNKNOWN,
                confidence=0.0,
                reasoning=f"Failed to parse response: {str(e)}",
                detected_items=[],
                target_gender=None,
                metadata={},
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def analyze_virtual_try_on_quality(
        self,
        result_image: Union[Image.Image, Path, str],
        original_apparel: Union[Image.Image, Path, str],
        model_image: Union[Image.Image, Path, str],
    ) -> FeedbackResult:
        """Analyze quality of Virtual Try-On result."""
        start_time = time.time()
        
        try:
            # Prepare images
            if isinstance(result_image, (str, Path)):
                result_image = Image.open(result_image)
            if isinstance(original_apparel, (str, Path)):
                original_apparel = Image.open(original_apparel)
            if isinstance(model_image, (str, Path)):
                model_image = Image.open(model_image)
            
            # Create feedback prompt
            prompt = self._create_vto_feedback_prompt()
            
            # Generate response
            response = self.client.generate_content(
                [prompt, result_image, original_apparel, model_image],
                generation_config=genai.types.GenerationConfig(
                    temperature=self.settings.gemini.temperature,
                    max_output_tokens=self.settings.gemini.max_output_tokens,
                ),
            )
            
            # Parse response
            result = self._parse_feedback_response(response.text)
            
            analysis_time = time.time() - start_time
            
            logger.info(
                "Virtual Try-On quality analysis completed",
                passed=result.passed,
                score=result.score,
                analysis_time=round(analysis_time, 2),
            )
            
            return result
            
        except Exception as e:
            analysis_time = time.time() - start_time
            logger.error(
                "Virtual Try-On quality analysis failed",
                error=str(e),
                analysis_time=round(analysis_time, 2),
            )
            raise
    
    def _create_vto_feedback_prompt(self) -> str:
        """Create prompt for Virtual Try-On feedback."""
        return """
        Analyze the quality of this Virtual Try-On result. You will see three images:
        1. The result image (person wearing the apparel)
        2. The original apparel item
        3. The model/person image
        
        QUALITY CRITERIA:
        - Realistic placement: Is the apparel positioned naturally on the person?
        - Proper fit: Does the apparel fit the person's body appropriately?
        - Color accuracy: Are the colors consistent with the original apparel?
        - Visual coherence: Does the result look realistic and believable?
        - Lighting consistency: Does the lighting match between the person and apparel?
        - Edge quality: Are the edges of the apparel clean and well-integrated?
        
        SCORING GUIDELINES:
        - Score 0.8-1.0: Excellent quality, very realistic
        - Score 0.6-0.8: Good quality, minor issues
        - Score 0.4-0.6: Acceptable quality, noticeable issues
        - Score 0.2-0.4: Poor quality, significant problems
        - Score 0.0-0.2: Unacceptable quality, major failures
        
        Pass threshold: 0.6 or higher
        
        Please respond with a JSON object containing:
        {
            "passed": true or false,
            "score": float between 0.0 and 1.0,
            "reasoning": "detailed explanation of the quality assessment",
            "suggestions": ["list", "of", "improvement", "suggestions"],
            "metadata": {
                "fit_quality": "assessment of how well the apparel fits",
                "color_accuracy": "assessment of color matching",
                "realism": "assessment of overall realism",
                "main_issues": ["list", "of", "main", "problems"]
            }
        }
        """
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def analyze_product_recontext_quality(
        self,
        result_image: Union[Image.Image, Path, str],
        original_product: Union[Image.Image, Path, str],
        generated_prompt: str,
    ) -> FeedbackResult:
        """Analyze quality of Product Recontext result."""
        start_time = time.time()
        
        try:
            # Prepare images
            if isinstance(result_image, (str, Path)):
                result_image = Image.open(result_image)
            if isinstance(original_product, (str, Path)):
                original_product = Image.open(original_product)
            
            # Create feedback prompt
            prompt = self._create_product_recontext_feedback_prompt(generated_prompt)
            
            # Generate response
            response = self.client.generate_content(
                [prompt, result_image, original_product],
                generation_config=genai.types.GenerationConfig(
                    temperature=self.settings.gemini.temperature,
                    max_output_tokens=self.settings.gemini.max_output_tokens,
                ),
            )
            
            # Parse response
            result = self._parse_feedback_response(response.text)
            
            analysis_time = time.time() - start_time
            
            logger.info(
                "Product Recontext quality analysis completed",
                passed=result.passed,
                score=result.score,
                analysis_time=round(analysis_time, 2),
            )
            
            return result
            
        except Exception as e:
            analysis_time = time.time() - start_time
            logger.error(
                "Product Recontext quality analysis failed",
                error=str(e),
                analysis_time=round(analysis_time, 2),
            )
            raise
    
    def _create_product_recontext_feedback_prompt(self, generated_prompt: str) -> str:
        """Create prompt for Product Recontext feedback."""
        return f"""
        Analyze the quality of this Product Recontextualization result. You will see:
        1. The result image (product in new context/scene)
        2. The original product image
        
        Generated prompt used: "{generated_prompt}"
        
        QUALITY CRITERIA:
        - Product visibility: Is the product clearly visible and well-presented?
        - Context appropriateness: Is the product placed in an appropriate high-end retail context?
        - Visual quality: Is the overall image quality high and professional?
        - Lighting: Is the lighting flattering and appropriate for retail?
        - Composition: Is the product well-positioned and composed in the scene?
        - Brand appeal: Does the result enhance the product's appeal for high-end retail?
        
        SCORING GUIDELINES:
        - Score 0.8-1.0: Excellent quality, highly appealing for retail
        - Score 0.6-0.8: Good quality, suitable for retail with minor issues
        - Score 0.4-0.6: Acceptable quality, some improvements needed
        - Score 0.2-0.4: Poor quality, significant issues
        - Score 0.0-0.2: Unacceptable quality, major failures
        
        Pass threshold: 0.6 or higher
        
        Please respond with a JSON object containing:
        {{
            "passed": true or false,
            "score": float between 0.0 and 1.0,
            "reasoning": "detailed explanation of the quality assessment",
            "suggestions": ["list", "of", "improvement", "suggestions"],
            "metadata": {{
                "product_visibility": "assessment of how well the product is showcased",
                "context_quality": "assessment of the context/scene quality",
                "retail_appeal": "assessment of appeal for high-end retail",
                "main_issues": ["list", "of", "main", "problems"]
            }}
        }}
        """
    
    def _parse_feedback_response(self, response_text: str) -> FeedbackResult:
        """Parse feedback response from Gemini."""
        try:
            # Clean response text
            response_text = response_text.strip()
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            response_data = json.loads(json_str)
            
            # Validate and create result
            passed = bool(response_data.get("passed", False))
            score = min(max(float(response_data.get("score", 0.0)), 0.0), 1.0)
            
            return FeedbackResult(
                passed=passed,
                score=score,
                reasoning=response_data.get("reasoning", "No reasoning provided"),
                suggestions=response_data.get("suggestions", []),
                metadata=response_data.get("metadata", {}),
            )
            
        except Exception as e:
            logger.warning("Failed to parse feedback response", error=str(e))
            return FeedbackResult(
                passed=False,
                score=0.0,
                reasoning=f"Failed to parse response: {str(e)}",
                suggestions=[],
                metadata={},
            )
    
    def generate_product_recontext_prompt(
        self,
        product_image: Union[Image.Image, Path, str],
        product_description: Optional[str] = None,
    ) -> str:
        """Generate a compelling prompt for Product Recontext."""
        start_time = time.time()
        
        try:
            # Prepare image
            if isinstance(product_image, (str, Path)):
                product_image = Image.open(product_image)
            
            # Create prompt generation prompt
            prompt = self._create_prompt_generation_prompt(product_description)
            
            # Generate response
            response = self.client.generate_content(
                [prompt, product_image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent prompts
                    max_output_tokens=512,
                ),
            )
            
            # Extract and clean the generated prompt
            generated_prompt = response.text.strip()
            
            # Remove any quotation marks or formatting
            generated_prompt = generated_prompt.strip('"').strip("'")
            
            generation_time = time.time() - start_time
            
            logger.info(
                "Product recontext prompt generated",
                prompt_length=len(generated_prompt),
                generation_time=round(generation_time, 2),
            )
            
            return generated_prompt
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(
                "Product recontext prompt generation failed",
                error=str(e),
                generation_time=round(generation_time, 2),
            )
            raise
    
    def _create_prompt_generation_prompt(self, product_description: Optional[str] = None) -> str:
        """Create prompt for generating Product Recontext prompts."""
        base_prompt = """
        Analyze this product image and generate a compelling prompt for high-end retail product recontextualization.
        
        REQUIREMENTS:
        - Create a scene that positions the product for luxury/high-end retail websites
        - Focus on premium, aspirational, and elegant contexts
        - Ensure the product remains the focal point
        - Use sophisticated, upscale environments
        - Consider lighting that enhances the product's appeal
        - Keep the prompt concise but descriptive (2-3 sentences maximum)
        
        STYLE GUIDELINES:
        - Professional photography style
        - Clean, minimalist backgrounds when appropriate
        - Elegant props and settings
        - Sophisticated color palettes
        - Premium materials and textures in the scene
        
        AVOID:
        - Cluttered or busy backgrounds
        - Cheap or low-quality contexts
        - Distracting elements
        - Overly complex scenes
        
        Generate ONLY the prompt text, no additional explanation or formatting.
        """
        
        if product_description:
            base_prompt += f"\n\nProduct context: {product_description}"
        
        return base_prompt


def create_analyzer(settings: Settings) -> GeminiAnalyzer:
    """Create a Gemini analyzer instance."""
    return GeminiAnalyzer(settings)