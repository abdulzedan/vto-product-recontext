"""Gemini-based image analyzer for classification and feedback."""

import asyncio
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
    garment_applied: bool = True  # Whether the garment was actually applied to the model
    length_accurate: bool = True  # Whether the garment length matches the original
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    length_issue: str = "none"  # Specific description of any length problems
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
    async def classify_image(
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
            
            # Generate response using native async
            response = await self.client.generate_content_async(
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
        - "apparel": ONLY clothing items that cover the body (shirts, pants, dresses, skirts, jackets, coats, sweaters, etc.) and shoes
        - "product": Accessories (handbags, belts, bracelets, earrings, necklaces, watches, wallets), furniture, electronics, home goods, food, tools, and all other non-clothing items
        
        IMPORTANT GUIDELINES:
        - Focus on the main subject of the image
        - If multiple items are present, classify based on the primary/dominant item
        - Handbags, belts, bracelets, earrings, and all jewelry/accessories should be classified as "product" (NOT apparel)
        - Only items that are worn as primary clothing should be "apparel"
        - Consider the context and typical use of the item
        
        GENDER DETECTION FOR APPAREL:
        If the item is classified as "apparel", also determine the target gender:
        - "woman": Women's clothing, feminine styles, dresses, skirts, blouses, women's shoes
        - "man": Men's clothing, masculine styles, suits, ties, men's shirts, men's shoes
        - "unisex": Items that can be worn by any gender (some t-shirts, jeans, sneakers, etc.)
        
        Please respond with a JSON object containing:
        {
            "category": "apparel" or "product",
            "confidence": float between 0.0 and 1.0,
            "reasoning": "detailed explanation of your classification decision",
            "detected_items": ["list", "of", "items", "detected", "in", "image"],
            "target_gender": "woman", "man", "unisex", or null (only for apparel items),
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
    async def analyze_virtual_try_on_quality(
        self,
        result_image: Union[Image.Image, Path, str],
        original_apparel: Union[Image.Image, Path, str],
        model_image: Union[Image.Image, Path, str],
        apparel_description: str = None,
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
            
            # First, get a detailed description of the apparel if not provided
            if not apparel_description:
                apparel_description = await self._get_apparel_description(original_apparel)
            
            # Create feedback prompt with apparel context
            prompt = self._create_vto_feedback_prompt(apparel_description)
            
            # Generate response using native async
            response = await self.client.generate_content_async(
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
    
    async def _get_apparel_description(self, apparel_image: Image.Image) -> str:
        """Get detailed description of apparel for comparison."""
        try:
            prompt = """
            Provide a detailed description of this apparel item focusing on:
            1. Type of garment (e.g., trousers, shirt, dress)
            2. Color (be specific - e.g., "black with white pinstripes" not just "black")
            3. Pattern or texture (stripes, checks, plain, etc.)
            4. Notable features (buttons, pockets, etc.)
            
            Be very specific about patterns - distinguish between:
            - Plain/solid colors
            - Pinstripes
            - Wide stripes
            - Checks
            - Other patterns
            
            Format: Return a single descriptive sentence.
            """
            
            response = await self.client.generate_content_async(
                [prompt, apparel_image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )
            
            description = response.text.strip()
            logger.info("Apparel description generated", description=description)
            return description
            
        except Exception as e:
            logger.warning("Failed to generate apparel description", error=str(e))
            return "apparel item"
    
    def _create_vto_feedback_prompt(self, apparel_description: str) -> str:
        """Create prompt for Virtual Try-On feedback."""
        return f"""
        Analyze the quality of this Virtual Try-On result. You will see three images:
        1. The result image (person wearing the apparel)
        2. The original apparel item: {apparel_description}
        3. The model/person image (original, before VTO)
        
        CRITICAL FIRST CHECK:
        - The apparel to be applied is: {apparel_description}
        - Check if THIS SPECIFIC apparel is visible on the person in image 1
        - Compare carefully with the model's original outfit in image 3
        - Look for specific patterns, textures, and details that distinguish the new garment
        
        IMPORTANT: The model may originally be wearing a similar colored item. You must check for:
        - Pattern changes (e.g., plain to pinstripe, solid to checkered)
        - Texture changes
        - Style changes (e.g., different cut or fit)
        - Any visual difference that indicates the new garment was applied
        
        CRITICAL LENGTH CHECK (MOST IMPORTANT):
        - Compare the LENGTH of the garment in image 2 (original) with the result in image 1
        - For shirts/tops: Check if it's cropped, regular, or long - must match original
        - For pants: Check if they're shorts, cropped, regular, or long - must match original
        - For dresses/skirts: Check if length is mini, knee, midi, or maxi - must match original
        - For jackets: Check if it's cropped, regular, or long coat - must match original
        
        Set "length_accurate": false if:
        - A long garment appears shortened (e.g., pants become shorts)
        - A short garment appears lengthened (e.g., crop top becomes regular shirt)
        - The hem line is significantly different from the original
        - The proportions are wrong (e.g., midi skirt becomes mini)
        
        Set "garment_applied": false ONLY if:
        - The result image is EXACTLY identical to the original model image
        - No visual changes whatsoever can be detected
        - The specific features of "{apparel_description}" are NOT present
        
        QUALITY CRITERIA (only evaluate if garment was applied):
        - LENGTH ACCURACY: Is the garment length true to the original? (CRITICAL)
        - Realistic placement: Is the apparel positioned naturally on the person?
        - Proper fit: Does the apparel fit the person's body appropriately?
        - Color accuracy: Are the colors AND PATTERNS consistent with the original apparel?
        - Pattern accuracy: Are pinstripes, checks, or other patterns correctly applied?
        - Visual coherence: Does the result look realistic and believable?
        - Lighting consistency: Does the lighting match between the person and apparel?
        - Edge quality: Are the edges of the apparel clean and well-integrated?
        
        SCORING GUIDELINES:
        - Score 0.0: Garment not applied or completely failed
        - Score 0.2-0.4: Garment applied but poor quality, wrong length, or significant problems
        - Score 0.4-0.6: Acceptable quality but noticeable issues (including minor length issues)
        - Score 0.6-0.8: Good quality with correct length, minor issues only
        - Score 0.8-1.0: Excellent quality, very realistic, perfect length match
        
        Pass threshold: 0.6 or higher AND garment must be applied AND length must be accurate
        
        Please respond with a JSON object containing:
        {{
            "garment_applied": true or false,
            "length_accurate": true or false,
            "passed": true or false,
            "score": float between 0.0 and 1.0,
            "reasoning": "detailed explanation of the quality assessment",
            "length_issue": "specific description of any length problems, or 'none' if accurate",
            "suggestions": ["list", "of", "improvement", "suggestions"],
            "metadata": {{
                "fit_quality": "assessment of how well the apparel fits",
                "length_accuracy": "detailed assessment of garment length matching",
                "color_accuracy": "assessment of color AND pattern matching",
                "realism": "assessment of overall realism",
                "main_issues": ["list", "of", "main", "problems"],
                "detected_changes": "describe what changed from original model to result"
            }}
        }}
        """
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def analyze_product_recontext_quality(
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
            
            # Generate response using native async
            response = await self.client.generate_content_async(
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
            garment_applied = bool(response_data.get("garment_applied", True))
            length_accurate = bool(response_data.get("length_accurate", True))
            passed = bool(response_data.get("passed", False))
            score = min(max(float(response_data.get("score", 0.0)), 0.0), 1.0)
            
            return FeedbackResult(
                garment_applied=garment_applied,
                length_accurate=length_accurate,
                passed=passed,
                score=score,
                reasoning=response_data.get("reasoning", "No reasoning provided"),
                length_issue=response_data.get("length_issue", "none"),
                suggestions=response_data.get("suggestions", []),
                metadata=response_data.get("metadata", {}),
            )
            
        except Exception as e:
            logger.warning("Failed to parse feedback response", error=str(e))
            return FeedbackResult(
                garment_applied=False,
                length_accurate=False,
                passed=False,
                score=0.0,
                reasoning=f"Failed to parse response: {str(e)}",
                length_issue="parsing_error",
                suggestions=[],
                metadata={},
            )
    
    async def generate_product_recontext_prompt(
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
            
            # Generate response using native async
            response = await self.client.generate_content_async(
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
        Analyze this product image and identify what type of accessory it is, then generate an appropriate scene prompt based on these guidelines:
        
        FOR HANDBAGS:
        - Café table: Bag resting on a small café table with a coffee cup and book, shallow depth of field, soft morning light
        - Entryway console: Bag placed on a console with keys and flowers, natural daylight coming from the side
        - On shoulder: Bag worn over the shoulder, cropped at torso only (no face), neutral clothing to keep focus on the product
        
        FOR BELTS:
        - Coiled on linen: Belt coiled in a clean circle on a textured linen surface, soft daylight highlighting the buckle
        - On folded clothing: Belt laid across neatly folded trousers or denim on a wooden shelf, buckle facing camera
        - Worn on waist: Belt styled on simple clothing, cropped midsection only, showing natural fit around the waist
        
        FOR BRACELETS:
        - Vanity dish: Bracelet resting in a ceramic dish on a vanity, blurred perfume bottle in background, daylight reflection
        - Velvet roll: Multiple bracelets stacked on a velvet jewelry roll, angled light to catch shine
        - On wrist: Bracelet worn on a wrist, cropped to hand/forearm only, natural pose with soft light
        
        FOR EARRINGS:
        - Tile surface: Earrings laid out on a neutral ceramic tile, soft daylight, gentle shadow detail
        - Jewelry card: Earrings pinned on a minimal jewelry card, leaning against a mirror base for reflection
        - On ear: Earrings worn in-ear, cropped close around ear/jawline (no face), soft even light
        
        GENERAL REQUIREMENTS:
        - Keep the product as the focal point
        - Use natural, soft lighting that enhances the product
        - Maintain clean, uncluttered compositions
        - Ensure professional photography style
        - Keep prompts concise (2-3 sentences maximum)
        
        Generate ONLY the prompt text for the appropriate accessory type, no additional explanation.
        """
        
        if product_description:
            base_prompt += f"\n\nProduct context: {product_description}"
        
        return base_prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def recommend_fashion_coordination(
        self,
        apparel_image: Union[Image.Image, Path, str],
        apparel_info: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        exclude_models: List[str] = None,
    ) -> Dict[str, Any]:
        """Recommend the best model for fashion coordination with the given apparel."""
        start_time = time.time()
        
        try:
            # Prepare apparel image
            if isinstance(apparel_image, (str, Path)):
                apparel_image = Image.open(apparel_image)
            
            # Filter available models by gender and exclusions
            target_gender = apparel_info.get('target_gender')
            exclude_models = exclude_models or []
            
            suitable_models = []
            for model in available_models:
                if model['id'] in exclude_models:
                    continue
                if target_gender and model['gender'] != target_gender:
                    continue
                suitable_models.append(model)
            
            if not suitable_models:
                raise ValueError(f"No suitable models available for gender '{target_gender}' after exclusions")
            
            # Create fashion coordination prompt
            prompt = self._create_fashion_coordination_prompt(apparel_info, suitable_models)
            
            # Generate response using native async
            response = await self.client.generate_content_async(
                [prompt, apparel_image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lower temperature for more consistent recommendations
                    max_output_tokens=1024,
                ),
            )
            
            # Parse response
            result = self._parse_fashion_coordination_response(response.text, suitable_models)
            
            coordination_time = time.time() - start_time
            
            logger.info(
                "Fashion coordination completed",
                recommended_model=result['recommended_model_id'],
                coordination_score=result.get('coordination_score', 0),
                analysis_time=round(coordination_time, 2),
            )
            
            return result
            
        except Exception as e:
            coordination_time = time.time() - start_time
            logger.error(
                "Fashion coordination failed",
                error=str(e),
                analysis_time=round(coordination_time, 2),
            )
            raise
    
    def _create_fashion_coordination_prompt(self, apparel_info: Dict[str, Any], models: List[Dict[str, Any]]) -> str:
        """Create prompt for fashion coordination recommendation."""
        # Build model options description
        model_descriptions = []
        for i, model in enumerate(models, 1):
            outfit = model.get('outfit', {})
            description = f"{i}. {model['id']}: {outfit.get('description', 'No description')} (File: {model['filename']})"
            model_descriptions.append(description)
        
        models_text = '\n'.join(model_descriptions)
        
        return f"""
        Analyze this apparel item and recommend the best model from the available options for optimal fashion coordination.
        
        APPAREL INFORMATION:
        - Detected items: {apparel_info.get('detected_items', [])}
        - Target gender: {apparel_info.get('target_gender', 'unknown')}
        - Confidence: {apparel_info.get('confidence', 0)}
        - Style metadata: {apparel_info.get('metadata', {})}
        
        AVAILABLE MODELS:
        {models_text}
        
        FASHION COORDINATION CRITERIA (in order of priority):
        1. **Garment Compatibility**: MOST IMPORTANT - Choose models wearing compatible base garments
           - Blazers/jackets/coats → models wearing PANTS/TROUSERS (not skirts or dresses)
           - Pants/trousers → models wearing TOPS (shirts, sweaters, not dresses)
           - Shirts/tops → models wearing BOTTOMS (pants, skirts)
           - Dresses → models wearing SIMPLE SEPARATES (avoid competing with the dress)
           - Skirts → models wearing COMPATIBLE TOPS
           
           EXAMPLES:
           ✅ Black blazer + model wearing trousers = GOOD (can layer blazer over top)
           ❌ Black blazer + model wearing dress/skirt = BAD (blazer doesn't work with dresses)
           ✅ Blue trousers + model wearing shirt = GOOD (trousers replace model's bottoms)
           ❌ Blue trousers + model wearing dress = BAD (can't replace dress with just trousers)
        
        2. **Style Coherence**: Match the formality and style level
           - Formal apparel (suits, blazers, dress shirts) → models in formal wear
           - Casual apparel (t-shirts, jeans, sneakers) → models in casual wear
           - Smart casual apparel → models in smart casual wear
        
        3. **Color Harmony**: Choose models wearing colors that complement the apparel
           - Complementary colors (opposite on color wheel)
           - Analogous colors (adjacent on color wheel)  
           - Neutral colors that work with any color
           - Avoid clashing color combinations
        
        4. **Pattern Compatibility**: Avoid pattern conflicts
           - If apparel has patterns, choose models in solid colors
           - If apparel is solid, models with subtle patterns are acceptable
        
        5. **Aesthetic Appeal**: Choose the combination that looks fashionable
           - Professional styling principles
           - Visual balance and proportion
        
        SCORING GUIDELINES:
        - Score 0.9-1.0: Perfect garment compatibility + excellent color/style coordination
        - Score 0.7-0.9: Good garment compatibility + good color coordination
        - Score 0.5-0.7: Acceptable compatibility but color/style issues
        - Score 0.3-0.5: Poor garment compatibility or major color clashing
        - Score 0.0-0.3: Incompatible garments (e.g., blazer with dress model)
        
        Please respond with a JSON object containing:
        {{
            "recommended_model_id": "exact model ID from the list",
            "coordination_score": float between 0.0 and 1.0,
            "reasoning": "detailed explanation of why this model was chosen",
            "color_analysis": "analysis of color coordination",
            "style_analysis": "analysis of style compatibility", 
            "alternative_models": ["list", "of", "second", "best", "options"],
            "fashion_tips": "tips for why this combination works well"
        }}
        """
    
    def _parse_fashion_coordination_response(self, response_text: str, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse fashion coordination response from Gemini."""
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
            
            # Validate recommended model ID exists
            recommended_id = response_data.get("recommended_model_id", "")
            valid_ids = [model['id'] for model in models]
            
            if recommended_id not in valid_ids:
                logger.warning(
                    "Recommended model ID not found in available models, using first available",
                    recommended_id=recommended_id,
                    valid_ids=valid_ids,
                )
                recommended_id = valid_ids[0] if valid_ids else "unknown"
            
            return {
                'recommended_model_id': recommended_id,
                'coordination_score': min(max(float(response_data.get('coordination_score', 0.5)), 0.0), 1.0),
                'reasoning': response_data.get('reasoning', 'No reasoning provided'),
                'color_analysis': response_data.get('color_analysis', ''),
                'style_analysis': response_data.get('style_analysis', ''),
                'alternative_models': response_data.get('alternative_models', []),
                'fashion_tips': response_data.get('fashion_tips', ''),
            }
            
        except Exception as e:
            logger.warning("Failed to parse fashion coordination response", error=str(e))
            # Fallback to first available model
            return {
                'recommended_model_id': models[0]['id'] if models else 'model_1_woman',
                'coordination_score': 0.5,
                'reasoning': f'Failed to parse response: {str(e)}',
                'color_analysis': '',
                'style_analysis': '',
                'alternative_models': [],
                'fashion_tips': '',
            }


def create_analyzer(settings: Settings) -> GeminiAnalyzer:
    """Create a Gemini analyzer instance."""
    return GeminiAnalyzer(settings)