"""Image processor for PNG, JPG, JPEG files using OpenAI Vision API OCR"""
import base64
import logging
from typing import List
from PIL import Image
from io import BytesIO
from app.services.processors.base import BaseProcessor, ProcessedChunk, ProcessedContent
from app.infra.config import settings

logger = logging.getLogger(__name__)


class ImageProcessor(BaseProcessor):
    """Processor for image files (PNG, JPG, JPEG) using OpenAI Vision API"""
    
    def __init__(self):
        self.supported_mime_types = [
            "image/png",
            "image/jpeg",
            "image/jpg",
        ]
        self.supported_extensions = [".png", ".jpg", ".jpeg"]
    
    def can_process(self, mime_type: str) -> bool:
        """Check if this processor can handle the MIME type"""
        return mime_type in self.supported_mime_types
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file extensions"""
        return self.supported_extensions
    
    async def process(self, file_content: bytes, filename: str) -> ProcessedContent:
        """
        Process image file and extract text using OpenAI Vision API OCR
        
        Steps:
        1. Validate image format with Pillow
        2. Base64 encode image
        3. Call OpenAI Vision API with prompt: "Extract all text from this image"
        4. Return single chunk with extracted text
        """
        # Validate image format
        try:
            image = Image.open(BytesIO(file_content))
            image.verify()  # Verify it's a valid image
        except Exception as e:
            logger.error(f"Invalid image format for {filename}: {e}")
            raise ValueError(f"Invalid image format: {e}")
        
        # Re-open image after verification (verify() closes it)
        image = Image.open(BytesIO(file_content))
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Call OpenAI Vision API
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Vision-capable model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the extracted text, no additional commentary."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            # If Vision API fails or returns empty, return placeholder
            if not extracted_text:
                logger.warning(f"Vision API returned empty text for {filename}")
                extracted_text = "[Image: Text extraction failed or no text found]"
            
            # Create single chunk (images return one chunk per Markdown spec)
            chunk = ProcessedChunk(
                content=extracted_text,
                chunk_type="image",
                chunk_index=0,
                metadata={}
            )
            
            return ProcessedContent(
                text=extracted_text,
                chunks=[chunk],
                metadata={"filename": filename, "method": "openai_vision_ocr"},
                structure={
                    "chunk_count": 1,
                    "method": "openai_vision_api",
                    "chunk_type": "image"
                }
            )
        
        except ImportError:
            logger.error("OpenAI library not installed or API key not configured")
            raise
        except Exception as e:
            logger.error(f"Error calling OpenAI Vision API for {filename}: {e}")
            # Fallback: return placeholder text
            fallback_text = f"[Image: OCR failed - {str(e)}]"
            return ProcessedContent(
                text=fallback_text,
                chunks=[ProcessedChunk(
                    content=fallback_text,
                    chunk_type="image",
                    chunk_index=0,
                    metadata={}
                )],
                metadata={"filename": filename, "method": "fallback"},
                structure={
                    "chunk_count": 1,
                    "method": "fallback",
                    "error": str(e)
                }
            )
