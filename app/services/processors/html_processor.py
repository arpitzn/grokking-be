"""HTML processor for HTML files using BeautifulSoup"""
import html
import re
import logging
from typing import List
from bs4 import BeautifulSoup
from app.services.processors.base import BaseProcessor, ProcessedChunk, ProcessedContent
from app.services.chunking import chunk_text_custom

logger = logging.getLogger(__name__)


class HTMLProcessor(BaseProcessor):
    """Processor for HTML files using BeautifulSoup"""
    
    def __init__(self):
        self.supported_mime_types = [
            "text/html",
            "application/xhtml+xml",
        ]
        self.supported_extensions = [".html", ".htm"]
    
    def can_process(self, mime_type: str) -> bool:
        """Check if this processor can handle the MIME type"""
        return mime_type in self.supported_mime_types
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file extensions"""
        return self.supported_extensions
    
    async def process(self, file_content: bytes, filename: str) -> ProcessedContent:
        """
        Process HTML file and extract text
        
        Steps:
        1. Decode bytes to UTF-8 string
        2. Parse HTML with BeautifulSoup
        3. Extract text from <main>, <article>, or content divs
        4. Clean HTML entities and normalize whitespace
        5. Apply custom chunking
        """
        try:
            # Decode bytes to UTF-8 string
            html_content = file_content.decode("utf-8")
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Try to find main content areas (priority order)
            text_content = None
            
            # Priority 1: <main> tag
            main_tag = soup.find("main")
            if main_tag:
                text_content = main_tag.get_text(separator=" ", strip=True)
            
            # Priority 2: <article> tag
            if not text_content:
                article_tag = soup.find("article")
                if article_tag:
                    text_content = article_tag.get_text(separator=" ", strip=True)
            
            # Priority 3: Content divs (common class names)
            if not text_content:
                content_selectors = [
                    soup.find("div", class_=re.compile(r"content|main|article|post", re.I)),
                    soup.find("div", id=re.compile(r"content|main|article|post", re.I)),
                ]
                for div in content_selectors:
                    if div:
                        text_content = div.get_text(separator=" ", strip=True)
                        break
            
            # Fallback: Extract all text from body
            if not text_content:
                body_tag = soup.find("body")
                if body_tag:
                    text_content = body_tag.get_text(separator=" ", strip=True)
                else:
                    # Last resort: entire document
                    text_content = soup.get_text(separator=" ", strip=True)
            
            # Clean HTML entities and normalize whitespace
            text_content = html.unescape(text_content)  # Decode HTML entities
            text_content = re.sub(r"\s+", " ", text_content)  # Normalize whitespace
            text_content = text_content.strip()
            
            if not text_content:
                logger.warning(f"No text content extracted from {filename}")
                text_content = "[HTML: No text content found]"
            
            # Apply custom chunking (2500 chars, 100 overlap)
            chunk_texts = chunk_text_custom(text_content, max_chars=2500, overlap=100)
            
            # Create ProcessedChunk objects
            chunks = []
            for idx, chunk_text in enumerate(chunk_texts):
                chunks.append(ProcessedChunk(
                    content=chunk_text,
                    chunk_type="text",
                    chunk_index=idx,
                    metadata={}
                ))
            
            return ProcessedContent(
                text=text_content,
                chunks=chunks,
                metadata={"filename": filename, "method": "beautifulsoup"},
                structure={
                    "chunk_count": len(chunks),
                    "method": "beautifulsoup_extraction",
                    "chunking_method": "custom_chunking",
                    "max_characters": 2500,
                    "overlap": 100
                }
            )
        
        except UnicodeDecodeError as e:
            logger.error(f"Error decoding HTML file {filename}: {e}")
            raise ValueError(f"Failed to decode file as UTF-8: {e}")
        except Exception as e:
            logger.error(f"Error processing HTML file {filename}: {e}")
            raise
