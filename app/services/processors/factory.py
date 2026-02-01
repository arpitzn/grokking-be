"""Processor factory for routing files to correct processor based on MIME type"""
import mimetypes
import os
import logging
from typing import Optional
from app.services.processors.base import BaseProcessor
from app.services.processors.document_processor import DocumentProcessor
from app.services.processors.image_processor import ImageProcessor
from app.services.processors.html_processor import HTMLProcessor

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """Factory for routing files to correct processor"""
    
    def __init__(self):
        # Initialize all processors
        self.processors: list[BaseProcessor] = [
            DocumentProcessor(),
            ImageProcessor(),
            HTMLProcessor(),
        ]
    
    def get_processor(self, mime_type: str, filename: str = "") -> Optional[BaseProcessor]:
        """
        Get processor for given MIME type
        
        Args:
            mime_type: MIME type string (e.g., "application/pdf")
            filename: Optional filename for fallback MIME type detection
            
        Returns:
            Processor instance or None if no processor found
        """
        # Try exact MIME type match
        for processor in self.processors:
            if processor.can_process(mime_type):
                return processor
        
        # Fallback: Try to guess MIME type from filename
        if filename:
            guessed_type, _ = mimetypes.guess_type(filename)
            if guessed_type:
                logger.info(f"Guessed MIME type {guessed_type} from filename {filename}")
                for processor in self.processors:
                    if processor.can_process(guessed_type):
                        return processor
        
        # Final fallback: If still no processor and MIME type is generic, try extension
        if mime_type == "application/octet-stream" and filename:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext:
                processor = self.get_processor_by_extension(file_ext)
                if processor:
                    logger.info(f"Found processor by extension {file_ext} for filename {filename}")
                    return processor
        
        logger.warning(f"No processor found for MIME type: {mime_type}, filename: {filename}")
        return None
    
    def get_processor_by_extension(self, extension: str) -> Optional[BaseProcessor]:
        """
        Get processor by file extension
        
        Args:
            extension: File extension (e.g., ".pdf")
            
        Returns:
            Processor instance or None if no processor found
        """
        # Map extension to MIME type
        mime_type, _ = mimetypes.guess_type(f"file{extension}")
        if mime_type:
            return self.get_processor(mime_type)
        
        # Fallback: Check processors directly
        for processor in self.processors:
            if extension.lower() in processor.get_supported_formats():
                return processor
        
        return None


# Global factory instance
_factory: Optional[ProcessorFactory] = None


def get_processor(mime_type: str, filename: str = "") -> Optional[BaseProcessor]:
    """
    Convenience function to get processor
    
    Args:
        mime_type: MIME type string
        filename: Optional filename for fallback
        
    Returns:
        Processor instance or None
    """
    global _factory
    if _factory is None:
        _factory = ProcessorFactory()
    return _factory.get_processor(mime_type, filename)
