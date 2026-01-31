"""Base processor interface for file type extraction"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ProcessedChunk:
    """Represents a single chunk of processed content"""
    content: str
    chunk_type: str  # "text" or "image"
    chunk_index: int
    metadata: Dict[str, Any]


@dataclass
class ProcessedContent:
    """Represents processed content from a file"""
    text: str  # Full extracted text (concatenated)
    chunks: List[ProcessedChunk]
    metadata: Dict[str, Any]
    structure: Dict[str, Any]  # chunk_count, method, etc.


class BaseProcessor(ABC):
    """Base interface for all file processors"""
    
    @abstractmethod
    async def process(self, file_content: bytes, filename: str) -> ProcessedContent:
        """
        Process file content and extract text chunks
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            ProcessedContent with extracted text and chunks
        """
        pass
    
    @abstractmethod
    def can_process(self, mime_type: str) -> bool:
        """
        Check if this processor can handle the given MIME type
        
        Args:
            mime_type: MIME type string (e.g., "application/pdf")
            
        Returns:
            True if processor can handle this MIME type
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file extensions
        
        Returns:
            List of extensions (e.g., [".pdf", ".docx"])
        """
        pass
