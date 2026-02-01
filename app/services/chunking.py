"""Custom chunking algorithm matching Markdown specification"""
from typing import List


def find_break_point(text: str, end_pos: int, start_pos: int) -> int:
    """
    Find the best break point before end_pos, looking backward.
    
    Priority: sentence (`. `) > newline (`\n`) > word (` `)
    
    Args:
        text: Full text string
        end_pos: Desired end position
        start_pos: Start position (don't go before this)
        
    Returns:
        Position to break at
    """
    # Look backward from end_pos, but not before start_pos
    search_start = max(start_pos, end_pos - 200)  # Look back up to 200 chars
    
    # Priority 1: Look for sentence boundary (`. ` or `.\n`)
    for i in range(end_pos - 1, search_start - 1, -1):
        if i < len(text) - 1:
            if text[i] == '.' and (text[i + 1] == ' ' or text[i + 1] == '\n'):
                return i + 2  # Include the period and space/newline
    
    # Priority 2: Look for newline
    for i in range(end_pos - 1, search_start - 1, -1):
        if text[i] == '\n':
            return i + 1  # Include the newline
    
    # Priority 3: Look for word boundary (space)
    for i in range(end_pos - 1, search_start - 1, -1):
        if text[i] == ' ':
            return i + 1  # Include the space
    
    # Fallback: Break at exact position (may split word)
    return end_pos


def chunk_text_custom(text: str, max_chars: int = 2500, overlap: int = 100) -> List[str]:
    """
    Chunk text using word-boundary algorithm with overlap.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        if end >= len(text):
            # Last chunk - take remaining text
            chunks.append(text[start:])
            break
        
        # Find break point (sentence > newline > word)
        break_pos = find_break_point(text, end, start)
        chunk = text[start:break_pos]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, break_pos - overlap)
    
    return chunks
