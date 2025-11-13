"""
Utility functions for Judicaita.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        sanitized = name[: 255 - len(ext)] + ext

    return sanitized


def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.

    Args:
        dt: Datetime object
        format_str: Format string

    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)


def parse_timestamp(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse timestamp string to datetime.

    Args:
        timestamp_str: Timestamp string
        format_str: Format string

    Returns:
        Datetime object
    """
    return datetime.strptime(timestamp_str, format_str)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def extract_metadata_from_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested dictionary for metadata extraction.

    Args:
        data: Dictionary to flatten
        prefix: Prefix for keys

    Returns:
        Flattened dictionary
    """
    result: dict[str, Any] = {}

    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(extract_metadata_from_dict(value, new_key))
        elif isinstance(value, (list, tuple)):
            result[new_key] = str(value)
        else:
            result[new_key] = value

    return result
