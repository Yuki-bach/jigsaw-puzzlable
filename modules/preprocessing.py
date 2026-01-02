"""
Preprocessing module for puzzle piece image processing.
Handles image loading and contour extraction.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


def load_pieces(directory: str, max_size: int = 800) -> Dict[str, np.ndarray]:
    """
    Load all puzzle piece images from a directory.

    Args:
        directory: Path to the directory containing piece images
        max_size: Maximum dimension (width or height) for resizing

    Returns:
        Dictionary mapping piece names to image arrays
    """
    pieces = {}
    dir_path = Path(directory)

    # Find all JPG files
    image_files = sorted(dir_path.glob("*.jpg")) + sorted(dir_path.glob("*.JPG"))

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is not None:
            # Resize to reduce memory usage
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            piece_name = image_path.stem  # e.g., "piece_001"
            pieces[piece_name] = image

    return pieces


def preprocess_piece(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Preprocess a single puzzle piece image to extract its contour.

    Uses a simple approach: grayscale -> Otsu's threshold -> morphology -> contour detection.

    Args:
        image: BGR image of a puzzle piece

    Returns:
        Tuple of (contour, binary_image, metadata)
        Returns (None, None, metadata) if processing fails
    """
    metadata = {
        'success': False,
        'method': 'otsu',
        'area': 0,
        'perimeter': 0
    }

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None, None, metadata

        # Get the largest contour (should be the puzzle piece)
        contour = max(contours, key=cv2.contourArea)

        # Filter by minimum area (avoid noise)
        area = cv2.contourArea(contour)
        if area < 10000:  # Minimum area threshold
            return None, None, metadata

        # Calculate perimeter
        perimeter = cv2.arcLength(contour, closed=True)

        metadata['success'] = True
        metadata['area'] = area
        metadata['perimeter'] = perimeter

        return contour, binary, metadata

    except Exception as e:
        metadata['error'] = str(e)
        return None, None, metadata


def preprocess_piece_adaptive(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Alternative preprocessing using adaptive thresholding.
    Use this if Otsu's method fails.

    Args:
        image: BGR image of a puzzle piece

    Returns:
        Tuple of (contour, binary_image, metadata)
    """
    metadata = {
        'success': False,
        'method': 'adaptive',
        'area': 0,
        'perimeter': 0
    }

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=-5
        )

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None, None, metadata

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < 10000:
            return None, None, metadata

        perimeter = cv2.arcLength(contour, closed=True)

        metadata['success'] = True
        metadata['area'] = area
        metadata['perimeter'] = perimeter

        return contour, binary, metadata

    except Exception as e:
        metadata['error'] = str(e)
        return None, None, metadata


def preprocess_all_pieces(pieces: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """
    Preprocess all puzzle pieces.

    Args:
        pieces: Dictionary mapping piece names to images

    Returns:
        Dictionary mapping piece names to processed data
    """
    processed = {}

    for piece_name, image in pieces.items():
        # Try Otsu's method first
        contour, binary, metadata = preprocess_piece(image)

        # Fallback to adaptive if Otsu fails
        if contour is None:
            contour, binary, metadata = preprocess_piece_adaptive(image)

        if contour is not None:
            processed[piece_name] = {
                'contour': contour,
                'area': metadata['area'],
                'perimeter': metadata['perimeter'],
                'method': metadata['method']
            }
            # Note: binary and original_image are not stored to save memory

    return processed


def smooth_contour(contour: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth a contour using a moving average filter.

    Args:
        contour: Input contour points
        window_size: Size of the smoothing window

    Returns:
        Smoothed contour
    """
    if len(contour) < window_size:
        return contour

    contour_2d = contour.reshape(-1, 2)
    n = len(contour_2d)

    # Pad the contour for circular smoothing
    half = window_size // 2
    padded = np.vstack([contour_2d[-half:], contour_2d, contour_2d[:half]])

    # Apply moving average
    smoothed = np.zeros_like(contour_2d, dtype=np.float64)
    for i in range(n):
        smoothed[i] = np.mean(padded[i:i + window_size], axis=0)

    return smoothed.astype(np.int32).reshape(-1, 1, 2)
