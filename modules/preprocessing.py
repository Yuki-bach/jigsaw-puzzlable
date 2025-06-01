import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


def load_pieces(directory_path: str) -> Dict[str, np.ndarray]:
    """Load all piece images from the specified directory."""
    import os
    pieces = {}
    
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.jpg'):
            filepath = os.path.join(directory_path, filename)
            image = cv2.imread(filepath)
            if image is not None:
                pieces[filename] = image
    
    return pieces


def preprocess_piece(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a single piece image to extract its contour.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        contour: The main contour of the piece
        binary_image: Binary image of the piece
    """
    # Resize image if too large (to improve processing speed)
    height, width = image.shape[:2]
    max_dimension = 1000
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Check if we have a light or dark background
    mean_val = np.mean(blurred)
    
    if mean_val > 127:  # Light background, dark pieces
        # Use inverted Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # Dark background, light pieces
        # Use regular Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Remove small objects
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # Find the largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.uint8(labels == largest_label) * 255
    
    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    
    # Find the largest contour (assumed to be the puzzle piece)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours (noise) - adjusted for resized images
        min_area = (max_dimension * max_dimension) * 0.01  # At least 1% of image area
        if cv2.contourArea(contour) < min_area:
            return None, binary
            
        return contour, binary
    
    return None, binary


def preprocess_all_pieces(pieces: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """
    Preprocess all pieces and extract their contours.
    
    Args:
        pieces: Dictionary of piece images
        
    Returns:
        Dictionary with processed piece information
    """
    processed_pieces = {}
    
    for piece_name, image in pieces.items():
        # Keep a copy of the original for preprocessing
        original_for_processing = image.copy()
        
        # Resize original for storage if too large
        height, width = image.shape[:2]
        max_dimension = 1000
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_original = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized_original = image
        
        contour, binary = preprocess_piece(original_for_processing)
        
        if contour is not None:
            processed_pieces[piece_name] = {
                'original': resized_original,
                'binary': binary,
                'contour': contour,
                'area': cv2.contourArea(contour),
                'perimeter': cv2.arcLength(contour, True)
            }
    
    return processed_pieces