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
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    
    # Find the largest contour (assumed to be the puzzle piece)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours (noise)
        if cv2.contourArea(contour) < 1000:
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
        contour, binary = preprocess_piece(image)
        
        if contour is not None:
            processed_pieces[piece_name] = {
                'original': image,
                'binary': binary,
                'contour': contour,
                'area': cv2.contourArea(contour),
                'perimeter': cv2.arcLength(contour, True)
            }
    
    return processed_pieces