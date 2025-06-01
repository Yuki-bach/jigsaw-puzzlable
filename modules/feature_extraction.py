import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import math


def find_corners_simple(contour: np.ndarray, num_corners: int = 4) -> List[Tuple[int, int]]:
    """
    Find the corners of a puzzle piece using a simpler bounding box approach.
    
    Args:
        contour: The contour of the puzzle piece
        num_corners: Expected number of corners (4 for standard puzzle pieces)
        
    Returns:
        List of corner points as (x, y) tuples
    """
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Convert to list of tuples
    corners = [tuple(point) for point in box]
    
    # Sort corners to ensure consistent ordering (top-left, top-right, bottom-right, bottom-left)
    corners = sorted(corners, key=lambda p: (p[1], p[0]))  # Sort by y, then x
    
    # Rearrange to get: top-left, top-right, bottom-right, bottom-left
    if len(corners) == 4:
        top_two = sorted(corners[:2], key=lambda p: p[0])  # Top row sorted by x
        bottom_two = sorted(corners[2:], key=lambda p: p[0], reverse=True)  # Bottom row sorted by x desc
        corners = top_two + bottom_two
    
    return corners


def extract_edge(contour: np.ndarray, start_point: Tuple[int, int], 
                 end_point: Tuple[int, int]) -> np.ndarray:
    """
    Extract edge points between two corner points.
    
    Args:
        contour: The full contour
        start_point: Starting corner point
        end_point: Ending corner point
        
    Returns:
        Array of points along the edge
    """
    contour_points = contour.reshape(-1, 2)
    
    # Find indices of closest points to start and end
    start_distances = np.sum((contour_points - np.array(start_point))**2, axis=1)
    end_distances = np.sum((contour_points - np.array(end_point))**2, axis=1)
    
    start_idx = np.argmin(start_distances)
    end_idx = np.argmin(end_distances)
    
    # Extract edge points going clockwise
    if end_idx > start_idx:
        edge_points = contour_points[start_idx:end_idx+1]
    else:
        # Wrap around the contour
        edge_points = np.concatenate([
            contour_points[start_idx:],
            contour_points[:end_idx+1]
        ])
    
    # If we got more than half the contour, we went the wrong way
    if len(edge_points) > len(contour_points) // 2:
        # Go the other way
        if start_idx > end_idx:
            edge_points = contour_points[end_idx:start_idx+1][::-1]
        else:
            edge_points = np.concatenate([
                contour_points[end_idx:],
                contour_points[:start_idx+1]
            ])[::-1]
    
    return edge_points


def classify_edge_type(edge_points: np.ndarray) -> str:
    """
    Classify whether an edge is convex, concave, or flat.
    Simplified version for MVP.
    
    Args:
        edge_points: Points along the edge
        
    Returns:
        'convex', 'concave', or 'flat'
    """
    if len(edge_points) < 10:
        return 'flat'
    
    # Fit a line through start and end points
    start = edge_points[0]
    end = edge_points[-1]
    
    # Find the middle point
    mid_idx = len(edge_points) // 2
    mid_point = edge_points[mid_idx]
    
    # Calculate distance from middle point to the line
    line_vec = end - start
    point_vec = mid_point - start
    
    # Cross product gives us which side of the line the point is on
    cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    
    # Distance from point to line
    dist = np.abs(cross) / np.linalg.norm(line_vec)
    
    if dist < 10:  # Threshold for flat edges (adjusted for resized images)
        return 'flat'
    elif cross > 0:
        return 'convex'
    else:
        return 'concave'


def create_edge_descriptor(edge_points: np.ndarray) -> np.ndarray:
    """
    Create a normalized descriptor for edge matching.
    
    Args:
        edge_points: Points along the edge
        
    Returns:
        Normalized edge descriptor
    """
    if len(edge_points) < 10:
        return np.array([])
    
    # Resample edge to fixed number of points
    num_samples = 50
    indices = np.linspace(0, len(edge_points) - 1, num_samples).astype(int)
    resampled = edge_points[indices]
    
    # Translate to origin
    start = resampled[0]
    resampled_centered = resampled - start
    
    # Rotate to align with x-axis
    end = resampled_centered[-1]
    angle = np.arctan2(end[1], end[0])
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    
    resampled_rotated = np.dot(resampled_centered, rotation_matrix.T)
    
    # Normalize by length
    length = np.linalg.norm(end)
    if length > 0:
        resampled_normalized = resampled_rotated / length
    else:
        resampled_normalized = resampled_rotated
    
    # Extract y-coordinates as descriptor
    descriptor = resampled_normalized[:, 1]
    
    return descriptor


def extract_edges(contour: np.ndarray, corners: List[Tuple[int, int]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract and analyze all four edges of a puzzle piece.
    
    Args:
        contour: The contour of the puzzle piece
        corners: List of 4 corner points
        
    Returns:
        Dictionary with edge information for each side
    """
    edges = {}
    edge_names = ['top', 'right', 'bottom', 'left']
    
    for i in range(4):
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 4]
        
        edge_points = extract_edge(contour, start_corner, end_corner)
        
        if len(edge_points) > 0:
            edge_type = classify_edge_type(edge_points)
            descriptor = create_edge_descriptor(edge_points)
            
            edges[edge_names[i]] = {
                'type': edge_type,
                'points': edge_points,
                'descriptor': descriptor,
                'start_corner': start_corner,
                'end_corner': end_corner,
                'length': cv2.arcLength(edge_points, False)
            }
        else:
            edges[edge_names[i]] = {
                'type': 'unknown',
                'points': np.array([]),
                'descriptor': np.array([]),
                'start_corner': start_corner,
                'end_corner': end_corner,
                'length': 0
            }
    
    return edges


def extract_features(processed_pieces: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract features from all processed pieces.
    
    Args:
        processed_pieces: Dictionary of preprocessed pieces
        
    Returns:
        Dictionary with extracted features for each piece
    """
    features = {}
    
    for piece_name, piece_data in processed_pieces.items():
        contour = piece_data['contour']
        
        # Find corners
        corners = find_corners_simple(contour)
        
        if len(corners) == 4:
            # Extract edges
            edges = extract_edges(contour, corners)
            
            features[piece_name] = {
                'corners': corners,
                'edges': edges,
                'area': piece_data['area'],
                'perimeter': piece_data['perimeter'],
                'contour': contour
            }
    
    return features