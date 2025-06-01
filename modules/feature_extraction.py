import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import math


def find_corners(contour: np.ndarray, num_corners: int = 4) -> List[Tuple[int, int]]:
    """
    Find the corners of a puzzle piece using the Douglas-Peucker algorithm
    and geometric analysis.
    
    Args:
        contour: The contour of the puzzle piece
        num_corners: Expected number of corners (4 for standard puzzle pieces)
        
    Returns:
        List of corner points as (x, y) tuples
    """
    # Approximate the contour to reduce points
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Calculate curvature for each point
    contour_points = contour.reshape(-1, 2)
    n_points = len(contour_points)
    
    # Calculate angles at each point
    angles = []
    window_size = 10  # Look at neighboring points
    
    for i in range(n_points):
        # Get neighboring points
        prev_idx = (i - window_size) % n_points
        next_idx = (i + window_size) % n_points
        
        prev_point = contour_points[prev_idx]
        curr_point = contour_points[i]
        next_point = contour_points[next_idx]
        
        # Calculate vectors
        v1 = prev_point - curr_point
        v2 = next_point - curr_point
        
        # Calculate angle
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle = np.abs(angle)
        if angle > np.pi:
            angle = 2 * np.pi - angle
            
        angles.append((angle, i, curr_point))
    
    # Sort by angle (corners have the smallest angles)
    angles.sort(key=lambda x: x[0])
    
    # Get the top N corner candidates
    corner_candidates = angles[:num_corners * 2]
    
    # Filter corners that are too close together
    corners = []
    min_distance = cv2.arcLength(contour, True) / (num_corners * 2)
    
    for angle, idx, point in corner_candidates:
        too_close = False
        for corner in corners:
            dist = np.linalg.norm(point - corner)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            corners.append(point)
            if len(corners) == num_corners:
                break
    
    # Sort corners clockwise starting from top-left
    if len(corners) == 4:
        center = np.mean(corners, axis=0)
        
        def angle_from_center(point):
            return math.atan2(point[1] - center[1], point[0] - center[0])
        
        corners.sort(key=angle_from_center)
        
        # Find top-left corner (minimum x+y)
        min_idx = np.argmin([p[0] + p[1] for p in corners])
        corners = corners[min_idx:] + corners[:min_idx]
    
    return [tuple(point) for point in corners]


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
    
    # Find indices of start and end points
    start_idx = None
    end_idx = None
    
    for i, point in enumerate(contour_points):
        if np.allclose(point, start_point, atol=5):
            start_idx = i
        if np.allclose(point, end_point, atol=5):
            end_idx = i
    
    if start_idx is None or end_idx is None:
        return np.array([])
    
    # Extract edge points
    if end_idx > start_idx:
        edge_points = contour_points[start_idx:end_idx+1]
    else:
        edge_points = np.concatenate([
            contour_points[start_idx:],
            contour_points[:end_idx+1]
        ])
    
    return edge_points


def classify_edge_type(edge_points: np.ndarray) -> str:
    """
    Classify whether an edge is convex, concave, or flat.
    
    Args:
        edge_points: Points along the edge
        
    Returns:
        'convex', 'concave', or 'flat'
    """
    if len(edge_points) < 3:
        return 'flat'
    
    # Fit a line through start and end points
    start = edge_points[0]
    end = edge_points[-1]
    
    # Calculate distances from points to the line
    distances = []
    for point in edge_points[1:-1]:
        # Distance from point to line
        dist = np.abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)
        
        # Determine which side of the line the point is on
        v1 = end - start
        v2 = point - start
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        distances.append(dist if cross > 0 else -dist)
    
    if not distances:
        return 'flat'
    
    # Determine edge type based on average deviation
    avg_distance = np.mean(distances)
    
    if abs(avg_distance) < 5:  # Threshold for flat edges
        return 'flat'
    elif avg_distance > 0:
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
        corners = find_corners(contour)
        
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