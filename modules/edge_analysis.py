"""
Edge analysis module for puzzle pieces.
Handles edge segmentation, classification, and profile extraction.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def analyze_edges(contour: np.ndarray, corners: List[Tuple[int, int]]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all four edges of a puzzle piece.

    Args:
        contour: Contour points of the puzzle piece
        corners: List of 4 corner points [top-left, top-right, bottom-right, bottom-left]

    Returns:
        Dictionary with edge information for each side (top, right, bottom, left)
    """
    edge_names = ['top', 'right', 'bottom', 'left']
    edges = {}

    for i, edge_name in enumerate(edge_names):
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 4]

        # Extract edge points
        edge_points = extract_edge_points(contour, start_corner, end_corner)

        if edge_points is None or len(edge_points) < 10:
            edges[edge_name] = create_empty_edge(start_corner, end_corner)
            continue

        # Classify edge type (convex, concave, flat)
        edge_type, type_confidence = classify_edge(edge_points)

        # Extract normalized profile
        profile = extract_profile(edge_points)

        # Calculate edge length
        length = cv2.arcLength(edge_points.reshape(-1, 1, 2).astype(np.int32), closed=False)

        edges[edge_name] = {
            'type': edge_type,
            'type_confidence': type_confidence,
            'profile': profile,
            'length': length,
            'points': edge_points,
            'start_corner': start_corner,
            'end_corner': end_corner
        }

    return edges


def create_empty_edge(start_corner: Tuple[int, int], end_corner: Tuple[int, int]) -> Dict[str, Any]:
    """Create an empty edge entry for failed extraction."""
    return {
        'type': 'unknown',
        'type_confidence': 0.0,
        'profile': np.array([]),
        'length': 0.0,
        'points': np.array([]),
        'start_corner': start_corner,
        'end_corner': end_corner
    }


def extract_edge_points(contour: np.ndarray,
                        start_corner: Tuple[int, int],
                        end_corner: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Extract points along an edge between two corners.

    Args:
        contour: Full contour of the puzzle piece
        start_corner: Starting corner point
        end_corner: Ending corner point

    Returns:
        Array of points along the edge, or None if extraction fails
    """
    contour_2d = contour.reshape(-1, 2)
    n = len(contour_2d)

    # Find indices closest to corners
    start_idx = find_closest_index(contour_2d, start_corner)
    end_idx = find_closest_index(contour_2d, end_corner)

    # Extract edge points (go in the shorter direction around the contour)
    if end_idx >= start_idx:
        path1_length = end_idx - start_idx
        path2_length = n - path1_length
    else:
        path1_length = n - start_idx + end_idx
        path2_length = n - path1_length

    # Choose the shorter path
    if path1_length <= path2_length:
        if end_idx >= start_idx:
            edge_points = contour_2d[start_idx:end_idx + 1]
        else:
            edge_points = np.vstack([contour_2d[start_idx:], contour_2d[:end_idx + 1]])
    else:
        if end_idx >= start_idx:
            edge_points = np.vstack([contour_2d[end_idx:], contour_2d[:start_idx + 1]])[::-1]
        else:
            edge_points = contour_2d[end_idx:start_idx + 1][::-1]

    if len(edge_points) < 5:
        return None

    return edge_points


def find_closest_index(contour_2d: np.ndarray, point: Tuple[int, int]) -> int:
    """Find the index of the closest contour point to a given point."""
    distances = np.linalg.norm(contour_2d - np.array(point), axis=1)
    return int(np.argmin(distances))


def classify_edge(edge_points: np.ndarray) -> Tuple[str, float]:
    """
    Classify an edge as convex, concave, or flat.

    Uses the deviation from a straight line between start and end points.
    Focuses on the central region where tabs/blanks typically appear.

    Args:
        edge_points: Points along the edge

    Returns:
        Tuple of (edge_type, confidence)
    """
    if len(edge_points) < 10:
        return 'flat', 0.5

    start = edge_points[0].astype(np.float64)
    end = edge_points[-1].astype(np.float64)

    # Compute signed distances from the line
    deviations = compute_perpendicular_distances(edge_points, start, end)

    if len(deviations) == 0:
        return 'flat', 0.5

    # Edge length
    edge_length = np.linalg.norm(end - start)

    if edge_length < 1e-6:
        return 'flat', 0.5

    # Focus on the central region (20%-80%) where tabs/blanks typically appear
    n = len(deviations)
    center_start = int(n * 0.2)
    center_end = int(n * 0.8)
    center_deviations = deviations[center_start:center_end]

    if len(center_deviations) == 0:
        center_deviations = deviations

    # Statistics focused on center region
    max_deviation = np.max(np.abs(deviations))
    center_max = np.max(center_deviations) if len(center_deviations) > 0 else 0
    center_min = np.min(center_deviations) if len(center_deviations) > 0 else 0

    # Relative deviation
    relative_max = max_deviation / edge_length

    # Classification threshold - extremely low since there are no edge pieces
    # All edges should be either convex (tab) or concave (blank)
    flat_threshold = 0.005  # Less than 0.5% deviation is flat (essentially never)

    if relative_max < flat_threshold:
        return 'flat', 0.95

    # Determine direction based on the dominant deviation in the center region
    # This depends on the contour orientation

    # Find the peak deviation (maximum absolute value in center)
    center_peak_idx = np.argmax(np.abs(center_deviations))
    center_peak_value = center_deviations[center_peak_idx]

    # Calculate the prominence of the peak
    relative_center_peak = abs(center_peak_value) / edge_length

    if relative_center_peak < flat_threshold:
        return 'flat', 0.9

    # Determine type based on the sign of the dominant peak
    # With counter-clockwise contour, positive deviation = left of line = outside
    if center_peak_value > 0:
        # Positive deviation = outside = convex (tab pointing outward)
        confidence = min(0.99, 0.7 + relative_center_peak * 2)
        return 'convex', confidence
    else:
        # Negative deviation = inside = concave (blank/indent)
        confidence = min(0.99, 0.7 + relative_center_peak * 2)
        return 'concave', confidence


def compute_perpendicular_distances(points: np.ndarray,
                                    line_start: np.ndarray,
                                    line_end: np.ndarray) -> np.ndarray:
    """
    Compute signed perpendicular distances from points to a line.

    Positive values indicate points to the left of the line (when going from start to end).

    Args:
        points: Array of points
        line_start: Start point of the line
        line_end: End point of the line

    Returns:
        Array of signed distances
    """
    points_float = points.astype(np.float64)
    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)

    if line_length < 1e-6:
        return np.zeros(len(points))

    # Normalized line direction
    line_dir = line_vec / line_length

    # Perpendicular vector (rotate 90 degrees counterclockwise)
    perp = np.array([-line_dir[1], line_dir[0]])

    # Compute distances
    point_vecs = points_float - line_start
    distances = np.dot(point_vecs, perp)

    return distances


def extract_profile(edge_points: np.ndarray, num_samples: int = 64) -> np.ndarray:
    """
    Extract a normalized profile of the edge shape.

    The profile represents the deviation from a straight line at regular intervals.

    Args:
        edge_points: Points along the edge
        num_samples: Number of samples in the output profile

    Returns:
        Normalized profile array
    """
    if len(edge_points) < 10:
        return np.zeros(num_samples)

    # Resample edge to fixed number of points
    edge_resampled = resample_edge(edge_points, num_samples)

    # Compute deviations from straight line
    start = edge_resampled[0].astype(np.float64)
    end = edge_resampled[-1].astype(np.float64)

    deviations = compute_perpendicular_distances(edge_resampled, start, end)

    # Normalize by edge length
    edge_length = np.linalg.norm(end - start)
    if edge_length > 1e-6:
        deviations = deviations / edge_length

    return deviations


def resample_edge(edge_points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resample edge points to a fixed number of equally-spaced points.

    Args:
        edge_points: Original edge points
        num_samples: Target number of points

    Returns:
        Resampled edge points
    """
    n = len(edge_points)

    if n <= num_samples:
        # Interpolate if we have fewer points
        indices = np.linspace(0, n - 1, num_samples)
    else:
        # Subsample if we have more points
        indices = np.linspace(0, n - 1, num_samples)

    # Linear interpolation
    indices_floor = np.floor(indices).astype(int)
    indices_ceil = np.ceil(indices).astype(int)
    indices_ceil = np.clip(indices_ceil, 0, n - 1)

    weights = indices - indices_floor

    resampled = (
        (1 - weights)[:, np.newaxis] * edge_points[indices_floor] +
        weights[:, np.newaxis] * edge_points[indices_ceil]
    )

    return resampled


def extract_all_features(processed_pieces: Dict[str, Dict[str, Any]],
                         corners_data: Dict[str, Tuple[List[Tuple[int, int]], str]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract features from all processed pieces.

    Args:
        processed_pieces: Dictionary of preprocessed pieces
        corners_data: Dictionary of detected corners for each piece

    Returns:
        Dictionary with extracted features for each piece
    """
    features = {}

    for piece_name, piece_data in processed_pieces.items():
        if piece_name not in corners_data:
            continue

        corners, corner_method = corners_data[piece_name]
        contour = piece_data['contour']

        # Analyze edges
        edges = analyze_edges(contour, corners)

        features[piece_name] = {
            'corners': corners,
            'corner_method': corner_method,
            'edges': edges,
            'area': piece_data['area'],
            'perimeter': piece_data['perimeter'],
            'contour': contour
        }

    return features
