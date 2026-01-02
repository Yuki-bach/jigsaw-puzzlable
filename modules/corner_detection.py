"""
Corner detection module for puzzle pieces.
Uses multiple methods with fallback strategy.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy.signal import find_peaks
import itertools


def detect_corners(contour: np.ndarray) -> Tuple[List[Tuple[int, int]], str]:
    """
    Detect the four corners of a puzzle piece.
    Uses multiple methods with fallback.

    Args:
        contour: Contour points of the puzzle piece

    Returns:
        Tuple of (corners, method_used)
        corners: List of 4 corner points in order [top-left, top-right, bottom-right, bottom-left]
    """
    # Primary: Curvature-based detection
    corners = detect_by_curvature(contour)
    if corners is not None and validate_corners(corners):
        return sort_corners(corners), "curvature"

    # Fallback 1: PCA-based detection
    corners = detect_by_pca(contour)
    if corners is not None and validate_corners(corners):
        return sort_corners(corners), "pca"

    # Fallback 2: Bounding rectangle
    corners = detect_by_rect(contour)
    return sort_corners(corners), "rect"


def compute_curvature(contour: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Compute k-curvature for each point in the contour.

    Args:
        contour: Contour points
        k: Window size for curvature computation

    Returns:
        Array of curvature values
    """
    contour_2d = contour.reshape(-1, 2).astype(np.float64)
    n = len(contour_2d)
    curvature = np.zeros(n)

    for i in range(n):
        p_prev = contour_2d[(i - k) % n]
        p_curr = contour_2d[i]
        p_next = contour_2d[(i + k) % n]

        # Compute signed area (cross product)
        v1 = p_curr - p_prev
        v2 = p_next - p_prev

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        base = np.linalg.norm(p_next - p_prev)

        if base > 1e-6:
            curvature[i] = 2 * cross / (base ** 2)

    return curvature


def detect_by_curvature(contour: np.ndarray, k: int = 20) -> Optional[List[Tuple[int, int]]]:
    """
    Detect corners using curvature analysis.

    Args:
        contour: Contour points
        k: Window size for curvature

    Returns:
        List of 4 corner points or None if detection fails
    """
    contour_2d = contour.reshape(-1, 2)
    n = len(contour_2d)

    if n < 100:
        return None

    # Compute curvature
    curvature = compute_curvature(contour, k)

    # Find peaks in curvature (convex corners have high positive curvature)
    # Use percentile for adaptive threshold
    height_threshold = np.percentile(curvature, 90)

    # Minimum distance between peaks (about 1/8 of contour length)
    min_distance = max(n // 8, 20)

    peaks, properties = find_peaks(
        curvature,
        height=height_threshold,
        distance=min_distance
    )

    if len(peaks) < 4:
        return None

    # Select the 4 peaks with highest curvature that form a valid quadrilateral
    peak_curvatures = [(i, curvature[i]) for i in peaks]
    peak_curvatures.sort(key=lambda x: x[1], reverse=True)

    # Try combinations of top peaks
    candidates = [idx for idx, _ in peak_curvatures[:min(12, len(peak_curvatures))]]

    for combo in itertools.combinations(candidates, 4):
        corners = [tuple(contour_2d[i]) for i in combo]
        if is_valid_quadrilateral(corners):
            return corners

    return None


def detect_by_pca(contour: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    Detect corners using PCA axis extremes.

    Args:
        contour: Contour points

    Returns:
        List of 4 corner points or None
    """
    contour_2d = contour.reshape(-1, 2).astype(np.float64)

    # Compute centroid
    centroid = np.mean(contour_2d, axis=0)

    # Center the points
    centered = contour_2d - centroid

    # Compute PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project points onto principal axes
    projected = centered @ eigenvectors

    # Find extreme points along each axis
    corners = []

    # Max and min along first principal axis
    idx_max_1 = np.argmax(projected[:, 0])
    idx_min_1 = np.argmin(projected[:, 0])

    # Max and min along second principal axis
    idx_max_2 = np.argmax(projected[:, 1])
    idx_min_2 = np.argmin(projected[:, 1])

    corner_indices = list(set([idx_max_1, idx_min_1, idx_max_2, idx_min_2]))

    if len(corner_indices) < 4:
        # Find additional corners if we have duplicates
        all_indices = set(range(len(contour_2d)))
        remaining = all_indices - set(corner_indices)

        # Add points that are farthest from centroid
        distances = np.linalg.norm(centered, axis=1)
        for idx in sorted(remaining, key=lambda x: distances[x], reverse=True):
            corner_indices.append(idx)
            if len(corner_indices) == 4:
                break

    if len(corner_indices) < 4:
        return None

    corners = [tuple(contour_2d[i].astype(int)) for i in corner_indices[:4]]

    return corners


def detect_by_rect(contour: np.ndarray) -> List[Tuple[int, int]]:
    """
    Detect corners using minimum area rectangle.

    Args:
        contour: Contour points

    Returns:
        List of 4 corner points
    """
    contour_2d = contour.reshape(-1, 2)

    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # Find the closest contour point to each box corner
    corners = []
    for box_corner in box:
        distances = np.linalg.norm(contour_2d - box_corner, axis=1)
        closest_idx = np.argmin(distances)
        corners.append(tuple(contour_2d[closest_idx]))

    return corners


def validate_corners(corners: List[Tuple[int, int]]) -> bool:
    """
    Validate that 4 corners form a reasonable quadrilateral.

    Args:
        corners: List of 4 corner points

    Returns:
        True if valid, False otherwise
    """
    if len(corners) != 4:
        return False

    return is_valid_quadrilateral(corners)


def is_valid_quadrilateral(corners: List[Tuple[int, int]]) -> bool:
    """
    Check if 4 points form a valid quadrilateral suitable for a puzzle piece.

    Args:
        corners: List of 4 corner points

    Returns:
        True if valid quadrilateral
    """
    if len(corners) != 4:
        return False

    # Convert to numpy for easier computation
    pts = np.array(corners, dtype=np.float64)

    # Check that all points are distinct
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < 10:
                return False

    # Sort corners for consistent ordering
    sorted_corners = sort_corners_internal(corners)
    pts = np.array(sorted_corners, dtype=np.float64)

    # Check angles (should be roughly 60-120 degrees for a puzzle piece corner)
    for i in range(4):
        p_prev = pts[(i - 1) % 4]
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-6 or n2 < 1e-6:
            return False

        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        if angle < 45 or angle > 135:
            return False

    # Check that diagonals intersect (convex quadrilateral)
    # Diagonal 1: pts[0] to pts[2]
    # Diagonal 2: pts[1] to pts[3]
    def line_intersection(p1, p2, p3, p4):
        """Check if line segments p1-p2 and p3-p4 intersect."""
        d1 = p2 - p1
        d2 = p4 - p3
        d3 = p3 - p1

        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-6:
            return False

        t1 = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
        t2 = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

        return 0 < t1 < 1 and 0 < t2 < 1

    if not line_intersection(pts[0], pts[2], pts[1], pts[3]):
        return False

    return True


def sort_corners_internal(corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Sort corners in order: top-left, top-right, bottom-right, bottom-left.
    Internal helper function.
    """
    pts = np.array(corners, dtype=np.float64)

    # Find centroid
    centroid = np.mean(pts, axis=0)

    # Calculate angles from centroid
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])

    # Sort by angle (counterclockwise from right)
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]

    # Rotate so that top-left is first
    # Top-left should be the point with smallest x+y sum (upper left)
    sums = sorted_pts[:, 0] + sorted_pts[:, 1]
    start_idx = np.argmin(sums)

    # Reorder
    reordered = np.roll(sorted_pts, -start_idx, axis=0)

    return [tuple(p.astype(int)) for p in reordered]


def sort_corners(corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Sort corners in order: top-left, top-right, bottom-right, bottom-left.

    Args:
        corners: List of 4 corner points (unordered)

    Returns:
        Sorted list of corners
    """
    return sort_corners_internal(corners)
