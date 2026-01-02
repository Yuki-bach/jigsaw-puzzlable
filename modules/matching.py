"""
Matching module for puzzle pieces.
Implements two-stage matching: type filter -> profile distance.
"""

import numpy as np
from typing import Dict, List, Any, Optional


def find_matching_pairs(features: Dict[str, Dict[str, Any]],
                        threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Find all matching pairs of edges between pieces.

    Uses two-stage matching:
    1. Filter by edge type compatibility (convex-concave or flat-flat)
    2. Compute profile distance and score

    Args:
        features: Dictionary of extracted features for all pieces
        threshold: Minimum score to consider a match

    Returns:
        List of matching pairs with scores
    """
    matches = []
    piece_names = list(features.keys())
    edge_names = ['top', 'right', 'bottom', 'left']

    # Organize edges by type for faster filtering
    convex_edges = []
    concave_edges = []
    flat_edges = []

    for piece_name in piece_names:
        piece_features = features[piece_name]
        for edge_name in edge_names:
            edge = piece_features['edges'][edge_name]
            entry = (piece_name, edge_name, edge)

            if edge['type'] == 'convex':
                convex_edges.append(entry)
            elif edge['type'] == 'concave':
                concave_edges.append(entry)
            elif edge['type'] == 'flat':
                flat_edges.append(entry)

    # Match convex with concave
    for conv_piece, conv_edge_name, conv_edge in convex_edges:
        for conc_piece, conc_edge_name, conc_edge in concave_edges:
            # Skip same piece
            if conv_piece == conc_piece:
                continue

            score = compute_match_score(conv_edge, conc_edge)

            if score >= threshold:
                matches.append({
                    'piece1': conv_piece,
                    'edge1': conv_edge_name,
                    'piece2': conc_piece,
                    'edge2': conc_edge_name,
                    'score': score,
                    'edge1_type': 'convex',
                    'edge2_type': 'concave'
                })

    # Match flat with flat (less common, usually border pieces)
    for i, (flat1_piece, flat1_edge_name, flat1_edge) in enumerate(flat_edges):
        for flat2_piece, flat2_edge_name, flat2_edge in flat_edges[i + 1:]:
            # Skip same piece
            if flat1_piece == flat2_piece:
                continue

            score = compute_match_score_flat(flat1_edge, flat2_edge)

            if score >= threshold:
                matches.append({
                    'piece1': flat1_piece,
                    'edge1': flat1_edge_name,
                    'piece2': flat2_piece,
                    'edge2': flat2_edge_name,
                    'score': score,
                    'edge1_type': 'flat',
                    'edge2_type': 'flat'
                })

    # Sort by score (highest first)
    matches.sort(key=lambda x: x['score'], reverse=True)

    return matches


def compute_match_score(edge1: Dict[str, Any], edge2: Dict[str, Any]) -> float:
    """
    Compute matching score between a convex and concave edge.

    Args:
        edge1: First edge data (convex or concave)
        edge2: Second edge data (the opposite type)

    Returns:
        Score between 0.0 and 1.0
    """
    # Check if profiles exist
    profile1 = edge1.get('profile', np.array([]))
    profile2 = edge2.get('profile', np.array([]))

    if len(profile1) == 0 or len(profile2) == 0:
        return 0.0

    if len(profile1) != len(profile2):
        return 0.0

    # Length check
    length1 = edge1.get('length', 0)
    length2 = edge2.get('length', 0)

    if length1 <= 0 or length2 <= 0:
        return 0.0

    length_ratio = min(length1, length2) / max(length1, length2)

    if length_ratio < 0.85:
        return 0.0

    # Prepare profiles for comparison
    # For convex-concave matching, we need to flip one profile
    # Convex edge should have negative deviations (bulging outward)
    # Concave edge should have positive deviations (indenting inward)

    p1 = profile1.copy()
    p2 = profile2.copy()

    # If edge1 is convex, we flip edge2 (reverse and negate)
    # to match the complementary shape
    if edge1['type'] == 'convex':
        p2 = -p2[::-1]
    elif edge2['type'] == 'convex':
        p1 = -p1[::-1]

    # Compute L2 distance (normalized)
    distance = np.linalg.norm(p1 - p2) / len(p1)

    # Convert distance to score
    # Smaller distance = higher score
    # Using exponential decay: score = exp(-distance * k)
    # k controls the sensitivity
    score = np.exp(-distance * 5)

    # Apply length ratio as a multiplier
    final_score = score * length_ratio

    # Apply type confidence as a factor
    confidence1 = edge1.get('type_confidence', 1.0)
    confidence2 = edge2.get('type_confidence', 1.0)
    final_score *= (confidence1 * confidence2) ** 0.25

    return float(final_score)


def compute_match_score_flat(edge1: Dict[str, Any], edge2: Dict[str, Any]) -> float:
    """
    Compute matching score between two flat edges.

    Flat edges typically occur on border pieces and should match
    with other border pieces of similar length.

    Args:
        edge1: First flat edge
        edge2: Second flat edge

    Returns:
        Score between 0.0 and 1.0
    """
    # Length check (more strict for flat edges)
    length1 = edge1.get('length', 0)
    length2 = edge2.get('length', 0)

    if length1 <= 0 or length2 <= 0:
        return 0.0

    length_ratio = min(length1, length2) / max(length1, length2)

    if length_ratio < 0.90:
        return 0.0

    # For flat edges, we mainly care about length similarity
    # The profile should be near-zero for truly flat edges
    profile1 = edge1.get('profile', np.array([]))
    profile2 = edge2.get('profile', np.array([]))

    if len(profile1) == 0 or len(profile2) == 0:
        return 0.0

    # Check that both profiles are actually flat
    max_dev1 = np.max(np.abs(profile1))
    max_dev2 = np.max(np.abs(profile2))

    if max_dev1 > 0.05 or max_dev2 > 0.05:
        return 0.0

    # Score based on length similarity
    score = length_ratio * 0.9  # Flat-flat matches get slightly lower max score

    return float(score)


def filter_by_type(edge1: Dict[str, Any], edge2: Dict[str, Any]) -> bool:
    """
    Check if two edges are compatible types for matching.

    Args:
        edge1: First edge
        edge2: Second edge

    Returns:
        True if edges can potentially match
    """
    type1 = edge1.get('type', 'unknown')
    type2 = edge2.get('type', 'unknown')

    if type1 == 'unknown' or type2 == 'unknown':
        return False

    # Convex-concave pairs
    if (type1 == 'convex' and type2 == 'concave') or \
       (type1 == 'concave' and type2 == 'convex'):
        return True

    # Flat-flat pairs
    if type1 == 'flat' and type2 == 'flat':
        return True

    return False


def get_edge_statistics(features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about all edges for analysis.

    Args:
        features: Dictionary of extracted features

    Returns:
        Dictionary with edge statistics
    """
    type_counts = {'convex': 0, 'concave': 0, 'flat': 0, 'unknown': 0}
    lengths = []
    confidences = []

    for piece_features in features.values():
        for edge_name in ['top', 'right', 'bottom', 'left']:
            edge = piece_features['edges'][edge_name]
            edge_type = edge.get('type', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1

            if edge.get('length', 0) > 0:
                lengths.append(edge['length'])

            if 'type_confidence' in edge:
                confidences.append(edge['type_confidence'])

    return {
        'type_counts': type_counts,
        'total_edges': sum(type_counts.values()),
        'avg_length': np.mean(lengths) if lengths else 0,
        'std_length': np.std(lengths) if lengths else 0,
        'avg_confidence': np.mean(confidences) if confidences else 0
    }
