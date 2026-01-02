"""
Group formation module for puzzle pieces.
Implements mutual best match + score gap filtering for high confidence grouping.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict


def form_groups(matches: List[Dict[str, Any]],
                min_score: float = 0.7,
                score_gap_ratio: float = 1.2) -> List[Dict[str, Any]]:
    """
    Form groups of connected pieces from validated matches.

    Uses two validation criteria:
    1. Mutual best match: Both edges consider each other as their best match
    2. Score gap: The best match score is significantly higher than the second best

    Args:
        matches: List of matching pairs with scores
        min_score: Minimum score threshold
        score_gap_ratio: Required ratio between best and second best score

    Returns:
        List of groups with piece connections
    """
    if not matches:
        return []

    # Find best matches for each edge
    best_matches = find_best_matches(matches)

    # Validate matches
    validated_matches = []

    for match in matches:
        if match['score'] < min_score:
            continue

        key1 = (match['piece1'], match['edge1'])
        key2 = (match['piece2'], match['edge2'])

        # Check mutual best
        is_best_for_1 = best_matches.get(key1) is not None and \
                        matches_equal(best_matches[key1], match)
        is_best_for_2 = best_matches.get(key2) is not None and \
                        matches_equal(best_matches[key2], match)

        if not (is_best_for_1 and is_best_for_2):
            continue

        # Check score gap
        second_best_1 = find_second_best(matches, key1, match)
        second_best_2 = find_second_best(matches, key2, match)

        gap_ok = True

        if second_best_1 is not None and second_best_1['score'] > 0:
            if match['score'] / second_best_1['score'] < score_gap_ratio:
                gap_ok = False

        if second_best_2 is not None and second_best_2['score'] > 0:
            if match['score'] / second_best_2['score'] < score_gap_ratio:
                gap_ok = False

        if gap_ok:
            validated_matches.append(match)

    # Build groups from validated matches
    groups = build_groups(validated_matches)

    return groups


def find_best_matches(matches: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Find the best match for each edge.

    Args:
        matches: List of all matches

    Returns:
        Dictionary mapping (piece, edge) to their best match
    """
    best = {}

    for match in matches:
        key1 = (match['piece1'], match['edge1'])
        key2 = (match['piece2'], match['edge2'])

        if key1 not in best or match['score'] > best[key1]['score']:
            best[key1] = match

        if key2 not in best or match['score'] > best[key2]['score']:
            best[key2] = match

    return best


def find_second_best(matches: List[Dict[str, Any]],
                     key: Tuple[str, str],
                     exclude: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the second best match for an edge.

    Args:
        matches: List of all matches
        key: (piece, edge) tuple to find second best for
        exclude: The match to exclude (the best match)

    Returns:
        The second best match or None
    """
    second_best = None
    best_score = -1

    for match in matches:
        if matches_equal(match, exclude):
            continue

        # Check if this match involves the key
        if (match['piece1'], match['edge1']) == key or \
           (match['piece2'], match['edge2']) == key:
            if match['score'] > best_score:
                best_score = match['score']
                second_best = match

    return second_best


def matches_equal(m1: Dict[str, Any], m2: Dict[str, Any]) -> bool:
    """Check if two matches are the same."""
    return (m1['piece1'] == m2['piece1'] and
            m1['edge1'] == m2['edge1'] and
            m1['piece2'] == m2['piece2'] and
            m1['edge2'] == m2['edge2'])


def build_groups(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build connected groups from matches using Union-Find.

    Args:
        matches: List of validated matches

    Returns:
        List of groups with their pieces and connections
    """
    if not matches:
        return []

    # Union-Find implementation
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build union-find structure
    for match in matches:
        union(match['piece1'], match['piece2'])

    # Group pieces and connections by root
    groups_dict = defaultdict(lambda: {'pieces': set(), 'connections': []})

    for match in matches:
        root = find(match['piece1'])
        groups_dict[root]['pieces'].add(match['piece1'])
        groups_dict[root]['pieces'].add(match['piece2'])
        groups_dict[root]['connections'].append(match)

    # Convert to list format
    groups = []
    for root, data in groups_dict.items():
        pieces = sorted(list(data['pieces']))
        connections = data['connections']

        if len(pieces) >= 2:
            # Calculate group statistics
            scores = [c['score'] for c in connections]

            groups.append({
                'pieces': pieces,
                'connections': connections,
                'size': len(pieces),
                'num_connections': len(connections),
                'avg_confidence': float(np.mean(scores)),
                'min_confidence': float(np.min(scores)),
                'max_confidence': float(np.max(scores))
            })

    # Sort by group size (largest first)
    groups.sort(key=lambda x: x['size'], reverse=True)

    return groups


def get_connection_summary(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of all connections.

    Args:
        groups: List of formed groups

    Returns:
        Summary statistics
    """
    total_pieces = 0
    total_connections = 0
    all_scores = []

    for group in groups:
        total_pieces += group['size']
        total_connections += group['num_connections']
        all_scores.extend([c['score'] for c in group['connections']])

    return {
        'num_groups': len(groups),
        'total_pieces': total_pieces,
        'total_connections': total_connections,
        'avg_score': float(np.mean(all_scores)) if all_scores else 0,
        'min_score': float(np.min(all_scores)) if all_scores else 0,
        'max_score': float(np.max(all_scores)) if all_scores else 0
    }


def validate_group_consistency(group: Dict[str, Any]) -> bool:
    """
    Validate that a group has consistent connections.

    Each edge should have at most one connection.

    Args:
        group: Group to validate

    Returns:
        True if consistent, False otherwise
    """
    edge_connections = defaultdict(list)

    for conn in group['connections']:
        key1 = (conn['piece1'], conn['edge1'])
        key2 = (conn['piece2'], conn['edge2'])
        edge_connections[key1].append(conn)
        edge_connections[key2].append(conn)

    # Check that each edge has at most one connection
    for key, connections in edge_connections.items():
        if len(connections) > 1:
            return False

    return True


def filter_groups_by_consistency(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out groups that have inconsistent connections.

    Args:
        groups: List of groups

    Returns:
        Filtered list of consistent groups
    """
    return [g for g in groups if validate_group_consistency(g)]
