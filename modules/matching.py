import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import cdist


def are_compatible_types(type1: str, type2: str) -> bool:
    """
    Check if two edge types are compatible for matching.
    
    Args:
        type1: Type of first edge ('convex', 'concave', or 'flat')
        type2: Type of second edge
        
    Returns:
        True if edges can potentially match
    """
    # Convex matches with concave
    if (type1 == 'convex' and type2 == 'concave') or \
       (type1 == 'concave' and type2 == 'convex'):
        return True
    
    # Flat edges can only match with flat edges
    if type1 == 'flat' and type2 == 'flat':
        return True
    
    return False


def match_edges(edge1: Dict[str, Any], edge2: Dict[str, Any], 
                piece1_name: str, piece2_name: str) -> float:
    """
    Calculate similarity score between two edges.
    
    Args:
        edge1: First edge data
        edge2: Second edge data
        piece1_name: Name of first piece
        piece2_name: Name of second piece
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Check if edge types are compatible
    if not are_compatible_types(edge1['type'], edge2['type']):
        return 0.0
    
    # If either edge has no descriptor, return 0
    if len(edge1['descriptor']) == 0 or len(edge2['descriptor']) == 0:
        return 0.0
    
    # Compare edge lengths (should be similar)
    length_ratio = min(edge1['length'], edge2['length']) / max(edge1['length'], edge2['length'])
    if length_ratio < 0.8:  # Edges with very different lengths unlikely to match
        return 0.0
    
    # For convex-concave pairs, invert one descriptor
    descriptor1 = edge1['descriptor'].copy()
    descriptor2 = edge2['descriptor'].copy()
    
    if edge1['type'] == 'convex' and edge2['type'] == 'concave':
        descriptor2 = -descriptor2
    elif edge1['type'] == 'concave' and edge2['type'] == 'convex':
        descriptor1 = -descriptor1
    
    # Calculate correlation between descriptors
    if len(descriptor1) == len(descriptor2):
        # Normalize descriptors
        if np.std(descriptor1) > 0 and np.std(descriptor2) > 0:
            descriptor1_norm = (descriptor1 - np.mean(descriptor1)) / np.std(descriptor1)
            descriptor2_norm = (descriptor2 - np.mean(descriptor2)) / np.std(descriptor2)
            
            # Calculate correlation
            correlation = np.corrcoef(descriptor1_norm, descriptor2_norm)[0, 1]
            
            # Convert correlation to similarity score (0 to 1)
            similarity = (correlation + 1) / 2
            
            # Weight by length ratio
            final_score = similarity * (0.7 + 0.3 * length_ratio)
            
            return final_score
    
    return 0.0


def find_matching_pairs(features: Dict[str, Dict[str, Any]], 
                       threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Find all matching pairs of edges between pieces.
    
    Args:
        features: Dictionary of extracted features for all pieces
        threshold: Minimum similarity score to consider a match
        
    Returns:
        List of matching pairs with similarity scores
    """
    matches = []
    piece_names = list(features.keys())
    
    # Compare all pairs of pieces
    for i in range(len(piece_names)):
        for j in range(i + 1, len(piece_names)):
            piece1_name = piece_names[i]
            piece2_name = piece_names[j]
            
            piece1_features = features[piece1_name]
            piece2_features = features[piece2_name]
            
            # Compare all edge combinations
            edge_names = ['top', 'right', 'bottom', 'left']
            
            for edge1_name in edge_names:
                for edge2_name in edge_names:
                    edge1 = piece1_features['edges'][edge1_name]
                    edge2 = piece2_features['edges'][edge2_name]
                    
                    score = match_edges(edge1, edge2, piece1_name, piece2_name)
                    
                    if score >= threshold:
                        matches.append({
                            'piece1': piece1_name,
                            'piece2': piece2_name,
                            'edge1': edge1_name,
                            'edge2': edge2_name,
                            'score': score,
                            'edge1_type': edge1['type'],
                            'edge2_type': edge2['type']
                        })
    
    # Sort matches by score (highest first)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches


def form_groups(matches: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Form groups of connected pieces from pairwise matches.
    
    Args:
        matches: List of matching pairs
        
    Returns:
        List of groups, where each group is a list of piece names
    """
    # Create adjacency list
    connections = {}
    
    for match in matches:
        piece1 = match['piece1']
        piece2 = match['piece2']
        
        if piece1 not in connections:
            connections[piece1] = []
        if piece2 not in connections:
            connections[piece2] = []
        
        connections[piece1].append(piece2)
        connections[piece2].append(piece1)
    
    # Find connected components using DFS
    visited = set()
    groups = []
    
    def dfs(piece, group):
        visited.add(piece)
        group.append(piece)
        
        if piece in connections:
            for neighbor in connections[piece]:
                if neighbor not in visited:
                    dfs(neighbor, group)
    
    for piece in connections:
        if piece not in visited:
            group = []
            dfs(piece, group)
            if len(group) >= 2:  # Only keep groups with at least 2 pieces
                groups.append(sorted(group))
    
    # Sort groups by size (largest first)
    groups.sort(key=len, reverse=True)
    
    return groups


def validate_group(group: List[str], features: Dict[str, Dict[str, Any]], 
                  matches: List[Dict[str, Any]]) -> bool:
    """
    Validate that a group of pieces can actually fit together.
    
    Args:
        group: List of piece names in the group
        features: Dictionary of piece features
        matches: List of all matches
        
    Returns:
        True if the group is valid
    """
    # For now, return True for all groups
    # In a more sophisticated implementation, we would check:
    # 1. Spatial consistency (pieces don't overlap)
    # 2. All connections are valid
    # 3. No impossible configurations
    
    return True