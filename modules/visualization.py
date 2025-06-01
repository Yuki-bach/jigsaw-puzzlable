import cv2
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json


def visualize_groups(groups: List[List[str]], pieces: Dict[str, np.ndarray],
                    features: Dict[str, Dict[str, Any]], matches: List[Dict[str, Any]],
                    output_path: str) -> None:
    """
    Create a visualization of discovered groups.
    
    Args:
        groups: List of groups (each group is a list of piece names)
        pieces: Dictionary of original piece images
        features: Dictionary of piece features
        matches: List of all matches
        output_path: Path to save the output image
    """
    # Calculate layout
    num_groups = len(groups)
    if num_groups == 0:
        # Create empty result image
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.text(0.5, 0.5, 'No groups found', fontsize=20, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Determine grid layout
    cols = min(3, num_groups)
    rows = (num_groups + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    # Create a mapping of matches for each piece pair
    match_map = {}
    for match in matches:
        key = (match['piece1'], match['piece2'])
        if key not in match_map:
            match_map[key] = []
        match_map[key].append(match)
    
    # Visualize each group
    for group_idx, group in enumerate(groups):
        row = group_idx // cols
        col = group_idx % cols
        ax = axes[row][col]
        
        # Create a canvas for the group
        group_canvas = create_group_visualization(group, pieces, features, match_map)
        
        # Display the group
        ax.imshow(group_canvas)
        ax.set_title(f'Group {group_idx + 1} ({len(group)} pieces)', fontsize=12)
        ax.axis('off')
        
        # Add piece names
        piece_text = ', '.join(group[:5])
        if len(group) > 5:
            piece_text += f' ... (+{len(group) - 5} more)'
        ax.text(0.5, -0.05, piece_text, transform=ax.transAxes,
                fontsize=8, ha='center', va='top')
    
    # Hide empty subplots
    for i in range(num_groups, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_group_visualization(group: List[str], pieces: Dict[str, np.ndarray],
                             features: Dict[str, Dict[str, Any]], 
                             match_map: Dict) -> np.ndarray:
    """
    Create a visualization of a single group of pieces.
    
    Args:
        group: List of piece names in the group
        pieces: Dictionary of original piece images
        features: Dictionary of piece features
        match_map: Dictionary mapping piece pairs to their matches
        
    Returns:
        Numpy array representing the group visualization
    """
    # For simplicity, arrange pieces in a grid
    grid_size = int(np.ceil(np.sqrt(len(group))))
    piece_size = 150  # Size of each piece in the visualization
    canvas_size = grid_size * piece_size + (grid_size - 1) * 20  # 20px spacing
    
    # Create white canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    
    # Place pieces
    for idx, piece_name in enumerate(group):
        row = idx // grid_size
        col = idx % grid_size
        
        if piece_name in pieces:
            # Get piece image
            piece_img = pieces[piece_name]
            
            # Resize piece to fit
            aspect_ratio = piece_img.shape[1] / piece_img.shape[0]
            if aspect_ratio > 1:
                new_width = piece_size
                new_height = int(piece_size / aspect_ratio)
            else:
                new_height = piece_size
                new_width = int(piece_size * aspect_ratio)
            
            resized_piece = cv2.resize(piece_img, (new_width, new_height))
            
            # Calculate position
            y_start = row * (piece_size + 20) + (piece_size - new_height) // 2
            x_start = col * (piece_size + 20) + (piece_size - new_width) // 2
            
            # Place piece on canvas
            canvas[y_start:y_start + new_height, 
                   x_start:x_start + new_width] = resized_piece
            
            # Add piece number
            cv2.putText(canvas, piece_name.split('_')[1], 
                       (x_start + 5, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return canvas


def save_matching_log(matches: List[Dict[str, Any]], groups: List[List[str]], 
                     output_path: str) -> None:
    """
    Save detailed matching information to a JSON file.
    
    Args:
        matches: List of all matches
        groups: List of formed groups
        output_path: Path to save the JSON file
    """
    log_data = {
        'total_matches': len(matches),
        'total_groups': len(groups),
        'matches': matches[:50],  # Save top 50 matches
        'groups': [
            {
                'group_id': i + 1,
                'pieces': group,
                'size': len(group)
            }
            for i, group in enumerate(groups)
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def save_summary(pieces_count: int, processed_count: int, matches_count: int,
                groups: List[List[str]], processing_time: float, output_path: str) -> None:
    """
    Save a text summary of the processing results.
    
    Args:
        pieces_count: Total number of input pieces
        processed_count: Number of successfully processed pieces
        matches_count: Number of matches found
        groups: List of formed groups
        processing_time: Total processing time in seconds
        output_path: Path to save the summary file
    """
    with open(output_path, 'w') as f:
        f.write("=== Jigsaw Puzzle Matching Results ===\n\n")
        f.write(f"Input pieces: {pieces_count}\n")
        f.write(f"Successfully processed: {processed_count}\n")
        f.write(f"Total matches found: {matches_count}\n")
        f.write(f"Groups formed: {len(groups)}\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n\n")
        
        if groups:
            f.write("=== Group Details ===\n\n")
            for i, group in enumerate(groups):
                f.write(f"Group {i + 1}: {len(group)} pieces\n")
                f.write(f"Pieces: {', '.join(group)}\n\n")
            
            # Statistics
            group_sizes = [len(group) for group in groups]
            f.write("=== Statistics ===\n\n")
            f.write(f"Largest group: {max(group_sizes)} pieces\n")
            f.write(f"Average group size: {np.mean(group_sizes):.1f} pieces\n")
            f.write(f"Total pieces in groups: {sum(group_sizes)}\n")
        else:
            f.write("No groups were formed.\n")


def save_results(groups: List[List[str]], pieces: Dict[str, np.ndarray],
                features: Dict[str, Dict[str, Any]], matches: List[Dict[str, Any]],
                processing_time: float, output_dir: str) -> None:
    """
    Save all results including visualization, log, and summary.
    
    Args:
        groups: List of formed groups
        pieces: Dictionary of original piece images
        features: Dictionary of piece features
        matches: List of all matches
        processing_time: Total processing time
        output_dir: Directory to save results
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    visualize_groups(groups, pieces, features, matches, 
                    os.path.join(output_dir, 'groups.png'))
    
    # Save matching log
    save_matching_log(matches, groups, 
                     os.path.join(output_dir, 'matching_log.json'))
    
    # Save summary
    pieces_count = len(pieces)
    processed_count = len(features)
    matches_count = len(matches)
    
    save_summary(pieces_count, processed_count, matches_count, groups,
                processing_time, os.path.join(output_dir, 'summary.txt'))