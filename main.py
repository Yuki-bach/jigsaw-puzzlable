#!/usr/bin/env python3
"""
Jigsaw Puzzlable - Main execution script

Finds matching puzzle pieces from a collection of white jigsaw pieces
by analyzing their edge shapes.
"""

import sys
import time
from pathlib import Path

from modules.preprocessing import load_pieces, preprocess_all_pieces
from modules.corner_detection import detect_corners
from modules.edge_analysis import analyze_edges
from modules.matching import find_matching_pairs, get_edge_statistics
from modules.group_formation import form_groups, get_connection_summary
from modules.visualization import save_results


def main():
    """Main execution function."""
    print("=" * 60)
    print("  Jigsaw Puzzlable - Puzzle Piece Matcher")
    print("=" * 60)
    print()

    start_time = time.time()

    # Configuration
    pieces_dir = "pieces/"
    output_dir = "results/"
    match_threshold = 0.7
    min_score = 0.92  # Very high minimum - only top matches
    score_gap_ratio = 1.005  # Nearly disabled - rely on mutual best match only

    # Step 1: Load pieces
    print("Step 1: Loading pieces...")
    pieces = load_pieces(pieces_dir)
    print(f"  Loaded {len(pieces)} pieces")

    if len(pieces) == 0:
        print("Error: No pieces found in the pieces/ directory")
        sys.exit(1)

    # Step 2: Preprocess pieces (contour extraction)
    print("\nStep 2: Preprocessing pieces (contour extraction)...")
    processed_pieces = preprocess_all_pieces(pieces)
    print(f"  Successfully processed {len(processed_pieces)} pieces")

    if len(processed_pieces) == 0:
        print("Error: No pieces could be processed successfully")
        sys.exit(1)

    # Step 3: Detect corners
    print("\nStep 3: Detecting corners...")
    corners_data = {}
    corner_methods = {'curvature': 0, 'pca': 0, 'rect': 0}

    for piece_name, piece_data in processed_pieces.items():
        contour = piece_data['contour']
        corners, method = detect_corners(contour)
        corners_data[piece_name] = (corners, method)
        corner_methods[method] = corner_methods.get(method, 0) + 1

    print(f"  Corner detection methods used:")
    print(f"    Curvature: {corner_methods['curvature']}")
    print(f"    PCA: {corner_methods['pca']}")
    print(f"    Rectangle: {corner_methods['rect']}")

    # Step 4: Analyze edges
    print("\nStep 4: Analyzing edges...")
    features = {}

    for piece_name, piece_data in processed_pieces.items():
        if piece_name not in corners_data:
            continue

        corners, corner_method = corners_data[piece_name]
        contour = piece_data['contour']

        edges = analyze_edges(contour, corners)

        features[piece_name] = {
            'corners': corners,
            'corner_method': corner_method,
            'edges': edges,
            'area': piece_data['area'],
            'perimeter': piece_data['perimeter'],
            'contour': contour
        }

    print(f"  Extracted features from {len(features)} pieces")

    # Print edge statistics
    edge_stats = get_edge_statistics(features)
    print(f"\n  Edge type distribution:")
    print(f"    Convex:  {edge_stats['type_counts']['convex']}")
    print(f"    Concave: {edge_stats['type_counts']['concave']}")
    print(f"    Flat:    {edge_stats['type_counts']['flat']}")
    print(f"    Unknown: {edge_stats['type_counts']['unknown']}")

    if len(features) < 2:
        print("Error: Need at least 2 pieces with valid features")
        sys.exit(1)

    # Step 5: Find matching pairs
    print("\nStep 5: Finding matching pairs...")
    print(f"  (This may take a few minutes for {len(features)} pieces...)")

    matches = find_matching_pairs(features, threshold=match_threshold)
    print(f"  Found {len(matches)} potential matches")

    if matches:
        top_scores = [m['score'] for m in matches[:10]]
        print(f"  Top 10 match scores: {', '.join(f'{s:.1%}' for s in top_scores)}")

    # Step 6: Form groups (with high-confidence filtering)
    print("\nStep 6: Forming groups with validation...")
    groups = form_groups(matches, min_score=min_score, score_gap_ratio=score_gap_ratio)
    print(f"  Formed {len(groups)} validated groups")

    # Print group summary
    if groups:
        print("\n  Group summary:")
        for i, group in enumerate(groups[:10]):
            print(f"    Group {i+1}: {group['size']} pieces, "
                  f"confidence: {group['avg_confidence']:.1%} "
                  f"(min: {group['min_confidence']:.1%})")

        if len(groups) > 10:
            print(f"    ... and {len(groups) - 10} more groups")

    # Step 7: Save results
    print("\nStep 7: Saving results...")
    processing_time = time.time() - start_time

    save_results(groups, pieces, features, matches, processing_time, output_dir)

    print(f"\nProcessing complete in {processing_time:.2f} seconds!")
    print("\nResults saved to:")
    print(f"  - {output_dir}groups.png (visualization)")
    print(f"  - {output_dir}summary.txt (summary)")
    print(f"  - {output_dir}connections.txt (connection details)")
    print(f"  - {output_dir}assembly_guide.txt (assembly guide)")
    print(f"  - {output_dir}matching_log.json (detailed data)")

    # Final summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    conn_summary = get_connection_summary(groups)
    print(f"  Total pieces processed: {len(features)}")
    print(f"  Total validated groups: {conn_summary['num_groups']}")
    print(f"  Total validated connections: {conn_summary['total_connections']}")

    if conn_summary['num_groups'] > 0:
        print(f"  Average confidence: {conn_summary['avg_score']:.1%}")
        print(f"  Confidence range: {conn_summary['min_score']:.1%} - {conn_summary['max_score']:.1%}")


if __name__ == "__main__":
    main()
