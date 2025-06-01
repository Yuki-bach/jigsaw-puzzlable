#!/usr/bin/env python3
"""
Jigsaw Puzzlable - Main execution script
Finds matching puzzle pieces from a collection of white jigsaw pieces.
"""

import time
import sys
import os
from datetime import datetime
from modules.preprocessing import load_pieces, preprocess_all_pieces
from modules.feature_extraction import extract_features
from modules.matching import find_matching_pairs, form_groups
from modules.visualization import save_results


def save_threshold_experiment(threshold, num_matches, groups, processing_time):
    """Save threshold experiment results to a file."""
    experiment_file = "threshold_experiment_log.txt"
    
    with open(experiment_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"実験日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"閾値: {threshold}\n")
        f.write(f"マッチ数: {num_matches}\n")
        f.write(f"グループ数: {len(groups)}\n")
        f.write(f"処理時間: {processing_time:.2f}秒\n")
        f.write(f"グループ詳細:\n")
        
        for i, group in enumerate(groups):
            f.write(f"  グループ{i+1}: {len(group)}個")
            if len(group) <= 5:
                f.write(f" ({', '.join(group)})\n")
            else:
                f.write(f" ({', '.join(group[:3])} ... 他{len(group)-3}個)\n")


def main():
    """Main execution function."""
    print("=== Jigsaw Puzzlable - Puzzle Piece Matcher ===\n")
    
    start_time = time.time()
    
    # Step 1: Load pieces
    print("Step 1: Loading pieces...")
    pieces = load_pieces("pieces/")
    print(f"Loaded {len(pieces)} pieces")
    
    if len(pieces) == 0:
        print("Error: No pieces found in the pieces/ directory")
        sys.exit(1)
    
    # Step 2: Preprocess pieces
    print("\nStep 2: Preprocessing pieces...")
    processed_pieces = preprocess_all_pieces(pieces)
    print(f"Successfully processed {len(processed_pieces)} pieces")
    
    if len(processed_pieces) == 0:
        print("Error: No pieces could be processed successfully")
        sys.exit(1)
    
    # Step 3: Extract features
    print("\nStep 3: Extracting features...")
    features = extract_features(processed_pieces)
    print(f"Extracted features from {len(features)} pieces")
    
    if len(features) < 2:
        print("Error: Need at least 2 pieces with valid features to find matches")
        sys.exit(1)
    
    # Step 4: Find matching pairs
    print("\nStep 4: Finding matching pairs...")
    print("This may take a few minutes...")
    
    # Start with the highest threshold for the most selective matching
    threshold = 0.99
    matches = find_matching_pairs(features, threshold=threshold)
    print(f"Found {len(matches)} potential matches with threshold {threshold}")
    
    # If too few matches, try slightly lower threshold
    if len(matches) < 50:
        print("Few matches found, trying lower threshold...")
        threshold = 0.985
        matches = find_matching_pairs(features, threshold=threshold)
        print(f"Found {len(matches)} potential matches with threshold {threshold}")
    
    # Step 5: Form groups
    print("\nStep 5: Forming groups...")
    groups = form_groups(matches)
    print(f"Formed {len(groups)} groups")
    
    # Print group summary
    if groups:
        print("\nGroup summary:")
        for i, group in enumerate(groups[:10]):  # Show first 10 groups
            print(f"  Group {i+1}: {len(group)} pieces - {', '.join(group[:5])}")
            if len(group) > 5:
                print(f"            ... and {len(group) - 5} more")
    
    # Step 6: Save results
    print("\nStep 6: Saving results...")
    processing_time = time.time() - start_time
    save_results(groups, pieces, features, matches, processing_time, "results/")
    
    # Save threshold experiment results
    save_threshold_experiment(threshold, len(matches), groups, processing_time)
    
    print(f"\nProcessing complete in {processing_time:.2f} seconds!")
    print("Results saved to:")
    print("  - results/groups.png (visualization)")
    print("  - results/connections.txt (piece connections)")
    print("  - results/matching_log.json (detailed matches)")
    print("  - results/summary.txt (summary)")
    
    # Success criteria check
    if len(groups) >= 5:
        print("\n✓ Success: Found at least 5 groups of matching pieces!")
    else:
        print("\n⚠ Warning: Found fewer than 5 groups. You may need to:")
        print("  - Adjust the matching threshold")
        print("  - Improve preprocessing for better edge detection")
        print("  - Check the quality of input images")


if __name__ == "__main__":
    main()