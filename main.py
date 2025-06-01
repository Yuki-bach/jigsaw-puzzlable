#!/usr/bin/env python3
"""
Jigsaw Puzzlable - Main execution script
Finds matching puzzle pieces from a collection of white jigsaw pieces.
"""

import time
import sys
from modules.preprocessing import load_pieces, preprocess_all_pieces
from modules.feature_extraction import extract_features
from modules.matching import find_matching_pairs, form_groups
from modules.visualization import save_results


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
    
    # Start with a lower threshold for MVP
    threshold = 0.75
    matches = find_matching_pairs(features, threshold=threshold)
    print(f"Found {len(matches)} potential matches with threshold {threshold}")
    
    # If too few matches, try lowering threshold
    if len(matches) < 5:
        print("Few matches found, trying lower threshold...")
        threshold = 0.7
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
    
    print(f"\nProcessing complete in {processing_time:.2f} seconds!")
    print("Results saved to:")
    print("  - results/groups.png (visualization)")
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