"""
Visualization module for puzzle piece matching results.
Generates visual outputs and assembly guides.
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Any
from datetime import datetime


def save_results(groups: List[Dict[str, Any]],
                 pieces: Dict[str, np.ndarray],
                 features: Dict[str, Dict[str, Any]],
                 matches: List[Dict[str, Any]],
                 processing_time: float,
                 output_dir: str) -> None:
    """
    Save all results to the output directory.

    Args:
        groups: List of formed groups
        pieces: Original piece images
        features: Extracted features
        matches: All matches found
        processing_time: Total processing time in seconds
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save visualizations
    save_groups_visualization(groups, pieces, output_dir)

    # Save text reports
    save_summary(groups, features, matches, processing_time, output_dir)
    save_connections_report(groups, output_dir)
    save_assembly_guide(groups, output_dir)

    # Save detailed JSON log
    save_matching_log(groups, matches, output_dir)


def save_groups_visualization(groups: List[Dict[str, Any]],
                              pieces: Dict[str, np.ndarray],
                              output_dir: str) -> None:
    """
    Create a visualization of the formed groups.

    Args:
        groups: List of formed groups
        pieces: Original piece images
        output_dir: Output directory
    """
    if not groups:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Determine layout
    num_groups = min(len(groups), 10)  # Show at most 10 groups
    cols = min(num_groups, 5)
    rows = (num_groups + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if num_groups == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, group in enumerate(groups[:num_groups]):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Create a simple visualization showing piece names
        piece_list = group['pieces'][:10]  # Show at most 10 pieces
        text = f"Group {idx + 1}\n"
        text += f"{group['size']} pieces\n"
        text += f"Confidence: {group['avg_confidence']:.1%}\n\n"
        text += "Pieces:\n"
        for piece in piece_list:
            piece_num = piece.replace('piece_', '')
            text += f"  {piece_num}\n"
        if group['size'] > 10:
            text += f"  ... and {group['size'] - 10} more"

        ax.text(0.5, 0.5, text, ha='center', va='center',
                fontsize=10, family='monospace',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Group {idx + 1}", fontsize=12, fontweight='bold')

    # Hide empty subplots
    for idx in range(num_groups, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'groups.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_summary(groups: List[Dict[str, Any]],
                 features: Dict[str, Dict[str, Any]],
                 matches: List[Dict[str, Any]],
                 processing_time: float,
                 output_dir: str) -> None:
    """
    Save a summary report.

    Args:
        groups: Formed groups
        features: Extracted features
        matches: All matches
        processing_time: Processing time
        output_dir: Output directory
    """
    # Calculate statistics
    total_pieces = len(features)
    total_matches = len(matches)
    num_groups = len(groups)

    # Edge type statistics
    type_counts = {'convex': 0, 'concave': 0, 'flat': 0, 'unknown': 0}
    for piece_features in features.values():
        for edge_name in ['top', 'right', 'bottom', 'left']:
            edge = piece_features['edges'][edge_name]
            edge_type = edge.get('type', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1

    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  Jigsaw Puzzlable - マッチング結果サマリー\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"処理時間: {processing_time:.2f}秒\n\n")

        f.write("-" * 40 + "\n")
        f.write("【基本統計】\n")
        f.write("-" * 40 + "\n")
        f.write(f"処理ピース数: {total_pieces}\n")
        f.write(f"検出マッチ数: {total_matches}\n")
        f.write(f"形成グループ数: {num_groups}\n\n")

        f.write("-" * 40 + "\n")
        f.write("【エッジタイプ分布】\n")
        f.write("-" * 40 + "\n")
        f.write(f"凸 (convex):  {type_counts['convex']}\n")
        f.write(f"凹 (concave): {type_counts['concave']}\n")
        f.write(f"平 (flat):    {type_counts['flat']}\n")
        f.write(f"不明:         {type_counts['unknown']}\n\n")

        if groups:
            f.write("-" * 40 + "\n")
            f.write("【グループサマリー】\n")
            f.write("-" * 40 + "\n")

            for i, group in enumerate(groups[:20]):
                f.write(f"グループ{i+1}: {group['size']}ピース, ")
                f.write(f"信頼度 {group['avg_confidence']:.1%} ")
                f.write(f"(最小: {group['min_confidence']:.1%})\n")

            if len(groups) > 20:
                f.write(f"... 他 {len(groups) - 20} グループ\n")


def save_connections_report(groups: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save detailed connection information.

    Args:
        groups: Formed groups
        output_dir: Output directory
    """
    with open(os.path.join(output_dir, 'connections.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  接続詳細レポート\n")
        f.write("=" * 60 + "\n\n")

        for i, group in enumerate(groups):
            f.write(f"【グループ {i+1}】\n")
            f.write(f"ピース数: {group['size']}\n")
            f.write(f"接続数: {group['num_connections']}\n")
            f.write(f"平均信頼度: {group['avg_confidence']:.1%}\n")
            f.write("-" * 40 + "\n")

            f.write("所属ピース: ")
            pieces = group['pieces']
            piece_nums = [p.replace('piece_', '') for p in pieces]
            f.write(", ".join(piece_nums[:20]))
            if len(pieces) > 20:
                f.write(f" ... 他{len(pieces) - 20}個")
            f.write("\n\n")

            f.write("接続一覧:\n")
            # Sort connections by score
            sorted_conns = sorted(group['connections'],
                                  key=lambda x: x['score'], reverse=True)

            for conn in sorted_conns:
                p1 = conn['piece1'].replace('piece_', '')
                p2 = conn['piece2'].replace('piece_', '')
                e1 = translate_edge_jp(conn['edge1'])
                e2 = translate_edge_jp(conn['edge2'])
                score = conn['score']

                f.write(f"  {p1}の{e1} <-> {p2}の{e2}  ")
                f.write(f"(信頼度: {score:.1%})\n")

            f.write("\n")


def save_assembly_guide(groups: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save an assembly guide for manual puzzle solving.

    Args:
        groups: Formed groups
        output_dir: Output directory
    """
    with open(os.path.join(output_dir, 'assembly_guide.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  ジグソーパズル組み立てガイド\n")
        f.write("=" * 60 + "\n\n")

        f.write("このガイドは、高信頼度で接続が確認されたピースの\n")
        f.write("組み合わせを信頼度順に示しています。\n")
        f.write("上から順に組み立てていくことをお勧めします。\n\n")

        for i, group in enumerate(groups):
            f.write("=" * 50 + "\n")
            f.write(f"グループ {i+1} ({group['size']}ピース)\n")
            f.write("=" * 50 + "\n")
            f.write(f"グループ全体の平均信頼度: {group['avg_confidence']:.1%}\n\n")

            # Sort connections by score (highest first)
            sorted_conns = sorted(group['connections'],
                                  key=lambda x: x['score'], reverse=True)

            f.write("【組み立て順序（信頼度順）】\n\n")

            for j, conn in enumerate(sorted_conns):
                p1 = conn['piece1'].replace('piece_', '')
                p2 = conn['piece2'].replace('piece_', '')
                t1 = translate_type_jp(conn.get('edge1_type', 'unknown'))
                t2 = translate_type_jp(conn.get('edge2_type', 'unknown'))
                score = conn['score']

                f.write(f"Step {j+1}. ピース {p1} ({t1})\n")
                f.write(f"         ↔ ピース {p2} ({t2})\n")
                f.write(f"         信頼度: {score:.1%}\n\n")

            f.write("\n")


def save_matching_log(groups: List[Dict[str, Any]],
                      matches: List[Dict[str, Any]],
                      output_dir: str) -> None:
    """
    Save detailed matching data as JSON.

    Args:
        groups: Formed groups
        matches: All matches
        output_dir: Output directory
    """
    # Convert matches to JSON-serializable format
    matches_data = []
    for match in matches:
        matches_data.append({
            'piece1': match['piece1'],
            'edge1': match['edge1'],
            'piece2': match['piece2'],
            'edge2': match['edge2'],
            'score': float(match['score']),
            'edge1_type': match.get('edge1_type', 'unknown'),
            'edge2_type': match.get('edge2_type', 'unknown')
        })

    # Convert groups to JSON-serializable format
    groups_data = []
    for group in groups:
        group_connections = []
        for conn in group['connections']:
            group_connections.append({
                'piece1': conn['piece1'],
                'edge1': conn['edge1'],
                'edge1_type': conn.get('edge1_type', 'unknown'),
                'piece2': conn['piece2'],
                'edge2': conn['edge2'],
                'edge2_type': conn.get('edge2_type', 'unknown'),
                'score': float(conn['score'])
            })

        groups_data.append({
            'pieces': group['pieces'],
            'size': group['size'],
            'num_connections': group['num_connections'],
            'avg_confidence': float(group['avg_confidence']),
            'min_confidence': float(group['min_confidence']),
            'max_confidence': float(group['max_confidence']),
            'connections': group_connections
        })

    data = {
        'timestamp': datetime.now().isoformat(),
        'total_matches': len(matches),
        'num_groups': len(groups),
        'groups': groups_data,
        'all_matches': matches_data
    }

    with open(os.path.join(output_dir, 'matching_log.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate_edge_jp(edge_name: str) -> str:
    """Translate edge name to Japanese."""
    translations = {
        'top': '上辺',
        'right': '右辺',
        'bottom': '下辺',
        'left': '左辺'
    }
    return translations.get(edge_name, edge_name)


def translate_type_jp(edge_type: str) -> str:
    """Translate edge type to Japanese."""
    translations = {
        'convex': '凸',
        'concave': '凹',
        'flat': '平',
        'unknown': '不明'
    }
    return translations.get(edge_type, edge_type)
