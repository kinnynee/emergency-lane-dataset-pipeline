import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SPLITS_CSV = Path(r"D:/UMT_EVIDENCE/dataset-v1-full/metadata/sequence_splits.csv")
META_CSV = Path(r"D:/UMT_EVIDENCE/dataset-v1-full/metadata/sequence_scene_metadata.csv")
DEFAULT_OUTPUT_IMAGE = Path(r"d:/Jack_Projects/Dataset traffic/emergency-lane-dataset-pipeline/tmp/external_4splits_heatmap.png")

def main(output_image: Path):
    splits = list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8-sig")))
    meta = list(csv.DictReader(open(META_CSV, encoding="utf-8-sig")))
    meta_dict = {(r["dataset_name"], r["sequence_id"]): r for r in meta}

    split_counts = {
        'train': Counter(),
        'val': Counter(),
        'test': Counter(),
        'overall': Counter()
    }

    road_types = ["HIGHWAY", "INTERSECTION", "URBAN_ROAD"]
    lightings = ["DAY", "TWILIGHT", "NIGHT"]

    for s in splits:
        ds = s.get("dataset", "")
        seq = s.get("sequence_id", "")
        sp = s.get("split", "")
        if sp in ("test", "cross_test"):
            sp = "test"
        m = meta_dict.get((ds, seq), {})
        road = m.get("road_type", "UNKNOWN")
        lighting = m.get("lighting", "UNKNOWN")

        if sp not in split_counts:
            continue
        split_counts[sp][(road, lighting)] += 1
        split_counts['overall'][(road, lighting)] += 1

    sequence_totals = {key: sum(counts.values()) for key, counts in split_counts.items()}
    overall_total = sequence_totals['overall']

    # Create 2x2 Subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phân Bố Heatmap Trực Quan Hóa Dữ Liệu Train / Val / Test & Overall Summary', fontsize=16, fontweight='bold', y=0.98)

    cmaps = {
        'train': 'Blues',
        'val': 'Oranges',
        'test': 'Reds',
        'overall': 'Purples'
    }

    titles = {
        'train': f"1. Tập Train ({sequence_totals['train']} Sequences / {sequence_totals['train'] * 100 / overall_total:.1f}%)",
        'val': f"2. Tập Validation ({sequence_totals['val']} Sequences / {sequence_totals['val'] * 100 / overall_total:.1f}%)",
        'test': f"3. Tập Test ({sequence_totals['test']} Sequences / {sequence_totals['test'] * 100 / overall_total:.1f}%)",
        'overall': f"4. Overall Summary ({overall_total} Sequences / 100%)"
    }

    split_keys = ['train', 'val', 'test', 'overall']

    for idx, key in enumerate(split_keys):
        ax = axes[idx // 2, idx % 2]
        counts = split_counts[key]

        matrix = pd.DataFrame(0, index=road_types, columns=lightings, dtype=int)
        for (r, c), cnt in counts.items():
            if r in matrix.index and c in matrix.columns:
                matrix.at[r, c] = cnt

        values = matrix.values
        im = ax.imshow(values, cmap=cmaps[key], aspect='auto')

        ax.set_title(titles[key], fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(len(lightings)))
        ax.set_xticklabels(lightings, fontsize=10, fontweight='bold')
        ax.set_yticks(range(len(road_types)))
        ax.set_yticklabels(road_types, fontsize=10, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count', rotation=270, labelpad=12, fontsize=9)

        # Annotations
        norm_max = values.max() if values.max() > 0 else 1
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iat[i, j]
                color = "white" if im.norm(val) > 0.55 else "black"
                ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=200)
    print(f"Saved 4-split heatmap image to: {output_image}")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a split-aware road-type and lighting heatmap.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_IMAGE)
    args = parser.parse_args()
    main(args.output)
