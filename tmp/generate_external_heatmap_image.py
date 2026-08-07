import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METADATA = Path(r"D:/UMT_EVIDENCE/dataset-v1-full/metadata/sequence_scene_metadata.csv")
OUTPUT_IMAGE = Path(__file__).resolve().parent / "external_heatmap.png"


def load_rows(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def build_matrix(rows):
    counts = Counter()
    road_types = set()
    lightings = set()
    for row in rows:
        road = row.get('road_type', 'UNKNOWN') or 'UNKNOWN'
        lighting = row.get('lighting', 'UNKNOWN') or 'UNKNOWN'
        road_types.add(road)
        lightings.add(lighting)
        counts[(road, lighting)] += 1

    road_types = sorted(road_types)
    lightings = sorted(lightings)
    matrix = pd.DataFrame(0, index=road_types, columns=lightings, dtype=int)
    for (road, lighting), count in counts.items():
        matrix.at[road, lighting] = count
    return matrix


def main():
    rows = load_rows(DEFAULT_METADATA)
    if not rows:
        raise RuntimeError(f"No rows loaded from {DEFAULT_METADATA}")

    matrix = build_matrix(rows)
    plt.figure(figsize=(max(8, len(matrix.columns) * 1.2), max(6, len(matrix.index) * 0.7)))
    im = plt.imshow(matrix.values, cmap='YlGnBu', aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Heatmap: road_type vs lighting')
    plt.ylabel('road_type')
    plt.xlabel('lighting')
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha='right')
    plt.yticks(range(len(matrix.index)), matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            plt.text(j, i, str(value), ha='center', va='center', color='black', fontsize=9)

    plt.tight_layout()
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=200)
    plt.close()
    print(f"Generated heatmap image: {OUTPUT_IMAGE}")


if __name__ == '__main__':
    main()
