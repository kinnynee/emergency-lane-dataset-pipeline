import csv
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

root = Path('data_collection')
meta = root / 'reports' / 'external_eda' / 'sequence_scene_metadata.csv'
rows = list(csv.DictReader(open(meta, encoding='utf-8-sig', newline='')))

pairs = Counter((r.get('road_type', 'UNKNOWN'), r.get('lighting', 'UNKNOWN')) for r in rows)
road_types = sorted({k[0] for k in pairs.keys()})
lightings = sorted({k[1] for k in pairs.keys()})

matrix = pd.DataFrame(index=road_types, columns=lightings, data=0, dtype=float)
for (rt, lt), v in pairs.items():
    if rt in matrix.index and lt in matrix.columns:
        matrix.loc[rt, lt] = v

plt.figure(figsize=(10, 5))
sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap: road type vs lighting')
plt.ylabel('Road type')
plt.xlabel('Lighting')
plt.tight_layout()
out = root / 'docs' / 'road_type_lighting_heatmap.png'
plt.savefig(out, dpi=200)
print(out)
