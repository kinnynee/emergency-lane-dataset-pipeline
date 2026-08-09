import csv
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent.parent
meta = root / 'data_collection' / 'reports' / 'external_eda' / 'sequence_scene_metadata.csv'
rows = list(csv.DictReader(open(meta, encoding='utf-8-sig', newline='')))
counts = Counter((r.get('road_type', 'UNKNOWN'), r.get('lighting', 'UNKNOWN')) for r in rows)
road_types = sorted({k[0] for k in counts.keys()})
lightings = sorted({k[1] for k in counts.keys()})

data = [
    {'roadType': rt, 'lighting': lt, 'value': counts[(rt, lt)]}
    for rt in road_types
    for lt in lightings
]

out = root / 'data_collection' / 'docs' / 'heatmap_d3.html'
html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Heatmap D3</title>
  <script src=\"https://d3js.org/d3.v7.min.js\"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h2 {{ margin-bottom: 10px; }}
    svg {{ background: white; }}
    .cell {{ stroke: #fff; stroke-width: 1px; }}
    .label {{ font-size: 12px; }}
    .value {{ font-size: 11px; fill: black; pointer-events: none; }}
  </style>
</head>
<body>
  <h2>Heatmap: road type vs lighting</h2>
  <div id=\"chart\"></div>
  <script>
    const data = {json.dumps(data)};
    const roadTypes = {json.dumps(road_types)};
    const lightings = {json.dumps(lightings)};

    const margin = {{top: 30, right: 20, bottom: 60, left: 120}};
    const width = 700 - margin.left - margin.right;
    const height = 360 - margin.top - margin.bottom;

    const svg = d3.select('#chart')
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    const x = d3.scaleBand().domain(lightings).range([0, width]).padding(0.05);
    const y = d3.scaleBand().domain(roadTypes).range([0, height]).padding(0.05);
    const color = d3.scaleSequential(d3.interpolateOrRd).domain([0, d3.max(data, d => d.value) || 1]);

    svg.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => x(d.lighting))
      .attr('y', d => y(d.roadType))
      .attr('width', x.bandwidth())
      .attr('height', y.bandwidth())
      .attr('fill', d => color(d.value))
      .append('title').text(d => d.roadType + ' / ' + d.lighting + ': ' + d.value);

    svg.selectAll('.value')
      .data(data)
      .enter().append('text')
      .attr('class', 'value')
      .text(d => d.value)
      .attr('x', d => x(d.lighting) + x.bandwidth() / 2)
      .attr('y', d => y(d.roadType) + y.bandwidth() / 2 + 4)
      .attr('text-anchor', 'middle');

    svg.append('g')
      .selectAll('.xLabel')
      .data(lightings)
      .enter().append('text')
      .attr('class', 'label xLabel')
      .text(d => d)
      .attr('x', d => x(d) + x.bandwidth() / 2)
      .attr('y', height + 20)
      .attr('text-anchor', 'middle');

    svg.append('g')
      .selectAll('.yLabel')
      .data(roadTypes)
      .enter().append('text')
      .attr('class', 'label yLabel')
      .text(d => d)
      .attr('x', -6)
      .attr('y', d => y(d) + y.bandwidth() / 2 + 4)
      .attr('text-anchor', 'end');
  </script>
</body>
</html>"""

out.write_text(html, encoding='utf-8')
print(out)
