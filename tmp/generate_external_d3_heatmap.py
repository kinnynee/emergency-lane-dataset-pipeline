import csv
import json
from pathlib import Path

DEFAULT_METADATA = Path(r"D:/UMT_EVIDENCE/dataset-v1-full/metadata/sequence_scene_metadata.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "external_heatmap_d3.html"


def load_metadata(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def build_heatmap_rows(rows):
    counts = {}
    road_types = set()
    lightings = set()
    for row in rows:
        road = row.get("road_type", "UNKNOWN") or "UNKNOWN"
        lighting = row.get("lighting", "UNKNOWN") or "UNKNOWN"
        road_types.add(road)
        lightings.add(lighting)
        counts[(road, lighting)] = counts.get((road, lighting), 0) + 1

    road_types = sorted(road_types)
    lightings = sorted(lightings)
    data = [
        {"roadType": rt, "lighting": lt, "value": counts.get((rt, lt), 0)}
        for rt in road_types
        for lt in lightings
    ]
    return data, road_types, lightings


def generate_html(data, road_types, lightings, output_path):
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>External Dataset Road Type vs Lighting Heatmap</title>
  <script src=\"https://d3js.org/d3.v7.min.js\"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8f9fb; color: #1b1c1d; }}
    h1 {{ font-size: 22px; margin-bottom: 12px; }}
    .chart-container {{ overflow-x: auto; }}
    svg {{ background: white; border: 1px solid #ddd; box-shadow: 0 0 12px rgba(0,0,0,0.04); }}
    .cell {{ stroke: #fff; stroke-width: 1px; }}
    .label {{ font-size: 12px; fill: #333; }}
    .value {{ font-size: 11px; fill: #111; pointer-events: none; }}
    .axis text {{ font-size: 12px; fill: #333; }}
    .legend text {{ font-size: 12px; fill: #333; }}
  </style>
</head>
<body>
  <h1>Heatmap: road_type vs lighting</h1>
  <p>Source: <code>{DEFAULT_METADATA}</code></p>
  <div class=\"chart-container\">
    <div id=\"chart\"></div>
  </div>
  <script>
    const data = {json.dumps(data)};
    const roadTypes = {json.dumps(road_types)};
    const lightings = {json.dumps(lightings)};

    const margin = {{top: 40, right: 20, bottom: 80, left: 140}};
    const width = Math.max(680, lightings.length * 110) - margin.left - margin.right;
    const height = Math.max(360, roadTypes.length * 40) - margin.top - margin.bottom;

    const svg = d3.select('#chart')
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    const x = d3.scaleBand().domain(lightings).range([0, width]).padding(0.05);
    const y = d3.scaleBand().domain(roadTypes).range([0, height]).padding(0.05);
    const maxValue = d3.max(data, d => d.value) || 1;
    const color = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxValue]);

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
      .append('title')
      .text(d => d.roadType + ' / ' + d.lighting + ': ' + d.value);

    svg.selectAll('.value')
      .data(data)
      .enter().append('text')
      .attr('class', 'value')
      .text(d => d.value)
      .attr('x', d => x(d.lighting) + x.bandwidth() / 2)
      .attr('y', d => y(d.roadType) + y.bandwidth() / 2 + 4)
      .attr('text-anchor', 'middle');

    svg.append('g')
      .attr('transform', 'translate(0, ' + height + ')')
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(25)')
      .style('text-anchor', 'start');

    svg.append('g')
      .call(d3.axisLeft(y));

    svg.append('text')
      .attr('x', width / 2)
      .attr('y', -18)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', '600')
      .text('Road type vs Lighting Count');
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    print(f"Generated heatmap HTML: {output_path}")


def main() -> None:
    rows = load_metadata(DEFAULT_METADATA)
    data, road_types, lightings = build_heatmap_rows(rows)
    generate_html(data, road_types, lightings, DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
