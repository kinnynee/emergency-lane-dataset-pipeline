"""Run a compact YOLO11n inference smoke test on evenly sampled dataset images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


def sample_paths(directory: Path, count: int) -> list[Path]:
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise ValueError(f"No images in {directory}")
    count = min(count, len(paths))
    if count == 1:
        return [paths[0]]
    return [paths[round(index * (len(paths) - 1) / (count - 1))] for index in range(count)]


def label_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()
    image_dir = args.dataset / "images" / "cross_test"
    label_dir = args.dataset / "labels" / "cross_test"
    paths = sample_paths(image_dir, args.samples)
    model = YOLO(str(args.model))
    vehicle_ids = [class_id for class_id, name in model.names.items() if str(name).lower() in VEHICLE_NAMES]
    results = model.predict(
        source=[str(path) for path in paths], classes=vehicle_ids, conf=args.confidence,
        imgsz=args.imgsz, device="cpu", verbose=False,
    )
    rendered: list[Image.Image] = []
    records: list[dict[str, object]] = []
    for path, result in zip(paths, results):
        prediction_count = 0 if result.boxes is None else len(result.boxes)
        expected = label_count(label_dir / f"{path.stem}.txt")
        image = Image.fromarray(result.plot()[:, :, ::-1]).convert("RGB")
        caption = f"{path.name} | prediction={prediction_count} | label={expected}"
        canvas = Image.new("RGB", (image.width, image.height + 28), "black")
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((8, image.height + 6), caption, fill="white")
        rendered.append(canvas)
        records.append({"image": path.name, "predicted_vehicle_boxes": prediction_count, "dataset_vehicle_boxes": expected})
    tile_width = max(image.width for image in rendered)
    tile_height = max(image.height for image in rendered)
    columns = 3
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "black")
    for index, image in enumerate(rendered):
        sheet.paste(image, ((index % columns) * tile_width, (index // columns) * tile_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"saved={args.output}")
    print(f"predicted_boxes={sum(int(row['predicted_vehicle_boxes']) for row in records)}")
    print(f"dataset_boxes={sum(int(row['dataset_vehicle_boxes']) for row in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
