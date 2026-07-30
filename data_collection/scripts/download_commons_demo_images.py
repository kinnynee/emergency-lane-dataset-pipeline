"""Tìm và tải ảnh demo từ Wikimedia Commons với metadata giấy phép từng file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from online_common import ONLINE_ROOT, ROOT, safe_name

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "UMT-K230-Dataset-Research/1.0 (educational dataset metadata)"
QUERIES = {
    "day": "traffic cars",
    "night": "night traffic cars road",
    "rain": "traffic cars road rain",
    "backlight": "sunset traffic cars",
    "negative": "empty highway road",
}
ALLOWED_LICENSE_PREFIXES = ("CC0", "CC BY", "CC BY-SA", "Public domain")
FIELDS = [
    "image_id", "condition", "file_name", "commons_title", "source_page_url",
    "original_url", "download_url", "creator", "license_name", "license_url",
    "license_verified", "checksum_sha256", "width", "height", "storage_path",
    "content_review", "privacy_review", "notes",
]


def clean_html(value: str) -> str:
    """Loại tag HTML đơn giản khỏi extmetadata."""
    import re

    return re.sub(r"<[^>]+>", "", value or "").replace("\n", " ").strip()


def api_search(query: str, limit: int, width: int) -> list[dict[str, Any]]:
    """Tìm file ảnh bằng API chính thức và trả imageinfo kèm license."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": "6",
        "gsrlimit": str(min(max(limit, 10), 50)),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(width),
        "origin": "*",
    }
    request = urllib.request.Request(
        API_URL + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return list((payload.get("query", {}).get("pages") or {}).values())


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    """Đọc extmetadata an toàn."""
    return clean_html(str((metadata.get(key) or {}).get("value") or ""))


def sha256(path: Path) -> str:
    """Tính checksum file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, max_mb: float) -> str:
    """Tải một ảnh không ghi đè, giới hạn dung lượng."""
    if target.exists():
        return sha256(target)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as handle:
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > max_mb * 1024 * 1024:
                        raise RuntimeError(f"Ảnh vượt giới hạn {max_mb} MB")
                    handle.write(block)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            wait_seconds = 2 ** (attempt + 1)
            print(f"Wikimedia giới hạn tốc độ; chờ {wait_seconds}s rồi thử lại.")
            time.sleep(wait_seconds)
    partial.replace(target)
    return sha256(target)


def main() -> int:
    """Tải khoảng N ảnh mỗi điều kiện và ghi metadata UTF-8."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-condition", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-file-size-mb", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--condition", choices=sorted(QUERIES))
    args = parser.parse_args()
    conditions = {args.condition: QUERIES[args.condition]} if args.condition else QUERIES
    metadata_path = ROOT / "planning/online_image_metadata.csv"
    existing: list[dict[str, str]] = []
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    known_pages = {row.get("source_page_url", "") for row in existing}
    new_rows: list[dict[str, str]] = []
    try:
        for condition, query in conditions.items():
            accepted = sum(row.get("condition") == condition for row in existing)
            pages = api_search(query, max(args.per_condition * 4, 20), args.width)
            for page in pages:
                if accepted >= args.per_condition:
                    break
                info = (page.get("imageinfo") or [{}])[0]
                metadata = info.get("extmetadata") or {}
                license_name = metadata_value(metadata, "LicenseShortName")
                source_page = str(info.get("descriptionurl") or "")
                mime = str(info.get("mime") or "")
                if not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
                    continue
                if source_page in known_pages or mime not in {"image/jpeg", "image/png", "image/webp"}:
                    continue
                extension = ".jpg" if mime == "image/jpeg" else ".png" if mime == "image/png" else ".webp"
                image_id = f"WM_{condition.upper()}_{accepted + 1:03d}"
                file_name = f"{image_id.lower()}_{safe_name(str(page.get('title') or 'image'))[:70]}{extension}"
                relative = Path("storage_placeholders/online_data/frames") / condition / file_name
                target = ROOT / relative
                download_url = str(info.get("thumburl") or info.get("url") or "")
                row = {
                    "image_id": image_id,
                    "condition": condition,
                    "file_name": file_name,
                    "commons_title": str(page.get("title") or ""),
                    "source_page_url": source_page,
                    "original_url": str(info.get("url") or ""),
                    "download_url": download_url,
                    "creator": metadata_value(metadata, "Artist"),
                    "license_name": license_name,
                    "license_url": metadata_value(metadata, "LicenseUrl"),
                    "license_verified": "TRUE",
                    "checksum_sha256": "DRY_RUN" if args.dry_run else "",
                    "width": str(info.get("thumbwidth") or info.get("width") or ""),
                    "height": str(info.get("thumbheight") or info.get("height") or ""),
                    "storage_path": relative.as_posix(),
                    "content_review": "NEEDS_MANUAL_REVIEW",
                    "privacy_review": "NEEDS_MANUAL_REVIEW",
                    "notes": "Ảnh bổ sung; condition là truy vấn tìm kiếm, chưa phải ground truth.",
                }
                print(f"[{'DRY-RUN' if args.dry_run else 'DOWNLOAD'}] {condition}: {source_page}")
                if not args.dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        row["checksum_sha256"] = download(download_url, target, args.max_file_size_mb)
                    except (OSError, RuntimeError) as exc:
                        print(f"BỎ QUA {source_page}: {exc}", file=sys.stderr)
                        continue
                new_rows.append(row)
                known_pages.add(source_page)
                accepted += 1
                if not args.dry_run:
                    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                        writer.writerows([*existing, *new_rows])
                time.sleep(1.0)
        if not args.dry_run and new_rows:
            with metadata_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows([*existing, *new_rows])
        print(f"Tìm được {len(new_rows)} ảnh mới; dry_run={args.dry_run}.")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
