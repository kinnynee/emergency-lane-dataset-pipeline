"""Tự động tìm đúng ba dataset được phép dùng cho EDA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from external_eda_common import ROOT

EXCLUDED_NAMES = ("radiate", "classification")


def _candidate_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    results: list[Path] = []
    if not root.exists():
        return results
    lowered = tuple(pattern.lower() for pattern in patterns)
    for current, dirs, files in __import__("os").walk(root):
        dirs[:] = [name for name in dirs if not any(blocked in name.lower() for blocked in EXCLUDED_NAMES)]
        base = Path(current)
        for name in files:
            path = base / name
            value = str(path).lower()
            if any(pattern in value for pattern in lowered):
                results.append(path)
        if any(pattern in str(base).lower() for pattern in lowered):
            results.append(base)
    return results


def _prefer(paths: list[Path], suffixes: tuple[str, ...], include: str = "") -> Path | None:
    unique = sorted(set(path.resolve() for path in paths if path.exists()), key=lambda path: (len(path.parts), str(path)))
    eligible = [
        path
        for path in unique
        if (not include or include.lower() in str(path).lower())
        and (path.is_dir() or path.suffix.lower() in suffixes)
    ]
    if not eligible:
        return None
    files = [path for path in eligible if path.is_file()]
    return files[0] if files else eligible[0]


def discover(
    search_root: Path | None = None,
    mio_path: str | None = None,
    aau_path: str | None = None,
    uadetrac_path: str | None = None,
) -> dict[str, Path | None]:
    root = (search_root or ROOT).resolve()
    raw = root / "storage_placeholders" / "online_data" / "raw"
    explicit = {
        "mio_tcd": Path(mio_path).resolve() if mio_path else None,
        "aau_rainsnow": Path(aau_path).resolve() if aau_path else None,
        "ua_detrac": Path(uadetrac_path).resolve() if uadetrac_path else None,
    }
    if explicit["mio_tcd"] and "classification" in str(explicit["mio_tcd"]).lower():
        raise ValueError("MIO-TCD Classification bị cấm; hãy dùng Localization.")
    found = dict(explicit)
    if not found["mio_tcd"]:
        known = raw / "mio_tcd" / "MIO-TCD-Localization.tar"
        found["mio_tcd"] = known if known.exists() else _prefer(
            _candidate_files(root, ("mio-tcd-localization", "mio_tcd", "localization")),
            (".tar", ".zip"), "localization",
        )
    if not found["aau_rainsnow"]:
        known = raw / "aau_rainsnow" / "aau-rainsnow"
        if (known / "aauRainSnow-rgb.json").exists():
            found["aau_rainsnow"] = known
        else:
            candidates = _candidate_files(root, ("aau-rainsnow", "aau_rainsnow", "rainsnow"))
            extracted = [path for path in candidates if path.is_dir() and (path / "aauRainSnow-rgb.json").exists()]
            found["aau_rainsnow"] = sorted(extracted, key=lambda path: len(path.parts))[0] if extracted else _prefer(candidates, (".zip",))
    if not found["ua_detrac"]:
        known = raw / "ua_detrac_orig" / "ua-detrac-orig.zip"
        found["ua_detrac"] = known if known.exists() else _prefer(
            _candidate_files(root, ("ua-detrac-orig", "ua_detrac", "detrac-images")),
            (".zip", ".tar"),
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--mio-path")
    parser.add_argument("--aau-path")
    parser.add_argument("--uadetrac-path")
    args = parser.parse_args()
    paths = discover(Path(args.root), args.mio_path, args.aau_path, args.uadetrac_path)
    for name, path in paths.items():
        print(f"{name}={path if path else 'NOT_FOUND'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
