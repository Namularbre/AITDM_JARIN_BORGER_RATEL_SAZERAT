"""Download the Renault AutoVI dataset (industrial anomaly detection) into data/raw/autovi/.

AutoVI = Automotive Visual Inspection dataset (Renault Group + UTC), public on Zenodo:
    https://auto-vi.utc.fr/            (project page)
    https://zenodo.org/records/10459003 (v1.0.0, DOI 10.5281/zenodo.10459003, ~4 GB)
Licence: CC BY-NC-SA 4.0 (non commercial). No registration needed.
Evaluation code (AUC-sPRO / image AUC-ROC): https://github.com/phcarval/autovi_evaluation_code

Works on Windows and Linux/WSL, standard library only. Run from the repository root:

    python data_splitting/get_autovi.py --list
    python data_splitting/get_autovi.py pipe_staple pipe_clip     # the small ones first
    python data_splitting/get_autovi.py --all                     # the 6 objects, ~4 GB

Each zip is checked against its Zenodo MD5, extracted to data/raw/autovi/<object>/,
then deleted (use --keep-zip to keep it). Existing objects are skipped (--force to redo).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "autovi"
RECORD = "https://zenodo.org/records/10459003/files/{name}?download=1"

# object -> (size, md5) from the Zenodo record v1.0.0
OBJECTS = {
    "pipe_staple": ("106 MB", "d2d165588a0a6d162e640a0578faa424"),
    "pipe_clip": ("144 MB", "3de22ee75e9276d7dc6d0a32d7d7e4b6"),
    "engine_wiring": ("249 MB", "3f37aac704c7ec44e2e5acabbe1cf8d1"),
    "underbody_pipes": ("950 MB", "a2fda55651bb86c2c618b5135ea5cbf0"),
    "tank_screw": ("1.2 GB", "c9a2ae21855f38c5905bef7837c054ac"),
    "underbody_screw": ("1.3 GB", "3a3c53cf2d3da1b7eb8ade82c28c3afb"),
}
EXTRA_FILES = ["readme.md", "license.txt", "changelog.txt"]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB "
                      f"({100 * done / total:5.1f} %)", end="", flush=True)
    print()
    tmp.replace(dest)


def get_object(name: str, keep_zip: bool, force: bool) -> None:
    target = OUT_DIR / name
    if target.exists() and any(target.iterdir()) and not force:
        print(f"{name}: already in {target} (use --force to download again)")
        return
    _, expected_md5 = OBJECTS[name]
    zip_path = OUT_DIR / f"{name}.zip"
    if not (zip_path.exists() and md5sum(zip_path) == expected_md5):
        print(f"{name}: downloading {OBJECTS[name][0]} ...")
        download(RECORD.format(name=f"{name}.zip"), zip_path)
        if md5sum(zip_path) != expected_md5:
            sys.exit(f"ERROR: MD5 mismatch for {zip_path}, delete it and retry")
    print(f"{name}: extracting ...")
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(OUT_DIR)
    if not target.exists():
        print(f"  WARNING: the zip did not create {target}; check the content of {OUT_DIR}")
    if not keep_zip:
        zip_path.unlink()
    print(f"{name}: ok -> {target}")


def show_list() -> None:
    print(f"AutoVI objects (Zenodo record 10459003), installed into {OUT_DIR}:")
    for name, (size, _) in OBJECTS.items():
        target = OUT_DIR / name
        state = "installed" if target.exists() and any(target.iterdir()) else ""
        print(f"  {name:<16} {size:>7}  {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("objects", nargs="*", metavar="OBJECT",
                        help="objects to download: " + ", ".join(OBJECTS))
    parser.add_argument("--all", action="store_true", help="download the 6 objects (~4 GB)")
    parser.add_argument("--list", action="store_true", help="show objects and what is installed")
    parser.add_argument("--keep-zip", action="store_true", help="keep the zip files")
    parser.add_argument("--force", action="store_true", help="re-download installed objects")
    args = parser.parse_args()

    if args.list:
        show_list()
        return
    names = list(OBJECTS) if args.all else args.objects
    unknown = [n for n in names if n not in OBJECTS]
    if unknown:
        parser.error(f"unknown object(s) {unknown}; choose from {', '.join(OBJECTS)}")
    if not names:
        parser.print_help()
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extra in EXTRA_FILES:
        if not (OUT_DIR / extra).exists():
            download(RECORD.format(name=extra), OUT_DIR / extra)
    for name in names:
        get_object(name, args.keep_zip, args.force)
    print(f"\nDone. Dataset licence: {OUT_DIR / 'license.txt'} (CC BY-NC-SA 4.0)")


if __name__ == "__main__":
    main()
