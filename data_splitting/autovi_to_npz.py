"""
@author Thomas SAZERAT

The goal of this file is to convert the autovi dataset to npz format to use it with the rest of the kit

Examples (run from the repository root, with the venv activated):

    python data_splitting/autovi_to_npz.py                          # every downloaded object, 224 px
    python data_splitting/autovi_to_npz.py --size 128
    python data_splitting/autovi_to_npz.py --objects pipe_staple pipe_clip engine_wiring --size 128

Objects that are not in data/raw/autovi/ yet are downloaded first (get_autovi.py).

Output: data/raw/autovi_<size>.npz (or --out) with
    images   (N, size, size, 3) uint8   RGB, centre-cropped to a square, then resized
    labels   (N,) int64                 0 = good, 1 = defect (any type)
    object   (N,) str                   object name, e.g. "pipe_clip" (use it with --group-col object)
    is_test  (N,) bool                  True if the image comes from the official test/ folder
    defect   (N,) str                   folder name: "good", "missing", "unclipped", ...

Then, for example:
    python data_splitting/split_data.py --input data/raw/autovi_224.npz --strategy group \
        --group-col object --num-clients 3 --class-names good,defect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_autovi  # noqa: E402

# Global variables
REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = get_autovi.OUT_DIR  # data/raw/autovi
ALL_OBJECTS = list(get_autovi.OBJECTS)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the AutoVI dataset to one npz file for split_data.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--objects", nargs="+", choices=ALL_OBJECTS, metavar="OBJECT",
                        help="objects to include (default: every object already downloaded): "
                             + ", ".join(ALL_OBJECTS))
    parser.add_argument("--size", type=int, default=224,
                        help="side of the square images in pixels (a multiple of 8, e.g. 64, "
                             "128, 224)")
    parser.add_argument("--out", type=Path,
                        help="output file (default: data/raw/autovi_<size>.npz)")
    args = parser.parse_args()
    if args.size <= 0 or args.size % 8:
        parser.error("--size must be a positive multiple of 8 (the CNN halves it 3 times)")
    if args.objects is None:
        args.objects = [o for o in ALL_OBJECTS if is_downloaded(o)]
        if not args.objects:
            parser.error(f"no AutoVI object in {ROOT}; pass --objects to download some")
    if args.out is None:
        args.out = REPO_ROOT / "data" / "raw" / f"autovi_{args.size}.npz"
    return args


def is_downloaded(obj: str) -> bool:
    return (ROOT / obj / "train").is_dir() and (ROOT / obj / "test").is_dir()


def download_missing(objects: list[str]) -> None:
    """Download (with get_autovi.py) the objects that are not in data/raw/autovi/ yet."""
    missing = [o for o in objects if not is_downloaded(o)]
    if not missing:
        return
    print(f"not downloaded yet: {missing}, downloading them first")
    ROOT.mkdir(parents=True, exist_ok=True)
    for extra in get_autovi.EXTRA_FILES:
        if not (ROOT / extra).exists():
            get_autovi.download(get_autovi.RECORD.format(name=extra), ROOT / extra)
    for obj in missing:
        get_autovi.get_object(obj, keep_zip=False, force=False)


def prepare_image(path: Path, size: int) -> np.ndarray:
    """RGB, crop centered and squared, resize size x size -> uint8 (size, size, 3)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        # center the image for the objects that use 1000*750 images
        img = ImageOps.fit(img, (size, size), Image.BILINEAR, centering=(0.5, 0.5))
        return np.asarray(img, dtype=np.uint8)


def main() -> None:
    args = get_arguments()
    download_missing(args.objects)
    print(f"loading data from {ROOT} with objects {args.objects}, size {args.size}")

    images, labels, objects, is_test, defects = [], [], [], [], []
    # foreach object in the dataset we have downloaded
    for obj in args.objects:
        # the official train (good only) and test (good + defects) folders
        for split in ("train", "test"):
            for class_dir in sorted(p for p in (ROOT / obj / split).iterdir() if p.is_dir()):
                # create the class for the model, 0 = good, the piece is in good shape, 1 otherwise.
                label = 0 if class_dir.name == "good" else 1
                # create the image, then the label and the object related to it
                for png in sorted(class_dir.glob("*.png")):
                    images.append(prepare_image(png, args.size))
                    labels.append(label)
                    objects.append(obj)
                    is_test.append(split == "test")
                    defects.append(class_dir.name)
        print(f"  {obj}: {objects.count(obj)} images")

    labels = np.array(labels, dtype=np.int64)
    print(f"saving {len(labels)} images ({(labels == 0).sum()} good, {(labels == 1).sum()} defect)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        images=np.stack(images),  # (N, size, size, 3) uint8
        labels=labels,  # (N,)
        object=np.array(objects),  # (N,) str
        is_test=np.array(is_test),  # (N,) bool
        defect=np.array(defects),  # (N,) str
    )

    print(f"done, result saved to {args.out}")


if __name__ == '__main__':
    main()
