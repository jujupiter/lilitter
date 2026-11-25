"""Prepare our own litter photos (#15) for review with review_commons.py --own.

Takes the photos in a local folder (they stay out of git) and writes, for each one, a review copy in
data/own_candidates/img/<id>.jpg and a row in data/own_candidates/candidates.csv, in the format of the Commons
candidates. Then `python review_commons.py --own` reviews them, and `--own --export` writes data/own_photos/.

The review copy is turned upright (phones store the rotation in EXIF), scaled down to the width of the Commons
thumbnails and re-encoded with no metadata at all: no GPS, phone model or date. The only place recorded is the one
given with --country (by default none). The original's SHA-256 is kept to recognise a photo already prepared, so the
script is safe to run again after adding photos to the folder.

usage: python prepare_own_photos.py ~/Documents/Projects/lilitter/photos --country Australia
       [--author NAME] [--license "CC BY 4.0"]   recorded per photo, for the credits; empty until chosen
"""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from download_commons import THUMB_WIDTH

OUT = Path("data/own_candidates")
COLUMNS = ["id", "title", "width", "height", "license", "artist", "source", "image_url", "commons_sha1",
           "query_category", "suggested_jcode", "suggested_label", "pool", "country", "sha256", "path"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="folder of photos (not in the repository)")
    parser.add_argument("--country", default="", help="where the photos were taken, e.g. Australia")
    parser.add_argument("--author", default="")
    parser.add_argument("--license", default="")
    args = parser.parse_args()

    csv = OUT / "candidates.csv"
    old = pd.read_csv(csv, dtype=str).fillna("") if csv.exists() else pd.DataFrame(columns=COLUMNS)
    known, ids = set(old.sha256), set(old.id)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXTENSIONS):
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if sha in known:
            continue
        # review_commons.py wants digits: the digits of the file name (a phone's date and time), made unique
        base = re.sub(r"\D", "", f.stem) or "1"
        pid = next(base + (str(k) if k else "") for k in range(1000) if base + (str(k) if k else "") not in ids)
        ids.add(pid)
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            width, height = im.size
            im.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 4))
            path = OUT / "img" / f"{pid}.jpg"
            im.save(path, quality=90)  # no exif= argument: the copy carries no metadata
        rows.append(dict(id=pid, title=f.name, width=width, height=height, license=args.license, artist=args.author,
                         source="own photo", image_url="", commons_sha1="", query_category="own photos",
                         suggested_jcode="", suggested_label="", pool="", country=args.country, sha256=sha,
                         path=str(path)))
    out = pd.concat([old, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    out.to_csv(csv, index=False)
    print(f"{len(rows)} new photos, {len(out)} in total, in {csv}: review them with python review_commons.py --own")


if __name__ == "__main__":
    main()
