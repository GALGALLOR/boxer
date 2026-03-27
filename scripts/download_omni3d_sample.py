#!/usr/bin/env python3
"""
Create a small sample of the Omni3D SUN-RGBD dataset for testing Boxer.

Downloads the Omni3D annotation JSON (publicly hosted by Meta) and copies
a subset of images from your local SUN-RGBD data into sample_data/.

Prerequisites:
    Download SUNRGBD V1 from https://rgbd.cs.princeton.edu/ and extract it
    so that images are at sample_data/Omni3D/SUNRGBD/kv2/.../image/*.jpg

Usage:
    # Create 20-image sample from default location
    python scripts/download_omni3d_sample.py

    # Custom source and count
    python scripts/download_omni3d_sample.py --data_root /path/to/Omni3D --num_images 10
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

OMNI3D_JSON_URL = "https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "sample_data", "Omni3D")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "sample_data", "Omni3D")


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create a sample Omni3D SUN-RGBD subset for Boxer."
    )
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help=f"Path to full Omni3D data (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="Number of sample images to include (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection (default: 42)",
    )
    args = parser.parse_args()

    # Step 1: Get the Omni3D JSON (download or use existing)
    json_path = os.path.join(args.data_root, "SUNRGBD_val.json")

    if not os.path.exists(json_path):
        print("SUNRGBD_val.json not found locally, downloading Omni3D JSON...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "Omni3D_json.zip")
            try:
                download_file(OMNI3D_JSON_URL, zip_path)
            except Exception as e:
                print(f"\nFailed to download Omni3D JSON: {e}")
                sys.exit(1)

            with zipfile.ZipFile(zip_path, "r") as zf:
                # Extract just SUNRGBD_val.json
                target = None
                for name in zf.namelist():
                    if name.endswith("SUNRGBD_val.json"):
                        target = name
                        break
                if target is None:
                    print("SUNRGBD_val.json not found in Omni3D_json.zip")
                    sys.exit(1)

                # Extract to output dir
                os.makedirs(args.output_dir, exist_ok=True)
                json_path = os.path.join(args.output_dir, "SUNRGBD_val.json")
                with zf.open(target) as src, open(json_path, "wb") as dst:
                    dst.write(src.read())
                print(f"Extracted SUNRGBD_val.json to {json_path}")

    # Step 2: Load JSON and select images
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    all_images = data["images"]
    all_annotations = data["annotations"]
    categories = data["categories"]

    print(f"Full dataset: {len(all_images)} images, {len(all_annotations)} annotations")

    # Select images deterministically
    import random
    random.seed(args.seed)
    selected = random.sample(all_images, min(args.num_images, len(all_images)))
    selected_ids = {img["id"] for img in selected}

    # Filter annotations to selected images
    selected_anns = [a for a in all_annotations if a["image_id"] in selected_ids]

    print(f"Selected {len(selected)} images with {len(selected_anns)} annotations")

    # Step 3: Check source data exists
    sunrgbd_root = os.path.join(args.data_root, "SUNRGBD")
    if not os.path.isdir(sunrgbd_root):
        print(f"\nError: SUNRGBD data not found at {sunrgbd_root}")
        print("Please download SUNRGBD V1 from https://rgbd.cs.princeton.edu/")
        print(f"and extract it to {args.data_root}/SUNRGBD/")
        sys.exit(1)

    # Step 4: Copy image, depth, and extrinsics for each selected image
    os.makedirs(args.output_dir, exist_ok=True)
    copied = 0
    skipped = 0

    for img in selected:
        file_path = img["file_path"]  # e.g. SUNRGBD/kv2/.../image/0000269.jpg

        # Directories to copy: image, depth, extrinsics
        image_dir = os.path.dirname(file_path)  # .../image
        scene_dir = os.path.dirname(image_dir)  # .../scene_folder

        # Source and destination scene dirs
        src_scene = os.path.join(args.data_root, scene_dir)
        dst_scene = os.path.join(args.output_dir, scene_dir)

        if not os.path.isdir(src_scene):
            print(f"  Warning: scene dir not found: {src_scene}")
            skipped += 1
            continue

        # Copy image/, depth/, extrinsics/ subdirectories
        for subdir in ["image", "depth", "extrinsics"]:
            src_sub = os.path.join(src_scene, subdir)
            dst_sub = os.path.join(dst_scene, subdir)
            if os.path.isdir(src_sub):
                os.makedirs(dst_sub, exist_ok=True)
                for fname in os.listdir(src_sub):
                    src_file = os.path.join(src_sub, fname)
                    dst_file = os.path.join(dst_sub, fname)
                    if os.path.isfile(src_file) and not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

        copied += 1

    print(f"Copied {copied} scenes ({skipped} skipped)")

    # Step 5: Write trimmed JSON
    trimmed = {
        "info": data.get("info", {}),
        "images": selected,
        "annotations": selected_anns,
        "categories": categories,
    }
    out_json = os.path.join(args.output_dir, "SUNRGBD_val.json")
    with open(out_json, "w") as f:
        json.dump(trimmed, f)
    print(f"Wrote trimmed JSON: {out_json} ({len(selected)} images)")

    print(f"\nDone! Sample data at: {args.output_dir}")
    print(f"\nRun Boxer on it:")
    print(f"  python run_boxer.py --input SUNRGBD --max_n 5")


if __name__ == "__main__":
    main()
