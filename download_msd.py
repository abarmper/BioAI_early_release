"""Download and install an MSD (Medical Segmentation Decathlon) dataset.

Usage
-----
    python download_msd.py --dataset spleen
    python download_msd.py --dataset liver --data-root /custom/data/root
    python download_msd.py --dataset spleen --skip-download  # if tar already exists

The script:
  1. Downloads the .tar archive from AWS S3 using wget
  2. Extracts it to a temp directory
  3. Converts dataset.json to the BioAI format
  4. Moves imagesTr/, labelsTr/, imagesTs/ (if present) to data/<name>/raw/
"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# ---------------------------------------------------------------------------
# MSD dataset catalogue
#   key        : BioAI dataset name (matches configs/data/*.yaml and data/ dir)
#   msd_task   : Task folder name inside the .tar archive
#   aws_file   : filename on the GCS bucket
# ---------------------------------------------------------------------------
DATASETS = {
    "spleen": {
        "msd_task": "Task09_Spleen",
        "aws_file": "Task09_Spleen.tar",
        "channel_names": {"0": "CT"},
        "labels_msd_to_bioai": {"0": "background", "1": "spleen"},
    },
    "liver": {
        "msd_task": "Task03_Liver",
        "aws_file": "Task03_Liver.tar",
        "channel_names": {"0": "CT"},
        "labels_msd_to_bioai": {"0": "background", "1": "liver", "2": "cancer"},
    },
    "pancreas": {
        "msd_task": "Task07_Pancreas",
        "aws_file": "Task07_Pancreas.tar",
        "channel_names": {"0": "CT"},
        "labels_msd_to_bioai": {"0": "background", "1": "pancreas", "2": "cancer"},
    },
    "hepaticVessel": {
        "msd_task": "Task08_HepaticVessel",
        "aws_file": "Task08_HepaticVessel.tar",
        "channel_names": {"0": "CT"},
        "labels_msd_to_bioai": {"0": "background", "1": "Vessel", "2": "Tumour"},
    },
    "heart": {
        "msd_task": "Task02_Heart",
        "aws_file": "Task02_Heart.tar",
        "channel_names": {"0": "MRI"},
        "labels_msd_to_bioai": {"0": "background", "1": "left atrium"},
    },
    "hippocampus": {
        "msd_task": "Task04_Hippocampus",
        "aws_file": "Task04_Hippocampus.tar",
        "channel_names": {"0": "MRI"},
        "labels_msd_to_bioai": {"0": "background", "1": "Anterior", "2": "Posterior"},
    },
    "lung": {
        "msd_task": "Task06_Lung",
        "aws_file": "Task06_Lung.tar",
        "channel_names": {"0": "CT"},
        "labels_msd_to_bioai": {"0": "background", "1": "cancer"},
    },
    "colon": {
        "msd_task": "Task10_Colon",
        "aws_file": "Task10_Colon.tar",
        "channel_names": {"0": "CT"},
        "labels_msd_to_bioai": {"0": "background", "1": "colon cancer primaries"},
    },
}

# NOTE: brats2023 is NOT from MSD — it is the 2023 BraTS challenge dataset
# and must be downloaded manually from https://www.synapse.org/brats2023
BRATS_NOTE = (
    "brats2023 is the 2023 BraTS challenge dataset and is NOT available via MSD.\n"
    "Download it manually from https://www.synapse.org/brats2023 and place the\n"
    "extracted contents (imagesTr/, labelsTr/, dataset.json) under:\n"
    "  data/brats2023/raw/"
)

AWS_BASE = "https://msd-for-monai.s3-us-west-2.amazonaws.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    """Download url to dest using wget (resumes partial downloads with -c)."""
    print(f"Downloading {url}")
    cmd = [
        "wget",
        "--continue",          # resume partial downloads
        "--show-progress",     # progress bar
        "--progress=bar:force",
        "--output-document", str(dest),
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nwget failed (exit {result.returncode}).", file=sys.stderr)
        sys.exit(result.returncode)


def _extract(tar_path: Path, extract_dir: Path) -> None:
    print(f"Extracting {tar_path.name} …")
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        for i, member in enumerate(members):
            tf.extract(member, path=extract_dir, filter="data")
            if i % 50 == 0:
                print(f"\r  {i + 1}/{len(members)} files", end="", flush=True)
    print(f"\r  {len(members)}/{len(members)} files — done")


def _build_dataset_json(info: dict, msd_json_path: Path) -> dict:
    """Build the BioAI-format dataset.json from the MSD dataset.json."""
    if msd_json_path.exists():
        with open(msd_json_path) as f:
            msd = json.load(f)
        # MSD labels are already {"0": "background", "1": "organ", ...} — keep as-is
        labels = msd.get("labels", info["labels_msd_to_bioai"])
    else:
        labels = info["labels_msd_to_bioai"]

    return {
        "channel_names": info["channel_names"],
        "labels": labels,
        "file_ending": ".nii.gz",
    }


def _install_dataset(info: dict, task_dir: Path, dest_raw: Path) -> None:
    """Copy MSD task folder contents into the BioAI raw directory."""
    dest_raw.mkdir(parents=True, exist_ok=True)

    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        src = task_dir / sub
        dst = dest_raw / sub
        if not src.exists():
            continue
        if dst.exists():
            print(f"  Removing existing {dst.name}/ …")
            shutil.rmtree(dst)
        print(f"  Moving {sub}/ …")
        shutil.copytree(src, dst)

    msd_json = task_dir / "dataset.json"
    bioai_json = _build_dataset_json(info, msd_json)
    out_json = dest_raw / "dataset.json"
    with open(out_json, "w") as f:
        json.dump(bioai_json, f, indent=2)
    print(f"  Wrote {out_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an MSD dataset from AWS S3 and install it for BioAI."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS) + ["brats2023"],
        help="Dataset name (must match an entry in configs/data/)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Temporary directory for download + extraction (default: <data-root>/tmp_msd)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if the .tar already exists in tmp-dir",
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep the downloaded .tar archive after installation",
    )
    args = parser.parse_args()

    if args.dataset == "brats2023":
        print(BRATS_NOTE)
        return

    info = DATASETS[args.dataset]
    data_root = Path(args.data_root)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else data_root / "tmp_msd"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tar_path = tmp_dir / info["aws_file"]
    url = f"{AWS_BASE}/{info['aws_file']}"

    # 1. Download
    if args.skip_download and tar_path.exists():
        print(f"Skipping download — using existing {tar_path}")
    else:
        _download(url, tar_path)

    # 2. Extract
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    _extract(tar_path, extract_dir)

    # 3. Locate extracted task directory
    task_dir = extract_dir / info["msd_task"]
    if not task_dir.exists():
        candidates = list(extract_dir.rglob(info["msd_task"]))
        if candidates:
            task_dir = candidates[0]
        else:
            raise FileNotFoundError(
                f"Could not find '{info['msd_task']}' inside extracted archive. "
                f"Contents: {list(extract_dir.iterdir())}"
            )

    # 4. Install into data/{dataset}/raw/
    dest_raw = data_root / args.dataset / "raw"
    print(f"Installing to {dest_raw} …")
    _install_dataset(info, task_dir, dest_raw)

    # 5. Cleanup
    shutil.rmtree(extract_dir)
    if not args.keep_tar:
        tar_path.unlink()
        print(f"Removed {tar_path.name}")
    if not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print(f"\nDone. Dataset '{args.dataset}' is ready at {dest_raw}")
    print(f"Next step:  python main2.py command=plan data={args.dataset}")


if __name__ == "__main__":
    main()
