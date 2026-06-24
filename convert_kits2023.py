"""
Standalone converter: KiTS2023 → BioAI raw format.

KiTS2023 source layout (per case):
    {kits_base_dir}/
        case_00000/
            imaging.nii.gz
            segmentation.nii.gz   (absent for competition test cases)
        case_00001/
            ...

BioAI output layout:
    {output_dir}/
        raw/
            dataset.json
            imagesTr/  case_XXXXX_0000.nii.gz
            labelsTr/  case_XXXXX.nii.gz
            imagesTs/  case_XXXXX_0000.nii.gz   (cases without segmentation)
    configs/data/kits2023.yaml                  (created unless --no-config)

Raw KiTS2023 segmentation values (as stored in segmentation.nii.gz):
    0  background
    1  kidney (non-cancerous kidney parenchyma)
    2  tumor
    3  cyst

dataset.json uses the BioAI/nnUNet integer-key format required by the
preprocessing pipeline ({"0": "background", "1": "kidney", ...}).
Region-based overlapping labels (nnUNet Dataset220_KiTS2023 style) are
NOT used here because BioAI's preprocessor expects simple integer-keyed
labels and does not support region-list values.

Usage:
    python convert_kits2023.py /path/to/kits23/dataset
    python convert_kits2023.py /path/to/kits23/dataset -o data/kits2023
    python convert_kits2023.py /path/to/kits23/dataset --symlink
    python convert_kits2023.py /path/to/kits23/dataset --no-config
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_or_link(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _discover_cases(kits_base: Path) -> tuple[list[Path], list[Path]]:
    """Return (cases_with_seg, cases_without_seg) sorted by case name."""
    all_case_dirs = sorted(
        d for d in kits_base.iterdir()
        if d.is_dir() and d.name.startswith("case_")
    )
    if not all_case_dirs:
        sys.exit(
            f"ERROR: No 'case_XXXXX' subdirectories found in {kits_base}.\n"
            "Make sure you point to the root of the downloaded KiTS2023 dataset."
        )

    with_seg, without_seg = [], []
    for case_dir in all_case_dirs:
        if (case_dir / "segmentation.nii.gz").exists():
            with_seg.append(case_dir)
        else:
            without_seg.append(case_dir)
    return with_seg, without_seg


# ---------------------------------------------------------------------------
# dataset.json
# ---------------------------------------------------------------------------

DATASET_JSON = {
    # channel_names: integer-string keys required by BioAI's plan/preprocess pipeline
    "channel_names": {"0": "CT"},
    # labels: integer-string keys required by preprocessing4.py (int(k) on keys)
    # Values are the human-readable class names; the key is the voxel integer value.
    "labels": {
        "0": "background",
        "1": "kidney",
        "2": "tumor",
        "3": "cyst",
    },
    "file_ending": ".nii.gz",
    "dataset_name": "KiTS2023",
    "description": (
        "KiTS2023 — kidney, tumor, and cyst segmentation challenge. "
        "4-class integer labels: 0=background, 1=kidney, 2=tumor, 3=cyst."
    ),
    "reference": "https://kits-challenge.org/kits23/",
    "release": "1.0.0",
}


# ---------------------------------------------------------------------------
# Hydra data config
# ---------------------------------------------------------------------------

HYDRA_CONFIG_TEMPLATE = """\
dataset_dir: {dataset_dir}
name: kits2023
results_data_dir: {dataset_dir}/results

# roi_size: [128, 128, 128]  # Override auto-computed patch size (edit experiment_plans.json instead)
"""


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_kits2023(
    kits_base_dir: str,
    output_dir: str = "data/kits2023",
    use_symlink: bool = False,
    create_config: bool = True,
    config_dir: str = "configs/data",
) -> None:
    kits_base = Path(kits_base_dir).expanduser().resolve()
    if not kits_base.is_dir():
        sys.exit(f"ERROR: kits_base_dir does not exist: {kits_base}")

    out = Path(output_dir)
    raw_dir = out / "raw"
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"
    images_ts = raw_dir / "imagesTs"

    # --- Discover cases -------------------------------------------------------
    cases_with_seg, cases_without_seg = _discover_cases(kits_base)
    print(
        f"Found {len(cases_with_seg)} training cases (with segmentation) and "
        f"{len(cases_without_seg)} unlabelled cases."
    )

    # --- Copy/link training data ----------------------------------------------
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for case_dir in cases_with_seg:
        case_id = case_dir.name  # e.g. "case_00000"
        img_src = case_dir / "imaging.nii.gz"
        seg_src = case_dir / "segmentation.nii.gz"

        if not img_src.exists():
            print(f"  WARNING: missing imaging.nii.gz for {case_id}, skipping.")
            continue

        _copy_or_link(img_src, images_tr / f"{case_id}_0000.nii.gz", use_symlink)
        _copy_or_link(seg_src, labels_tr / f"{case_id}.nii.gz", use_symlink)

    print(
        f"  {'Linked' if use_symlink else 'Copied'} {len(cases_with_seg)} "
        f"image-label pairs → {raw_dir.relative_to(Path.cwd()) if raw_dir.is_relative_to(Path.cwd()) else raw_dir}"
    )

    # --- Copy/link unlabelled test cases (if any) ----------------------------
    if cases_without_seg:
        images_ts.mkdir(parents=True, exist_ok=True)
        for case_dir in cases_without_seg:
            case_id = case_dir.name
            img_src = case_dir / "imaging.nii.gz"
            if not img_src.exists():
                print(f"  WARNING: missing imaging.nii.gz for {case_id}, skipping.")
                continue
            _copy_or_link(img_src, images_ts / f"{case_id}_0000.nii.gz", use_symlink)
        print(
            f"  {'Linked' if use_symlink else 'Copied'} {len(cases_without_seg)} "
            f"unlabelled images → {images_ts}"
        )

    # --- Write dataset.json ---------------------------------------------------
    dataset_json_path = raw_dir / "dataset.json"
    json_data = dict(DATASET_JSON)
    json_data["num_training"] = len(cases_with_seg)
    with open(dataset_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Wrote {dataset_json_path}")

    # --- Write Hydra data config ----------------------------------------------
    if create_config:
        config_path = Path(config_dir) / "kits2023.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_dir_str = str(out).replace("\\", "/")
        config_path.write_text(HYDRA_CONFIG_TEMPLATE.format(dataset_dir=dataset_dir_str))
        print(f"  Wrote Hydra config → {config_path}")

    # --- Summary --------------------------------------------------------------
    print("\nDone. Next steps:")
    print(f"  python main2.py command=plan      data=kits2023")
    print(f"  python main2.py command=preprocess data=kits2023")
    print(f"  python main2.py command=train      data=kits2023 model=unet fold=0")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert KiTS2023 dataset to BioAI raw format "
            "(based on nnUNet Dataset220_KiTS2023)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "kits_base_dir",
        help=(
            "Root directory of the downloaded KiTS2023 dataset "
            "(must contain case_XXXXX/ subdirectories)."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="data/kits2023",
        help="Output directory for BioAI raw data (default: data/kits2023).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        default=False,
        help="Create symlinks instead of copying files (saves disk space).",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        help="Skip writing configs/data/kits2023.yaml.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/data",
        help="Directory for the Hydra data config (default: configs/data).",
    )
    args = parser.parse_args()

    convert_kits2023(
        kits_base_dir=args.kits_base_dir,
        output_dir=args.output_dir,
        use_symlink=args.symlink,
        create_config=not args.no_config,
        config_dir=args.config_dir,
    )


if __name__ == "__main__":
    main()
