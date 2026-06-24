#!/usr/bin/env python3
"""
Standalone converter: AMOS22 (Abdominal Multi-Organ Segmentation 2022) → BioAI raw format.

AMOS22 source layout (after unzipping amos22.zip):
    {amos_base_dir}/
        dataset.json           (original challenge metadata, ignored)
        imagesTr/
            amos_0001.nii.gz   (CT, cases 0001-0500)
            ...
            amos_0507.nii.gz   (MRI, cases 0501-0600)
            ...
        labelsTr/
            amos_0001.nii.gz
            ...
        imagesVa/              (validation images with labels)
            amos_0301.nii.gz
            ...
        labelsVa/
            amos_0301.nii.gz
            ...
        imagesTs/              (test images, no labels)
            amos_0561.nii.gz
            ...

Modality split (by case number):
    CT : amos_0001 – amos_0500  (int suffix ≤ 500)
    MRI: amos_0501 – amos_0600  (int suffix > 500)

BioAI output layout:
    {output_dir}/
        raw/
            dataset.json
            imagesTr/  amos_XXXX_0000.nii.gz
            labelsTr/  amos_XXXX.nii.gz
            imagesTs/  amos_XXXX_0000.nii.gz

AMOS22 label values:
     0  background
     1  spleen
     2  right kidney
     3  left kidney
     4  gallbladder
     5  esophagus
     6  liver
     7  stomach
     8  aorta
     9  inferior vena cava
    10  pancreas
    11  right adrenal gland
    12  left adrenal gland
    13  duodenum
    14  bladder
    15  prostate/uterus

Usage:
    python convert_amos22.py /path/to/amos22
    python convert_amos22.py /path/to/amos22 -o data/amos22
    python convert_amos22.py /path/to/amos22 --modality ct
    python convert_amos22.py /path/to/amos22 --modality mri -o data/amos22_mri
    python convert_amos22.py /path/to/amos22 --merge-val --symlink
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


CT_MAX_IDX  = 500   # case numbers ≤ this are CT; > this are MRI


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


def _case_num(filename_stem: str) -> int:
    """amos_0001 → 1"""
    return int(filename_stem.split("_")[-1])


def _is_ct(filename_stem: str) -> bool:
    return _case_num(filename_stem) <= CT_MAX_IDX


def _modality_filter(stems: list[str], modality: str) -> list[str]:
    if modality == "ct":
        return [s for s in stems if _is_ct(s)]
    if modality == "mri":
        return [s for s in stems if not _is_ct(s)]
    return stems  # "all"


# ---------------------------------------------------------------------------
# dataset.json
# ---------------------------------------------------------------------------

LABELS = {
    "0":  "background",
    "1":  "spleen",
    "2":  "right kidney",
    "3":  "left kidney",
    "4":  "gallbladder",
    "5":  "esophagus",
    "6":  "liver",
    "7":  "stomach",
    "8":  "aorta",
    "9":  "inferior vena cava",
    "10": "pancreas",
    "11": "right adrenal gland",
    "12": "left adrenal gland",
    "13": "duodenum",
    "14": "bladder",
    "15": "prostate/uterus",
}

_CHANNEL_NAMES = {
    "ct":  {"0": "CT"},
    "mri": {"0": "MRI"},
    "all": {"0": "CT_or_MRI"},
}


# ---------------------------------------------------------------------------
# Hydra data config
# ---------------------------------------------------------------------------

HYDRA_CONFIG_TEMPLATE = """\
dataset_dir: {dataset_dir}
name: {name}
results_data_dir: {dataset_dir}/results

# roi_size: [128, 128, 128]  # Override auto-computed patch size (edit experiment_plans.json instead)
"""


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_amos22(
    amos_base_dir: str,
    output_dir: str = "data/amos22",
    modality: str = "all",
    merge_val: bool = False,
    use_symlink: bool = False,
    create_config: bool = True,
    config_dir: str = "configs/data",
) -> None:
    amos_base = Path(amos_base_dir).expanduser().resolve()
    if not amos_base.is_dir():
        sys.exit(f"ERROR: amos_base_dir does not exist: {amos_base}")

    images_tr_src = amos_base / "imagesTr"
    labels_tr_src = amos_base / "labelsTr"
    images_va_src = amos_base / "imagesVa"
    labels_va_src = amos_base / "labelsVa"
    images_ts_src = amos_base / "imagesTs"

    for d, name in [(images_tr_src, "imagesTr"), (labels_tr_src, "labelsTr")]:
        if not d.is_dir():
            sys.exit(f"ERROR: Required directory not found: {d}\n"
                     f"Make sure '{name}' exists inside {amos_base}.")

    out = Path(output_dir)
    raw_dir   = out / "raw"
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"
    images_ts = raw_dir / "imagesTs"

    # --- Collect training cases -----------------------------------------------
    train_stems = sorted(
        f.name[:-len(".nii.gz")]
        for f in images_tr_src.glob("amos_*.nii.gz")
    )
    train_stems = _modality_filter(train_stems, modality)

    # Optionally merge validation cases into training
    val_stems: list[str] = []
    if merge_val and images_va_src.is_dir() and labels_va_src.is_dir():
        val_stems = sorted(
            f.name[:-len(".nii.gz")]
            for f in images_va_src.glob("amos_*.nii.gz")
        )
        val_stems = _modality_filter(val_stems, modality)

    all_train = train_stems + val_stems
    if not all_train:
        sys.exit(
            f"ERROR: No training cases found for modality='{modality}' in {images_tr_src}."
        )

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    n_train = 0
    for stem in all_train:
        # Determine source directories (regular train or merged validation)
        if stem in train_stems:
            img_src = images_tr_src / f"{stem}.nii.gz"
            seg_src = labels_tr_src / f"{stem}.nii.gz"
        else:
            img_src = images_va_src / f"{stem}.nii.gz"
            seg_src = labels_va_src / f"{stem}.nii.gz"

        if not img_src.exists():
            print(f"  WARNING: image missing for {stem}, skipping.")
            continue
        if not seg_src.exists():
            print(f"  WARNING: label missing for {stem}, skipping.")
            continue

        _copy_or_link(img_src, images_tr / f"{stem}_0000.nii.gz", use_symlink)
        _copy_or_link(seg_src, labels_tr / f"{stem}.nii.gz",      use_symlink)
        n_train += 1

    rel = raw_dir.relative_to(Path.cwd()) if raw_dir.is_relative_to(Path.cwd()) else raw_dir
    print(
        f"{'Linked' if use_symlink else 'Copied'} {n_train} training image-label pairs → {rel}"
    )
    if merge_val and val_stems:
        print(f"  (includes {len(val_stems)} validation cases merged into training)")

    # --- Collect validation cases as test (if not merged) --------------------
    n_test = 0
    sources_ts: list[tuple[Path, str]] = []

    if not merge_val and images_va_src.is_dir():
        va_stems = sorted(
            f.name[:-len(".nii.gz")]
            for f in images_va_src.glob("amos_*.nii.gz")
        )
        va_stems = _modality_filter(va_stems, modality)
        for stem in va_stems:
            sources_ts.append((images_va_src / f"{stem}.nii.gz", stem))

    if images_ts_src.is_dir():
        ts_stems = sorted(
            f.name[:-len(".nii.gz")]
            for f in images_ts_src.glob("amos_*.nii.gz")
        )
        ts_stems = _modality_filter(ts_stems, modality)
        for stem in ts_stems:
            sources_ts.append((images_ts_src / f"{stem}.nii.gz", stem))

    if sources_ts:
        images_ts.mkdir(parents=True, exist_ok=True)
        for img_src, stem in sources_ts:
            if not img_src.exists():
                print(f"  WARNING: test image missing for {stem}, skipping.")
                continue
            _copy_or_link(img_src, images_ts / f"{stem}_0000.nii.gz", use_symlink)
            n_test += 1
        print(f"{'Linked' if use_symlink else 'Copied'} {n_test} test images → {images_ts}")

    # --- Write dataset.json ---------------------------------------------------
    dataset_name = Path(output_dir).name
    json_data = {
        "channel_names": _CHANNEL_NAMES[modality],
        "labels": LABELS,
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
        "description": (
            f"AMOS22 — abdominal multi-organ segmentation, 15 classes "
            f"({'CT only' if modality == 'ct' else 'MRI only' if modality == 'mri' else 'CT + MRI'})."
        ),
        "reference": "https://amos22.grand-challenge.org/",
        "release": "1.0.0",
        "num_training": n_train,
    }
    dataset_json_path = raw_dir / "dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Wrote {dataset_json_path}")

    # --- Write Hydra data config ----------------------------------------------
    if create_config:
        config_path = Path(config_dir) / f"{dataset_name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_dir_str = str(out).replace("\\", "/")
        config_path.write_text(
            HYDRA_CONFIG_TEMPLATE.format(dataset_dir=dataset_dir_str, name=dataset_name)
        )
        print(f"Wrote Hydra config → {config_path}")

    # --- Summary --------------------------------------------------------------
    print("\nDone. Next steps:")
    print(f"  python main2.py command=plan      data={dataset_name}")
    print(f"  python main2.py command=preprocess data={dataset_name}")
    print(f"  python main2.py command=train      data={dataset_name} model=unet fold=0")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AMOS22 dataset to BioAI raw format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "amos_base_dir",
        help=(
            "Root directory of the unzipped AMOS22 dataset "
            "(must contain imagesTr/ and labelsTr/)."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="data/amos22",
        help="Output directory for BioAI raw data (default: data/amos22).",
    )
    parser.add_argument(
        "--modality",
        choices=["ct", "mri", "all"],
        default="all",
        help=(
            "Which modality to include: 'ct' (cases 0001-0500), "
            "'mri' (cases 0501-0600), or 'all' (default)."
        ),
    )
    parser.add_argument(
        "--merge-val",
        action="store_true",
        default=False,
        help=(
            "Merge imagesVa/labelsVa into training data instead of treating "
            "them as test cases (recommended for maximum training data)."
        ),
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
        help="Skip writing the Hydra data config yaml.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/data",
        help="Directory for the Hydra data config (default: configs/data).",
    )
    args = parser.parse_args()

    convert_amos22(
        amos_base_dir=args.amos_base_dir,
        output_dir=args.output_dir,
        modality=args.modality,
        merge_val=args.merge_val,
        use_symlink=args.symlink,
        create_config=not args.no_config,
        config_dir=args.config_dir,
    )


if __name__ == "__main__":
    main()
