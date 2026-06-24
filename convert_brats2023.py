#!/usr/bin/env python3
"""
Convert a BraTS 2023/2024 dataset folder to BioAI/nnUNet raw format.

The script takes the top-level BraTS folder (e.g. data/BraTS_2023) which
contains one or more sub-directories of cases.  Sub-directories whose cases
have a segmentation file (-seg.nii.gz) are treated as training data;
those without are treated as test data.

BraTS input layout:
    BraTS_2023/
        ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/
            BraTS-SSA-00002-000/
                BraTS-SSA-00002-000-t1n.nii.gz
                BraTS-SSA-00002-000-t1c.nii.gz
                BraTS-SSA-00002-000-t2f.nii.gz
                BraTS-SSA-00002-000-t2w.nii.gz
                BraTS-SSA-00002-000-seg.nii.gz
        BraTS2024-SSA-Challenge-ValidationData/
            BraTS-SSA-00132-000/
                BraTS-SSA-00132-000-t1n.nii.gz  (no seg)
                ...

Output layout (nnUNet raw format):
    <output_dir>/raw/
        imagesTr/
            BraTS-SSA-00002_0000.nii.gz   # t1n
            BraTS-SSA-00002_0001.nii.gz   # t1c
            BraTS-SSA-00002_0002.nii.gz   # t2f
            BraTS-SSA-00002_0003.nii.gz   # t2w
        labelsTr/
            BraTS-SSA-00002.nii.gz
        imagesTs/
            BraTS-SSA-00132_0000.nii.gz
            ...

Usage:
    python convert_brats2023.py --brats_dir data/BraTS_2023 --output_dir data/brats2023
    python convert_brats2023.py --brats_dir data/BraTS_2023 --output_dir data/brats2023 --copy
    python convert_brats2023.py --brats_dir data/BraTS_2023 --output_dir data/brats2023 --modalities t1c t2f
"""

import argparse
import shutil
import sys
from pathlib import Path


MODALITIES = ["t1n", "t1c", "t2f", "t2w"]


def get_case_files(case_dir: Path, modalities: list) -> tuple:
    """Return ({modality: path}, seg_path_or_None) for a case directory."""
    mod_files = {}
    seg_path = None
    for f in case_dir.iterdir():
        if not f.name.endswith(".nii.gz"):
            continue
        suffix = f.name[:-7].rsplit("-", 1)[-1].lower()
        if suffix == "seg":
            seg_path = f
        elif suffix in modalities:
            mod_files[suffix] = f
    return mod_files, seg_path


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def process_cases(case_dirs: list, images_out: Path, labels_out: Path | None,
                  modalities: list, copy: bool) -> None:
    for case_dir in sorted(case_dirs):
        case_id = case_dir.name.rsplit("-", 1)[0]  # drop trailing -000 suffix
        mod_files, seg_path = get_case_files(case_dir, modalities)

        missing = [m for m in modalities if m not in mod_files]
        if missing:
            print(f"  WARNING: skipping {case_id} — missing {missing}", file=sys.stderr)
            continue

        if labels_out is not None and seg_path is None:
            print(f"  WARNING: skipping {case_id} — no seg file", file=sys.stderr)
            continue

        for ch_idx, mod in enumerate(modalities):
            link_or_copy(mod_files[mod], images_out / f"{case_id}_{ch_idx:04d}.nii.gz", copy)

        if labels_out is not None:
            link_or_copy(seg_path, labels_out / f"{case_id}.nii.gz", copy)


def write_hydra_config(config_path: Path, dataset_name: str, num_channels: int) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"processed_data_dir: data/{dataset_name}/preprocessed\n"
        f"raw_data_dir: data/{dataset_name}/raw\n"
        f"validation_data_dir: data/{dataset_name}/preprocessed\n"
        f"test_data_dir: data/{dataset_name}/preprocessed\n"
        f"results_data_dir: data/{dataset_name}/results\n"
        f"roi_size: [128, 128, 128]\n"
        f"is_ct: false\n"
        f"channels: {num_channels}\n"
        f"name: {dataset_name}\n"
    )


def main():
    p = argparse.ArgumentParser(description="Convert BraTS 2023 to BioAI/nnUNet format.")
    p.add_argument("--brats_dir", required=True, type=Path,
                   help="Top-level BraTS folder containing split sub-directories.")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Output root (e.g. data/brats2023). Creates raw/ inside.")
    p.add_argument("--modalities", nargs="+", default=MODALITIES,
                   choices=MODALITIES, metavar="MOD",
                   help=f"Modalities to include in order. Default: {MODALITIES}")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of symlinking.")
    args = p.parse_args()

    brats_dir = args.brats_dir.resolve()
    output_dir = args.output_dir.resolve()
    modalities = args.modalities

    if not brats_dir.is_dir():
        sys.exit(f"ERROR: {brats_dir} is not a directory")

    # Discover split sub-directories: each sub-dir that contains case folders
    # A "case folder" is a sub-dir that holds .nii.gz files.
    train_cases, test_cases = [], []

    for split_dir in sorted(brats_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        cases = [d for d in split_dir.iterdir() if d.is_dir()]
        if not cases:
            continue
        # Check if the first case has a seg file to determine split type
        _, seg = get_case_files(cases[0], modalities)
        if seg is not None:
            train_cases.extend(cases)
        else:
            test_cases.extend(cases)

    raw_dir = output_dir / "raw"
    process_cases(train_cases, raw_dir / "imagesTr", raw_dir / "labelsTr", modalities, args.copy)
    process_cases(test_cases,  raw_dir / "imagesTs", None,                  modalities, args.copy)

    dataset_name = output_dir.name
    config_path = Path(__file__).resolve().parent / "configs" / "data" / f"{dataset_name}.yaml"
    write_hydra_config(config_path, dataset_name, len(modalities))

    n_tr = len(list((raw_dir / "imagesTr").glob("*_0000.nii.gz"))) if (raw_dir / "imagesTr").exists() else 0
    n_ts = len(list((raw_dir / "imagesTs").glob("*_0000.nii.gz"))) if (raw_dir / "imagesTs").exists() else 0
    print(f"Done.  train={n_tr} cases  test={n_ts} cases")
    print(f"Config → {config_path}")
    print(f"Next:  python main2.py command=plan data={dataset_name}")


if __name__ == "__main__":
    main()
    # Example usage (run from the project root):
    # python convert_brats2023.py --brats_dir ../download_data/BraTS_2023 --output_dir data/brats2023
    # python convert_brats2023.py --brats_dir ../download_data/BraTS_2023 --output_dir data/brats2023 --copy
    # python convert_brats2023.py --brats_dir ../download_data/BraTS_2023 --output_dir data/brats2023 --modalities t1c t2f
