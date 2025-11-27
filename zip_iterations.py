"""
File: zip_iterations.py
Summary: Zips iteration outputs for distribution.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one ZIP archive of .pkl files per iteration directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations-dir",
        required=True,
        type=str,
        help="Directory whose immediate subdirectories are iteration runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "If provided, write archives here as '<iteration_name>.pkls.zip'. "
            "If omitted, write 'pkls.zip' inside each iteration directory."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.pkl",
        help="Glob pattern (relative to each iteration dir) selecting files to include.",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of iteration directory names to process (others are skipped).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing archives if they already exist.",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original .pkl files after successful archiving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating or deleting anything.",
    )
    return parser.parse_args()


def find_iteration_directories(iterations_root: Path) -> List[Path]:
    return sorted([p for p in iterations_root.iterdir() if p.is_dir()])


def collect_files(iteration_dir: Path, pattern: str) -> List[Path]:
    # Path.glob supports '**' for recursive matches
    return sorted([p for p in iteration_dir.glob(pattern) if p.is_file()])


def write_zip_archive(
    files: Iterable[Path], base_dir: Path, destination_zip: Path
) -> None:
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination_zip, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for file_path in files:
            # Store relative paths for portability
            arcname = file_path.relative_to(base_dir).as_posix()
            zf.write(file_path, arcname)


def main() -> int:
    args = parse_args()
    iterations_root = Path(args.iterations_dir).expanduser().resolve()
    if not iterations_root.exists() or not iterations_root.is_dir():
        print(
            f"ERROR: --iterations-dir does not exist or is not a directory: {iterations_root}"
        )
        return 2

    output_dir: Path | None = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()

    iteration_dirs = find_iteration_directories(iterations_root)
    if args.only:
        only_set = set(args.only)
        iteration_dirs = [d for d in iteration_dirs if d.name in only_set]

    if not iteration_dirs:
        print("No iteration directories found.")
        return 0

    total = len(iteration_dirs)
    created = 0
    skipped = 0
    empty = 0
    deleted_files = 0

    for idx, iter_dir in enumerate(iteration_dirs, start=1):
        files = collect_files(iter_dir, args.pattern)
        if not files:
            empty += 1
            print(
                f"[{idx}/{total}] {iter_dir.name}: no files matched pattern '{args.pattern}', skipping"
            )
            continue

        if output_dir is not None:
            archive_path = output_dir / f"{iter_dir.name}.pkls.zip"
        else:
            archive_path = iter_dir / "pkls.zip"

        if archive_path.exists() and not args.overwrite:
            skipped += 1
            print(
                f"[{idx}/{total}] {iter_dir.name}: archive exists, use --overwrite to replace -> {archive_path}"
            )
            continue

        if args.dry_run:
            print(
                f"[{idx}/{total}] {iter_dir.name}: would create {archive_path} with {len(files)} file(s)"
            )
        else:
            try:
                write_zip_archive(
                    files, base_dir=iter_dir, destination_zip=archive_path
                )
                created += 1
                print(
                    f"[{idx}/{total}] {iter_dir.name}: created {archive_path} ({len(files)} file(s))"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[{idx}/{total}] {iter_dir.name}: ERROR creating {archive_path}: {exc}"
                )
                continue

        if args.delete_originals:
            to_delete = [p for p in files if p.suffix == ".pkl"]
            if args.dry_run:
                print(
                    f"[{idx}/{total}] {iter_dir.name}: would delete {len(to_delete)} original .pkl file(s)"
                )
            else:
                for p in to_delete:
                    try:
                        p.unlink(missing_ok=True)
                        deleted_files += 1
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[{idx}/{total}] {iter_dir.name}: ERROR deleting {p}: {exc}"
                        )

    print(
        "Summary: "
        f"created={created}, skipped_existing={skipped}, empty={empty}, deleted_pkls={deleted_files}, total_iters={total}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
