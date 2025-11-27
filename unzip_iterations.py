"""
File: unzip_iterations.py
Summary: Unzips archived iteration artifacts.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unzip .pkl archives per iteration directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations-dir",
        required=True,
        type=str,
        help="Directory whose immediate subdirectories are iteration runs (extraction targets).",
    )
    parser.add_argument(
        "--archives-dir",
        type=str,
        default=None,
        help=(
            "Directory containing '<iteration_name>.pkls.zip' files. "
            "If omitted, the script expects 'pkls.zip' inside each iteration directory."
        ),
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        default="pkls.zip",
        help="Archive filename to search for inside each iteration directory (Mode 1).",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of iteration directory names to process (others are skipped).",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete archive file(s) after successful extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without extracting or deleting anything.",
    )
    return parser.parse_args()


def find_iteration_directories(iterations_root: Path) -> List[Path]:
    return sorted([p for p in iterations_root.iterdir() if p.is_dir()])


def extract_archive(archive_path: Path, destination_dir: Path, dry_run: bool) -> bool:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        # Return True to indicate a would-be success
        return True
    try:
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(path=destination_dir)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR extracting {archive_path} -> {destination_dir}: {exc}")
        return False


def main() -> int:
    args = parse_args()
    iterations_root = Path(args.iterations_dir).expanduser().resolve()
    if not iterations_root.exists() or not iterations_root.is_dir():
        print(
            f"ERROR: --iterations-dir does not exist or is not a directory: {iterations_root}"
        )
        return 2

    only_set = set(args.only) if args.only else None

    successes = 0
    skipped = 0
    missing = 0
    deleted = 0

    if args.archives_dir:
        archives_root = Path(args.archives_dir).expanduser().resolve()
        if not archives_root.exists() or not archives_root.is_dir():
            print(
                f"ERROR: --archives-dir does not exist or is not a directory: {archives_root}"
            )
            return 2

        zip_files = sorted(archives_root.glob("*.pkls.zip"))
        if only_set is not None:
            # Accept either '<name>.pkls.zip' or '<name>.zip' with suffix match
            zip_files = [
                z
                for z in zip_files
                if (
                    z.name.endswith(".pkls.zip") and z.name[:-9] in only_set
                )  # strip '.pkls.zip'
            ]

        total = len(zip_files)
        if total == 0:
            print("No archives found in archives directory.")
            return 0

        for idx, zf_path in enumerate(zip_files, start=1):
            # iteration name = filename without '.pkls.zip'
            if not zf_path.name.endswith(".pkls.zip"):
                skipped += 1
                print(f"[{idx}/{total}] skip non-matching archive: {zf_path.name}")
                continue
            iteration_name = zf_path.name[:-9]
            if only_set is not None and iteration_name not in only_set:
                skipped += 1
                print(f"[{idx}/{total}] {iteration_name}: not in --only, skipping")
                continue

            dest_dir = iterations_root / iteration_name
            if args.dry_run:
                print(
                    f"[{idx}/{total}] {iteration_name}: would extract {zf_path} -> {dest_dir}"
                )
            ok = extract_archive(zf_path, dest_dir, dry_run=args.dry_run)
            if ok:
                successes += 1
                print(
                    f"[{idx}/{total}] {iteration_name}: extracted {zf_path} -> {dest_dir}"
                )
                if args.delete_archives:
                    if args.dry_run:
                        print(
                            f"[{idx}/{total}] {iteration_name}: would delete archive {zf_path}"
                        )
                    else:
                        try:
                            zf_path.unlink(missing_ok=True)
                            deleted += 1
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[{idx}/{total}] {iteration_name}: ERROR deleting {zf_path}: {exc}"
                            )
            else:
                print(
                    f"[{idx}/{total}] {iteration_name}: extraction failed; archive retained"
                )

    else:
        # Mode 1: look for 'pkls.zip' inside each iteration directory
        iteration_dirs = find_iteration_directories(iterations_root)
        if only_set is not None:
            iteration_dirs = [d for d in iteration_dirs if d.name in only_set]

        total = len(iteration_dirs)
        if total == 0:
            print("No iteration directories found.")
            return 0

        for idx, iter_dir in enumerate(iteration_dirs, start=1):
            archive_path = iter_dir / args.archive_name
            if not archive_path.exists():
                missing += 1
                print(
                    f"[{idx}/{total}] {iter_dir.name}: archive not found -> {archive_path}"
                )
                continue

            if args.dry_run:
                print(
                    f"[{idx}/{total}] {iter_dir.name}: would extract {archive_path} -> {iter_dir}"
                )
            ok = extract_archive(archive_path, iter_dir, dry_run=args.dry_run)
            if ok:
                successes += 1
                print(f"[{idx}/{total}] {iter_dir.name}: extracted {archive_path}")
                if args.delete_archives:
                    if args.dry_run:
                        print(
                            f"[{idx}/{total}] {iter_dir.name}: would delete archive {archive_path}"
                        )
                    else:
                        try:
                            archive_path.unlink(missing_ok=True)
                            deleted += 1
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[{idx}/{total}] {iter_dir.name}: ERROR deleting {archive_path}: {exc}"
                            )
            else:
                print(
                    f"[{idx}/{total}] {iter_dir.name}: extraction failed; archive retained"
                )

    print(
        "Summary: "
        f"extracted={successes}, skipped={skipped}, missing={missing}, deleted_archives={deleted}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
