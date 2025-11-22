#!/usr/bin/env python3
import argparse
import os
import sys
import tarfile
import zipfile
import urllib.request


def download_file(url, out_path):
    """Safe downloading with progress bar."""
    if os.path.exists(out_path):
        print(f"[INFO] File already exists: {out_path}")
        return

    def report(chunk, block_size, total_size):
        percent = int(chunk * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}%")
        sys.stdout.flush()

    print(f"[INFO] Downloading: {url}")
    urllib.request.urlretrieve(url, out_path, report)
    print("\n[INFO] Download complete.")


def extract_archive(archive_path, target_dir):
    print(f"[INFO] Extracting {archive_path} ...")

    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
    else:
        raise ValueError("Unknown archive format")

    print("[INFO] Extraction finished.")


def download_musan(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    url = "http://www.openslr.org/resources/17/musan.tar.gz"
    archive = os.path.join(target_dir, "musan.tar.gz")

    download_file(url, archive)
    extract_archive(archive, target_dir)


def download_rirs(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    url = "http://www.openslr.org/resources/28/rirs_noises.zip"
    archive = os.path.join(target_dir, "rirs_noises.zip")

    download_file(url, archive)
    extract_archive(archive, target_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download MUSAN and RIRS_NOISES datasets"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="datasets",
        help="Root directory where datasets will be stored"
    )
    parser.add_argument("--musan", action="store_true", help="Download MUSAN dataset")
    parser.add_argument("--rirs", action="store_true", help="Download RIRS_NOISES dataset")
    parser.add_argument("--all", action="store_true", help="Download both datasets")

    args = parser.parse_args()
    target_dir = args.target_dir

    if not any([args.musan, args.rirs, args.all]):
        print("You must choose at least one option: --musan, --rirs, --all")
        return

    if args.all or args.musan:
        print("\n=== MUSAN ===")
        download_musan(os.path.join(target_dir, "musan"))

    if args.all or args.rirs:
        print("\n=== RIRS_NOISES ===")
        download_rirs(os.path.join(target_dir, "rirs_noises"))

    print("\nDone!")


if __name__ == "__main__":
    main()
