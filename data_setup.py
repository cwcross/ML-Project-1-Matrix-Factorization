"""This script sets up the raw data necessary in a raw_data directory. It creates the directory
if it does not already exist. It checks that the parent directory is correct (e.g. you're
executing the script from the repo root)."""

import tarfile
import urllib.request
import shutil
from pathlib import Path

REQUIRED_PARENT_NAME = "ML-Project-1-Matrix-Factorization"
DATA_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"
RAW_DATA_DIR = Path("raw_data")
ARCHIVE = RAW_DATA_DIR / "lastfm-dataset-360K.tar.gz"
DESTINATION_PATH = RAW_DATA_DIR / "lastfm_360k_dataset.tsv"

def verify_execution_context():
    if Path.cwd().name != REQUIRED_PARENT_NAME:
        raise RuntimeError("Aborted. Try executing from the repo root, otherwise data will be downloaded to an arbitrary location on your device.")


def verify_raw_data_dir():
    dir_path = Path("raw_data")

    # check for existence and ensure 
    if dir_path.is_dir():
        print("raw_data directory exists")
    else:
        print("raw_data directory does not exist, creating now...",sep=" ")
        dir_path.mkdir(parents=True, exist_ok=True)


def download_data():
    if DESTINATION_PATH.exists():
        print("Data already exists")
        return

    print("downloading data. this may take a while. if interrupted, you will need to clean up download fragments and restart.")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(ARCHIVE, "wb") as f:
        shutil.copyfileobj(response, f)

    print("extracting data from tarfile")
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(RAW_DATA_DIR, filter="data")

    extracted_dir = RAW_DATA_DIR / "lastfm-dataset-360K"
    extracted = extracted_dir / "usersha1-artmbid-artname-plays.tsv"
    extracted.rename(DESTINATION_PATH)
    ARCHIVE.unlink()
    shutil.rmtree(extracted_dir)


def main():
    verify_execution_context()
    verify_raw_data_dir()
    download_data()


if __name__ == "__main__":
    main()