"""Download the two datasets the review app is built from, into ./data.

- the predicted edit scripts of retexo (the app's content)
- the Loci Similes labels (the Latin texts and their English translations)

Both are public: https://huggingface.co/collections/julian-schelb/datasets-for-latin-intertextuality-search
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

DATASETS = {
    "edit_scripts": "julian-schelb/latin-classical-intertextuality-edit-scripts",
    "labels": "julian-schelb/latin-classical-intertextuality-labels",
}


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None  # optional; the datasets are public
    for name, repo_id in DATASETS.items():
        print(f"Downloading {repo_id} ...")
        local = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=["data/*.parquet"], token=token)
        parts = sorted(Path(local).glob("data/*.parquet"))
        df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        out = data_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"  {len(df):,} rows, {len(df.columns)} columns -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    download(parser.parse_args().data_dir)
