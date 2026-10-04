"""Download the datasets of the demo from the Hugging Face Hub into ./data.

The Loci Similes benchmark (corpus, queries, labels) and retexo's predicted edit scripts; all are public:
https://huggingface.co/collections/julian-schelb/datasets-for-latin-intertextuality-search
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

DATASETS = {
    "corpus": "julian-schelb/latin-classical-intertextuality-corpus",
    "queries": "julian-schelb/latin-classical-intertextuality-queries",
    "labels": "julian-schelb/latin-classical-intertextuality-labels",
    "edit_scripts": "julian-schelb/latin-classical-intertextuality-edit-scripts",
}


#: The page still builds without these; its section says that they are missing.
OPTIONAL = {"edit_scripts"}


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None  # optional; datasets are public
    for name, repo_id in DATASETS.items():
        print(f"Downloading {repo_id} ...")
        try:
            local = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=["data/*.parquet"],
                token=token,
            )
        except RepositoryNotFoundError:
            if name not in OPTIONAL:
                raise
            print(f"  not found on the Hub; the page is built without {name}")
            continue
        parts = sorted(Path(local).glob("data/*.parquet"))
        df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        out = data_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"  {len(df):,} rows, {len(df.columns)} columns -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    download(parser.parse_args().data_dir)
