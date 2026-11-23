"""Load the UCI "gene expression cancer RNA-Seq" dataset (TCGA pan-cancer subset).

801 samples x 20531 genes, five tumor types: BRCA, KIRC, LUAD, PRAD, COAD.

The script looks for data/data.csv and data/labels.csv. If they are missing it
downloads the archive from the UCI Machine Learning Repository and extracts them.
"""
import io
import os
import tarfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLASS_NAMES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]


def download_dataset(data_dir=DATA_DIR):
    """Download the UCI archive and extract data.csv and labels.csv into data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading dataset from {UCI_URL} (about 70 MB) ...")
    with urllib.request.urlopen(UCI_URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith(".tar.gz"))
    with tarfile.open(fileobj=io.BytesIO(outer.read(inner_name)), mode="r:gz") as tar:
        for member in tar.getmembers():
            base = os.path.basename(member.name)
            if base in ("data.csv", "labels.csv"):
                with open(os.path.join(data_dir, base), "wb") as f:
                    f.write(tar.extractfile(member).read())
    print(f"Saved data.csv and labels.csv to {data_dir}")


def load_dataset(data_dir=DATA_DIR):
    """Return X (float32, samples x genes), y (int labels) and the class names."""
    data_path = os.path.join(data_dir, "data.csv")
    label_path = os.path.join(data_dir, "labels.csv")
    if not (os.path.exists(data_path) and os.path.exists(label_path)):
        download_dataset(data_dir)
    X = pd.read_csv(data_path, index_col=0).values.astype(np.float32)
    labels = pd.read_csv(label_path, index_col=0)["Class"].values
    y = np.array([CLASS_NAMES.index(c) for c in labels])
    return X, y, CLASS_NAMES


def make_synthetic(n_samples=801, n_genes=20531, seed=0):
    """Random data with the same shape and class sizes, for a quick smoke test only."""
    rng = np.random.default_rng(seed)
    sizes = [300, 78, 146, 141, 136]  # BRCA, COAD, KIRC, LUAD, PRAD
    y = np.concatenate([np.full(n, k) for k, n in enumerate(sizes)])[:n_samples]
    centers = rng.normal(0, 1, size=(len(sizes), n_genes)).astype(np.float32)
    X = centers[y] * 0.3 + rng.normal(0, 1, size=(len(y), n_genes)).astype(np.float32)
    return X, y, CLASS_NAMES
