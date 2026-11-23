# Deep Learning for Cancer Type Identification from RNA-Seq Gene Expression: A Comparison with ML

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FCMSS69636.2026.11688231-blue)](https://doi.org/10.1109/CMSS69636.2026.11688231)
[![IEEE](https://img.shields.io/badge/IEEE-CMSS%202026-00629B)](https://doi.org/10.1109/CMSS69636.2026.11688231)

Code and summary for our paper published in the **Proceedings of the 2026 International Conference on Modern Sustainable Systems (CMSS)**, IEEE, pp. 1404–1407.

**Paper:** [doi.org/10.1109/CMSS69636.2026.11688231](https://doi.org/10.1109/CMSS69636.2026.11688231) (IEEE Xplore)

## Authors

- **Mithun Das**, School of Business, International American University, Los Angeles, CA
- **Akbor Aziz Susom**, College of AI, Cyber and Computing, University of Texas at San Antonio, TX
- **Md Azmir Bhuiyan**, School of Engineering, San Francisco Bay University, Fremont, CA
- **Abdullah Al Maimun**, Lamar University, Beaumont, TX
- **Rakib Hasan**, Dept. of ECE, Lamar University, Beaumont, TX

## What the paper asks

On a standard cancer classification task from RNA-Seq gene expression, does deep learning really beat simple machine learning, and if so, when?

We compared three neural networks against two classical baselines on 801 tumor samples (20,531 genes, five cancer types), using stratified 5-fold cross-validation repeated over three seeds.

## Key findings

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Linear SVM | 0.9996 | 0.9997 |
| **MLP (ours)** | **0.9996** | **0.9996** |
| Random Forest | 0.9958 | 0.9965 |
| Autoencoder + softmax | 0.9938 | 0.9929 |
| 1D-CNN | 0.9197 | 0.9203 |

*Mean over 5-fold cross-validation × 3 seeds (Table I of the paper).*

- **On the full data, the models are nearly tied.** A plain multilayer perceptron matches the linear SVM and edges out the random forest.
- **The 1D-CNN trails (F1 0.920)** because genes in the profile have no spatial order, so a sliding convolution filter has no local pattern to learn. The architecture has to fit the structure of the data.
- **Deep learning helps most when labels are scarce.** With only about 28 labeled samples, the MLP reaches 0.970 accuracy, ahead of the random forest (0.952) and the autoencoder (0.964). The gap closes as training data grows, and by about 220 samples all three models exceed 0.99.
- **Errors are rare** and fall between breast and lung tumors. Kidney, prostate and colon are separated cleanly by every model.

## Reproduction with this code

We re-ran all three experiments with the code in this repository on the real UCI dataset (CPU, PyTorch 2.x, scikit-learn 1.x). The output is in [`results/`](results).

### Full-data benchmark (5-fold CV × 3 seeds)

| Model | Paper: Macro F1 | This code: Macro F1 (± std) | This code: Accuracy |
|---|---:|---:|---:|
| Linear SVM | 0.9997 | **1.0000** ± 0.0000 | 1.0000 |
| MLP | 0.9996 | 0.9986 ± 0.0032 | 0.9983 |
| Random Forest | 0.9965 | 0.9968 ± 0.0039 | 0.9963 |
| Autoencoder + softmax | 0.9929 | 0.9979 ± 0.0038 | 0.9983 |
| 1D-CNN | 0.9203 | 0.9887 ± 0.0084 | 0.9892 |

![Macro F1 by model](results/fig1_macro_f1.png)

### Label scarcity (fixed test set, mean over 3 seeds)

| Labeled training samples | MLP | Random Forest | Autoencoder + softmax |
|---:|---:|---:|---:|
| 28 | 0.967 | 0.963 | 0.975 |
| 56 | 0.990 | 0.990 | 0.992 |
| 112 | 0.995 | 0.996 | 0.993 |
| 224 | 0.997 | 0.996 | 0.996 |
| 392 | 0.999 | 0.996 | 0.999 |
| 560 (all) | 1.000 | 0.996 | 1.000 |

![Accuracy vs. labeled training samples](results/fig2_label_scarcity.png)

### MLP confusion matrix (one 5-fold run, all 801 samples)

798 of 801 samples are classified correctly (accuracy 0.9975). The only errors are 2 lung (LUAD) samples predicted as breast (BRCA), the same breast–lung confusion described in the paper.

![MLP confusion matrix](results/fig3_confusion_matrix.png)

### How this compares with the paper

- **Same overall picture:** all models score very high on the full data, the 1D-CNN is the lowest and most variable model, and every model passes 0.99 accuracy by about 220 labeled samples.
- **The 1D-CNN gap is much smaller here** (0.989 vs. 0.920). The paper does not give the exact CNN architecture, and the version in `src/models.py` (strided convolutions with adaptive pooling) evidently copes better with the unordered gene axis. The paper's point that convolution is a poor structural fit still shows: it remains the weakest model, but the margin depends on the architecture.
- **Label scarcity:** with 28 labeled samples both neural models beat the random forest, as in the paper. In this run the autoencoder (0.975) edges out the MLP (0.967), while the paper reports the MLP ahead.
- Differences like these are expected from a reimplementation with different, unpublished hyperparameters. The numbers in the *Key findings* section above are the published ones.

## Dataset

[Gene expression cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) from the UCI Machine Learning Repository, a curated subset of the TCGA Pan-Cancer collection.

| Tumor type | Code | Samples |
|---|---|---:|
| Breast invasive carcinoma | BRCA | 300 |
| Kidney renal clear cell carcinoma | KIRC | 146 |
| Lung adenocarcinoma | LUAD | 141 |
| Prostate adenocarcinoma | PRAD | 136 |
| Colon adenocarcinoma | COAD | 78 |

The dataset is **not** stored in this repository. The code downloads it from UCI on first run (about 70 MB) into `data/`.

## Methods

- **MLP:** two hidden layers with batch normalization and dropout
- **1D-CNN:** strided 1D convolutions and pooling over the standardized gene vector, then a small classifier head
- **Autoencoder + softmax:** compresses each profile to a 64-value code and classifies from it, trained with a combined reconstruction + classification loss
- **Random Forest** and **Linear SVM** (scikit-learn) as classical baselines
- All neural models use the Adam optimizer, dropout and batch normalization
- Features are standardized with **training-fold statistics only**, to avoid data leakage
- **Label-scarcity study:** a fixed held-out test set; models are trained on 5%–100% of the remaining data, over three seeds

## Repository structure

```
cancer-type-classification-rnaseq/
├── src/
│   ├── data.py              # downloads and loads the UCI dataset
│   ├── models.py            # MLP, 1D-CNN, Autoencoder (PyTorch) + RF, Linear SVM (scikit-learn)
│   └── run_experiments.py   # benchmark (Table I, Fig. 1), label scarcity (Fig. 2), confusion matrix (Fig. 3)
├── results/                 # tables (.csv) and figures (.png) from our run on the real dataset
├── CITATION.cff
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

Requires Python 3.9+.

```bash
git clone https://github.com/azmir9083/cancer-type-classification-rnaseq.git
cd cancer-type-classification-rnaseq
pip install -r requirements.txt

# quick check that everything runs (random data, a few minutes on CPU)
python src/run_experiments.py all --synthetic --seeds 1 --epochs 3

# full reproduction on the real dataset (downloads it automatically)
python src/run_experiments.py benchmark    # Table I + Fig. 1
python src/run_experiments.py scarcity     # Fig. 2
python src/run_experiments.py confusion    # Fig. 3
```

Tables (`.csv`) and figures (`.png`) are written to `results/`. The full benchmark trains 75 models (5 models × 5 folds × 3 seeds), so expect it to take a while on CPU. A GPU is used automatically if available.

> **Note on the code:** this is a clean reimplementation written from the methods described in the paper. The paper does not list every hyperparameter (exact layer widths, learning rate, epochs), so the values in `src/models.py` are reasonable defaults. See *Reproduction with this code* above for how its results compare with the published numbers.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{das2026deep,
  title     = {Deep Learning for Cancer Type Identification from RNA-Seq Gene Expression: A Comparison with ML},
  author    = {Das, Mithun and Susom, Akbor Aziz and Bhuiyan, Md Azmir and Al Maimun, Abdullah and Hasan, Rakib},
  booktitle = {2026 International Conference on Modern Sustainable Systems (CMSS)},
  pages     = {1404--1407},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/CMSS69636.2026.11688231}
}
```

The published paper is © 2026 IEEE and is available through [IEEE Xplore](https://doi.org/10.1109/CMSS69636.2026.11688231).
