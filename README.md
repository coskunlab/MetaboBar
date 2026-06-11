# MetaBar

**Interactive application for multimodal spatial omics processing and metabolic barcode generation.**

MetaBar integrates multiplexed immunofluorescence (IF) and mass spectrometry imaging (MSI) data into a unified point-and-click analysis pipeline. No programming experience required.

Developed by the [Coskun Lab](https://coskunlab.org) at Georgia Institute of Technology.

---

## Download

> **Windows only. Requires ~20 GB free disk space and an internet connection on first launch.**

Download the latest installer from the [**Releases**](https://github.com/coskunlab/MetaBar/releases/latest) page:

| File | Description |
|---|---|
| `MetaBar_Setup.exe` | Installer (run this) |
| `MetaBar_Setup-1.bin` | Installer data part 1 |

⚠️ **Both files must be in the same folder before running `MetaBar_Setup.exe`.**

---

## Installation

1. Download both files from the [Releases](https://github.com/coskunlab/MetaBar/releases/latest) page into the same folder.
2. Double-click `MetaBar_Setup.exe` and follow the wizard.
3. On first launch, the app will automatically download and install its Python dependencies (10–20 minutes, internet required). This only happens once.
4. The app opens in your browser at `http://localhost:8501`.

For detailed instructions, see the [User Manual](MetaBar_User_Manual.docx).

---

## Demo Data

Two datasets are available on Zenodo to explore MetaBar with real data.

### Raw Input Files — run the full pipeline from scratch
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20647909.svg)](https://doi.org/10.5281/zenodo.20647909)

**[MetaBar Demo Dataset — 10.5281/zenodo.20647909](https://doi.org/10.5281/zenodo.20647909)**

| File | Dataset | Size | Contents |
|---|---|---|---|
| `demo_clozapine_dose.zip` | Clozapine dose–response (mouse brain) | 7.2 GB | IF TIFF, MALDI TIFF, channel names |
| `demo_triomic.zip` | Trimodal AD vs WT (mouse brain) | — | IF TIFF, MALDI TIFF, channel names |
| `demo_human_colorectal_cancer.zip` | Human colorectal cancer | 65.6 MB | IF TIFF, MALDI TIFF, cell mask, phenotype annotations, channel names |

**How to use:** Load the IF and MSI TIFFs from the sidebar, then follow the Analysis Pipeline (Cell Segmentation → Clustering → GNN). The colorectal cancer sample also includes a pre-made `cell_mask.tif` and `phenotypes.csv` you can load directly in the **Custom Data** tab to skip segmentation.

---

### Processed Results — inspect pre-computed outputs immediately
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20648515.svg)](https://doi.org/10.5281/zenodo.20648515)

**[MetaBar Processed Results — 10.5281/zenodo.20648515](https://doi.org/10.5281/zenodo.20648515)**

| File | Dataset | Size | Contents |
|---|---|---|---|
| `results_clozapine_dose.zip` | Clozapine dose–response | 3.1 GB | Segmentation, projection, clustering, positivity, GNN, cross-sample comparison |
| `results_triomic.zip` | Trimodal AD vs WT | — | Segmentation, projection, clustering, positivity, GNN, cross-sample comparison |
| `results_human_colorectal_cancer.zip` | Human colorectal cancer | 464.2 MB | Segmentation, projection, annotations, GNN, cross-sample comparison |

**How to use:**
- **Napari viewer** — extract the zip, open the app, go to *Interactive Viewer (napari)*, set the results folder path, and click Launch napari.
- **Cross-sample comparison** — go to *Cross-Sample Comparative Analysis*, add each sample subfolder as a separate entry, and run comparison.
- **GNN Explainability** — go to *GNN Explainability*, set the results folder, and browse pre-computed feature importance plots.

> Each zip contains one representative sample plus the cross-sample comparison outputs for that dataset.

---

## Pipeline

MetaBar provides a complete workflow:

| Step | Description |
|---|---|
| Data Loading | IF (TIFF) and MSI (TIFF or imzML/IBD) |
| Preprocessing | Rotate, flip, interactive crop |
| Registration | MSI → IF alignment via Fiji SIFT |
| Cell Segmentation | Nuclear segmentation using Mesmer (DeepCell) |
| Nuclei Expansion | Cytoplasmic region approximation |
| MBP Mask | Myelinated tissue region detection |
| MSI Projection | LR → HR Gaussian-weighted interpolation |
| Superpixel Segmentation | SLIC-based tissue parcellation |
| Clustering | PCA + UMAP + Leiden + k-means |
| Positivity Thresholding | Per-channel GMM-based binary labelling |
| GNN Explainability | GraphSAGE/GCN/GATv2 feature importance |
| Comparative Analysis | Cross-sample GNN importance comparison |
| Custom Data | Upload your own masks and cell annotations |
| Interactive Viewer | napari-based spatial exploration |

---

## System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10 or 11 (64-bit) |
| RAM | 16 GB (32 GB recommended) |
| Disk | 20 GB free |
| GPU | NVIDIA GPU recommended (CPU fallback automatic) |
| Internet | Required on first launch only |

---

## Citation

If you use MetaBar in your research, please cite:

> Ozturk et al. (2026). *MetaBar: Interactive application for multimodal spatial omics processing and metabolic barcode generation.* Nature Communications.

---

## Contact

For questions or bug reports, please open an [issue](https://github.com/coskunlab/MetaBar/issues).

Coskun Lab · School of Biological Sciences · Georgia Institute of Technology
