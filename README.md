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

## Pipeline

MetaBar provides a complete 12-step workflow:

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

---

## Contact

For questions or bug reports, please open an [issue](https://github.com/coskunlab/MetaBar/issues).

Coskun Lab · Georgia Institute of Technology
