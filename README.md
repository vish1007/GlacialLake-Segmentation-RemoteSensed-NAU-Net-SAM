# Glacier Lake Segmentation from Remote Sensing Data using NAU-Net and SAM

This repository presents a complete pipeline for glacial lake segmentation leveraging multi-band remote sensing data and deep learning models, including NAU-Net and the Segment Anything Model (SAM). The input data is derived from Landsat imagery and Digital Elevation Models (DEMs), enhanced with additional indices such as NDWI (Normalized Difference Water Index), NDSI (Normalized Difference Snow Index), and slope information.

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/vish1007/GlacialLake-Segmentation-RemoteSensed-NAU-Net-SAM.git
cd glacier-lake-segmentation
````

### 2. Create and activate a virtual environment (recommended)

```bash
conda create -n glacier-segmentation python=3.9 -y
conda activate glacier-segmentation

```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 💡 Note on Project Imports

Python allows importing sibling directories using relative or absolute imports as long as the project root is in the Python path.

For example, `training/train_nau_net.py` can import from `utils` like this:

```python
from utils.metrics import compute_iou
from utils.data_loader import get_dataloader
```

To run the training script properly, execute it from the **project root**:

```bash
python training/train_nau_net.py
```

Avoid running the script from within the `training/` folder directly.

---

## 🚀 How to Run

### 1. Train NAU-Net

```bash
python training/train_nau_net.py
```

### 2. Train SAM model with UNet-generated masks as prompts

```bash
python training/train_sam.py
```

### 3. Inference using trained models

```bash
python inference_testing/NAU-SAM_test.py
```

---

## 📦 Dataset

Detailed instructions for data preparation, including:

* Downloading Landsat and DEM data
* Computing NDWI, NDSI, slope
* Creating patches
* Band stacking and formatting

See [`data/README.md`](data/README.md) for full preprocessing steps.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✍️ Citation

If you use this code or dataset in your research, please cite:

> Chen, F., et al. “Annual 30 m dataset for glacial lakes in high mountain Asia from 2008 to 2017.” Earth System Science Data (2021).

---
