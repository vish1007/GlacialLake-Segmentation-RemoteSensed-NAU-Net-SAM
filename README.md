```markdown
# Glacier Lake Segmentation from Remote Sensing Data using UNet and SAM

This repository provides a pipeline for glacial lake segmentation using multi-band remote sensing data and deep learning models such as NAU-Net and SAM (Segment Anything Model). The data is processed from Landsat imagery and DEMs, enhanced with NDWI, NDSI, and slope bands.

---

## 📁 Project Structure

```

glacier-lake-segmentation/
│
├── data/                    # Dataset setup and preprocessing instructions
│   └── README.md
│
├── models/                  # Model definitions
│   ├── nau\_net.py
│   ├── sam\_model.py
│
├── training/                # Training scripts
│   ├── train\_nau\_net.py
│   ├── train\_sam.py
│
├── inference/              # Inference scripts
│   ├── infer\_nau\_net.py
│   ├── infer\_sam.py
│
├── utils/                  # Helper functions
│   ├── metrics.py
│   ├── data\_loader.py
│   ├── visualization.py
│
├── notebooks/              # Jupyter notebooks (EDA, comparison, visualization)
│   └── results\_comparison.ipynb
│
├── results/                # Outputs or checkpoints (optional)
│
├── requirements.txt        # Required Python packages
├── README.md               # Project overview and usage
├── .gitignore              # Ignore patterns
├── LICENSE                 # License file (e.g., MIT)

````

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/glacier-lake-segmentation.git
cd glacier-lake-segmentation
````

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
python inference/infer_nau_net.py
python inference/infer_sam.py
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

## 🤝 Contributions

Pull requests and suggestions are welcome. Please open an issue for major changes.

```

Let me know if you'd like to include example model outputs, Hugging Face deployment steps, or visualizations!
```
