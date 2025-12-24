# Project Memory: 3D Tooth Segmentation (PyTorch/MONAI)

## 1. Project Overview
This project implements a Deep Learning framework for 3D Tooth Segmentation. It ranges from baseline methods to advanced Neural Architecture Search (NAS).
- **Core Frameworks**: PyTorch, MONAI
- **Task**: 3D Volumetric Segmentation (CBCT/MRI)
- **Key Methods**: 3D U-Net, DiNTS (Differentiable Neural Network Topology Search), TMO (Topology-Aware Multi-Objective / Semi-supervised).

## 2. Environment & Config
- **Conda Environment**: `tooth`
- **Project Root**: `/home/lzf/Code/3D-Tooth-Segment`
- **Dataset Path**: `/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI`
  > **⚠️ Crucial**: `DATA_DIR` is **hardcoded** inside python scripts. Must verify before running.

## 3. Key Scripts (Entry Points)
All commands should be run from the Project Root.

### A. Baseline Training
- **Script**: `scripts/unet3D_train.py`
- **Model**: Standard `UNet3D` (src/models/unet3D.py)
- **Input**: Labeled 3D volumes.
- **Output**: `weights/best_unet3D_model.pth`
- **Default Params**: 5 Epochs, Batch=1, ROI=(64,64,64).

### B. Architecture Search (NAS)
- **Supervised Search**: `scripts/dints_search.py`
- **Semi-Supervised (TMO)**: `scripts/dints_tmo_search.py`
  - **Method**: Student-Teacher model (`DiNTSWrapper`).
  - **Loader**: `NASComboDataLoader` (Mixed Labeled/Unlabeled).
  - **Goal**: Search for optimal topology (`arch_code`).
  - **Output**: `results/dints_tmo_search/best_arch.json`

### C. Advanced Training
- **SSL Pre-training**: `scripts/unet3D_ssl_train.py`
- **Retrain Searched Arch**: `scripts/dints_retrain.py` (Likely loads JSON architecture).

## 4. Code Structure
- **`src/models/`**:
  - `unet3D.py`: Baseline model.
  - `dints.py`: Dynamic network definition for NAS.
- **`src/dataloaders/`**:
  - `basic_loader.py`: Standard supervised loader.
  - `combo_loader.py`: Complex loader for SSL/NAS (4-stream data).
- **`src/ssl/`**: TMO optimizer and Consistency Loss utilities.
- **`results/`**: Stores architecture search results (JSON).
- **`weights/`**: Stores model checkpoints (`.pth`).

## 5. Developer Notes
- **VRAM Constraints**: 
  - `dints_tmo_search.py` defaults to **Batch_Size=2**.
  - `unet3D_train.py` defaults to **Batch_Size=1**.
- **Determinism**: Random seeds are set to `2025` or `0` for reproducibility.
- **Warnings**: `monai.inferers.utils` UserWarnings are suppressed in scripts.
