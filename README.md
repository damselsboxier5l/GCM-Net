# GCM-Net: End-to-End Falling Object Detection via Motion-Aware Global Context Modeling

Official implementation of **GCM-Net**, a high-efficiency one-stage detector specifically designed for Falling Object Detection around Buildings (FODB).

## 🚀 Highlights

- **Motion-Aware Selective Scan (MASS):** Integrates motion priors into a State Space Model (Mamba) for global spatiotemporal context modeling, effectively recovering features from severe motion blur.
- **End-to-End Efficiency:** A streamlined one-stage framework based on the YOLO architecture, achieving **45.2 FPS** (3$	imes$ faster than previous SOTA specialized detectors).
- **Tiny Object Sensitivity:** Features **Scale-Sensitive Weighted Loss (SSW-Loss)** to address the scarcity of pixel-level information for objects smaller than $10 	imes 10$.
- **Stable Optimization:** Employs **Drop-MuSGD**, a randomized curvature-aware optimizer tailored for hybrid CNN-SSM architectures.

---

## 🛠 Installation

```bash
# Install in editable mode
pip install -e .
```
*Note: It is recommended to run all commands in the same Python environment where the package is installed.*

---

## 📈 Training & Evaluation

### 1) Data Preparation
Configure your dataset in `data/paowu6.yaml`. GCM-Net uses a **6-channel input** stream:
- **Channels 1-3:** RGB Frames
- **Channels 4-6:** Motion Masks (generated via MOG2)

Key YAML settings:
```yaml
channels: 6
paired_mask: true
mask_dir: mask  # Path to paired background subtraction masks
```

### 2) Training (GCM-Net on YOLO26/YOLO11)
To train the model on the FADE benchmark:

```bash
# Using YOLO26-based backbone (GCM-Net-L)
python3 -u train.py --model ultralytics/cfg/models/26/yolo26-fade.yaml --data data/paowu6.yaml --imgsz 960 --device 0
```

### 3) Inference & Prediction
```bash
python3 -u predict.py --model ultralytics/cfg/models/26/yolo26-fade.yaml --source path/to/video_or_images --imgsz 960 --device 0
```

---

## 🏗 Architecture & Code Structure

The core modules of GCM-Net are integrated into the `ultralytics/` framework:

- `ultralytics/nn/modules/`: Contains the **MASS Block** and Mamba-based selective scan implementations.
- `ultralytics/data/base.py`: Handles the **6-channel spatiotemporal data flow** (RGB + paired masks).
- `ultralytics/utils/loss.py`: Implementation of the **Scale-Sensitive Weighted Loss**.
- `ultralytics/utils/optimizers.py`: Implementation of the **Drop-MuSGD** optimizer.

<!-- ---

## 📊 Benchmark Results (FADE Dataset)

| Method | F-measure | Precision | Recall | FPS |
| :--- | :---: | :---: | :---: | :---: |
| FADE-Net (Two-stage) | 72.08 | 73.52 | 70.69 | 15.7 |
| **GCM-Net (Ours)** | **70.67** | **72.15** | **69.25** | **45.2** |
| YOLOv11-L (Baseline) | 62.63 | 66.10 | 59.50 | 48.5 | -->

<!-- --- -->

<!-- ## 📝 Citation -->
<!-- 
If you find GCM-Net useful in your research, please cite our paper:

```bibtex
@article{bai2026gcmnet,
  title={GCM-Net: end-to-end falling object detection via motion-aware global context modeling},
  author={Bai, Shan},
  journal={Journal of Electronic Imaging},
  year={2026}
}
``` -->
