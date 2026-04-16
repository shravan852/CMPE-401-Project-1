# CMPE 401 — Project 1: Advanced Object Detection using YOLOv11

**Design, Optimization, and Comparative Evaluation of Modern YOLO Models for Real-World Object Detection**

---
Note: Training outputs are not stored in the notebook cells as models were trained on Google Colab A100 with long-running sessions. All results, metrics, and visualizations are saved in the results/ folder and documented in this README.

## Overview

This project builds a complete object detection pipeline using YOLOv11 on the VisDrone dataset — a challenging real-world drone imagery dataset featuring small objects, dense scenes, and complex backgrounds. The project covers baseline training, loss analysis, structured experiments, iterative improvements, and multi-version comparison across 5 YOLO versions.

| Part | Description | Status |
|------|-------------|--------|
| I    | Baseline Model — YOLOv11n |   Complete|
| II   | Loss Curve & Fitting Analysis |  Complete |
| III  | Structured Experiment — Model Size (n / s / m) |  Complete |
| IV   | Iterative Improvement — Image Resolution (640 → 1280) |  Complete |
| V    | Multi-Version Comparison — v11 vs v10 vs v9 vs v8 vs v5 |  Complete |

---

## Dataset

**VisDrone DET (Task 1 — Image Detection)**
- Train set: 1.44 GB (~6,471 images)
- Val set: 0.07 GB (~548 images)
- 10 object classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- Key challenges: small objects, dense scenes, drone viewpoint, varying lighting

---

## Repository Structure

```
mpe401-project1/
├── CMPE401_Project1.ipynb           # Main notebook — Parts I, II, III, IV
├── other_training_models.ipynb      # Part V — YOLOv8, v9, v10, v5 comparison runs
├── train_yolov11m.ipynb             # YOLOv11m retraining with saved weights
├── test_challenge_inference.ipynb   # Test-challenge set inference (Part V bonus)
├── README.md
├── baseline_loss_curves.png         # Part II — Train/val loss curves
├── confusion_matrix.png             # YOLOv11m confusion matrix
├── confusion_matrix_normalized.png  # YOLOv11m normalized confusion matrix
├── results.png                      # YOLOv11m training metrics overview
├── results.csv                      # YOLOv11m epoch-by-epoch training log
├── part3_experiment.csv             # Part III — Model size experiment results
└── part5_comparison.csv             # Part V — Multi-version comparison results

---

## Part I — Baseline Results (YOLOv11n)

**Settings:** epochs=50, imgsz=640, batch=16, lr=0.01, optimizer=SGD

| Metric | Value |
|--------|-------|
| mAP@50 | 0.3027 |
| mAP@50-95 | 0.1702 |
| Precision | 0.4376 |
| Recall | 0.3260 |
| Training Time | 0.77 hrs |
| Parameters | ~2.6M |

The YOLOv11n baseline achieves a mAP@50 of 0.3027 on VisDrone. This is a reasonable starting point for a nano model on a challenging small-object dataset. The relatively low recall (0.326) reflects the difficulty of detecting small drone-view objects with a lightweight model.

---

## Part II — Loss Curve Analysis

![Baseline Loss Curves](results/baseline_loss_curves.png)

**Fitting Analysis (last 5 epochs average):**
- Train Box Loss: 1.4153
- Val Box Loss: 1.4592
- Gap: 0.0439
- **Diagnosis: GOOD FIT**

**Observations:**
- Both training and validation loss decrease steadily and consistently over 50 epochs, indicating stable learning
- The small gap (0.044) between train and val loss shows no significant overfitting
- The mAP@50 curve is still increasing at epoch 50, suggesting the model would benefit from more training epochs
- The nano model (2.6M parameters) is appropriately sized relative to the VisDrone dataset — larger models show more capacity gains (see Part III)

**Why good fit rather than overfitting:** VisDrone is a large, diverse dataset. With only 2.6M parameters, YOLOv11n has limited capacity to memorize training samples, which naturally prevents overfitting. The small train-val gap confirms this — the model generalizes well within its capacity constraints.

---

## Part III — Structured Experiment: Model Size

**Experimental variable:** Model size (n / s / m)
**Controlled constants:** epochs=50, imgsz=640, batch=16, lr=0.01, optimizer=SGD

| Model | mAP@50 | mAP@50-95 | Precision | Recall | Time (hrs) |
|-------|--------|-----------|-----------|--------|------------|
| YOLOv11n | 0.3027 | 0.1702 | 0.4376 | 0.3260 | 0.77 |
| YOLOv11s | 0.3796 | 0.2201 | 0.5274 | 0.3805 | 0.78 |
| YOLOv11m | 0.4496 | 0.2690 | 0.5818 | 0.4446 | 0.95 |

**Analysis:**
- Moving from nano → small gives a **+7.7% mAP@50 improvement** with virtually identical training time (0.77 → 0.78 hrs)
- Moving from small → medium gives another **+7.0% mAP@50 improvement** with only +22% more training time
- The medium model achieves the best accuracy across all metrics
- **Conclusion:** For VisDrone's small-object detection task, model capacity matters significantly. The YOLOv11m offers the best accuracy-to-time trade-off. The small object nature of VisDrone benefits from deeper feature extraction available in larger models.

---

## Part IV — Iterative Improvement: Image Resolution

**Baseline:** YOLOv11n, imgsz=640
**Modification:** YOLOv11n, imgsz=1280
**Motivation:** VisDrone objects are extremely small (pedestrians at drone altitude cover only ~10-20 pixels at 640 resolution). Higher resolution preserves spatial detail and improves small object recall.

**Improvement cycle:** Baseline → Higher resolution → Evaluate → Analyze

| Setting | mAP@50 | mAP@50-95 | Precision | Recall | Time (hrs) |
|---------|--------|-----------|-----------|--------|------------|
| Baseline (640) | 0.3027 | 0.1702 | 0.4376 | 0.3260 | 0.77 |
| Improved (1280) | 0.3833 | 0.2243 | 0.5047 | 0.3989 | ~1.5 (est.) |

**Analysis:**
- Resolution increase from 640 → 1280 yields a **+8.1% mAP@50 improvement** (0.3027 → 0.3833)
- Recall improves from 0.326 → 0.399 — confirming that higher resolution helps detect more small objects
- Precision also improves from 0.438 → 0.505
- **Conclusion:** For small-object datasets like VisDrone, image resolution is one of the highest-impact hyperparameters. The improvement validates the hypothesis that spatial detail is the primary bottleneck for detecting drone-view objects. The trade-off is doubled training time and halved batch size (16 → 8) due to GPU memory constraints.

---

## Part V — Multi-Version YOLO Comparison

All models trained with identical settings: epochs=50, imgsz=640, batch=16, lr=0.01, optimizer=SGD

| Model | mAP@50 | mAP@50-95 | Precision | Recall | Time (hrs) |
|-------|--------|-----------|-----------|--------|------------|
| **YOLOv9c** | **0.4445** | **0.2649** | **0.5697** | **0.4459** | 1.24 |
| YOLOv11n | 0.3027 | 0.1702 | 0.4376 | 0.3260 | 0.77 |
| YOLOv8n | 0.3087 | 0.1734 | 0.4328 | 0.3327 | 1.03 |
| YOLOv10n | 0.2971 | 0.1659 | 0.4244 | 0.3269 | 0.96 |
| YOLOv5nu | 0.2866 | 0.1607 | 0.4254 | 0.3099 | 0.70 |

**Analysis:**
- **YOLOv9c is the clear winner** at 0.4445 mAP@50 — significantly outperforming all other versions. This is expected as YOLOv9c is a larger model (compared to the nano/small variants of other versions used here)
- **YOLOv11n vs YOLOv8n:** Despite being newer, YOLOv11n and YOLOv8n perform similarly at the nano scale (0.3027 vs 0.3087), suggesting architectural improvements in v11 are more impactful at larger model sizes
- **YOLOv10n underperforms** relative to v8n and v11n despite being more recent — YOLOv10's NMS-free design trades some accuracy for inference speed
- **YOLOv5nu is fastest** (0.70 hrs) but lowest accuracy — reflects its older architecture
- **Notable:** YOLOv11m from Part III (0.4496 mAP@50) actually outperforms YOLOv9c (0.4445) — demonstrating that model size within the same version family has a greater impact than version generation alone

---

## Key Findings Summary

1. **Model size** is the single highest-impact variable — YOLOv11n → YOLOv11m improves mAP@50 by 48%
2. **Image resolution** is the second highest-impact variable — 640 → 1280 improves mAP@50 by 27%
3. **Version generation** matters less than model size — YOLOv9c (large) beats YOLOv11n (nano) but loses to YOLOv11m (medium)
4. **VisDrone is difficult** — even the best model (YOLOv11m at 0.4496) leaves significant room for improvement, reflecting the inherent challenge of small-object drone imagery

---

## How to Run

1. Download VisDrone DET dataset from https://github.com/VisDrone/VisDrone-Dataset
2. Upload zip files to Google Drive
3. Open `CMPE401_Project1_YOLO.ipynb` in Google Colab
4. Set runtime to **A100 GPU**
5. Run all cells in order

**Total runtime on A100:** ~8-10 hours

---

## References

- Ultralytics YOLOv11: https://docs.ultralytics.com/models/yolo11/
- VisDrone Dataset: https://github.com/VisDrone/VisDrone-Dataset
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics
