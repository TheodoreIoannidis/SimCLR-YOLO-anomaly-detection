# Small Intestine Anomaly Detection using Self-Supervised Learning and YOLOv8

This repository contains the implementation of my undergraduate thesis titled **"Detection of Small Bowel Abnormalities using Self-Supervised Learning Techniques"**, which explores the use of deep learning for real-time detection and classification of small bowel anomalies in endoscopic capsule images.

## Project Overview

Wireless capsule endoscopy (WCE) is a powerful diagnostic tool for small bowel diseases, but it produces tens of thousands of frames per session, making manual review extremely time-consuming. This project aims to automate the detection and categorization of pathological findings in WCE images using the YOLOv8 object detection model, enhanced through self-supervised learning with SimCLR.

### Main Objectives:
- Detect and localize pathological findings in endoscopy frames.
- Classify detected findings into two visual classes: **Red** and **White**.
- Improve performance using self-supervised learning on unlabeled data with **SimCLR**.
- Address class imbalance with **data augmentation** techniques.

## Methods & Models

### 1. **YOLOv8 for Object Detection**
Used as the base detector to:
- Identify pathological regions in WCE frames.
- Output bounding boxes around anomalies.

### 2. **SimCLR for Self-Supervised Pretraining**
- Pretrained the YOLOv8 backbone using contrastive learning (SimCLR) on unlabeled images.
- Fine-tuned the pretrained network for anomaly detection and classification.

### 3. **Data Augmentation**
- Employed techniques like horizontal/vertical flipping, color jittering, and blurring.
- Balanced underrepresented classes and improved generalization.

## Results

- Achieved **95% accuracy** on the test set.
- Demonstrated significant improvements when using SimCLR-pretrained backbones, especially in scenarios with limited labeled data.
- Efficiently distinguished between Red and White classes using the enhanced YOLOv8 architecture.

## Dataset

The experiments utilized:
- **KVASIR Dataset**: Annotated WCE frames.
- **Rhode Island Gastroenterology Dataset**: Used for SimCLR pretraining (unlabeled frames).

Due to licensing restrictions, datasets are not included in this repository.


## How to Run

1. Clone this repository.
2. Set up your Python environment (Python ≥ 3.8, PyTorch ≥ 2.0).
3. Place your dataset in the expected folder structure.
4. Run the `Yolo_simCLR.ipynb` notebook to:
   - Pretrain with SimCLR.
   - Fine-tune with YOLOv8.
   - Evaluate performance.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Albumentations
- scikit-learn
- Ultralytics YOLOv8

(Use `pip install -r requirements.txt`.)

## Author

**Theodoros Ioannidis**  
Aristotle University of Thessaloniki  
Electrical and Computer Engineering Department

## License

This project is for academic and research purposes. For dataset licensing, please refer to the official sources.
