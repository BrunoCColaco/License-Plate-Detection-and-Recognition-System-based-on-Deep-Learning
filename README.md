# Automatic License Plate Recognition (ALPR) System

This repository contains the implementation of my master's thesis project titled **"Development of License Plate Detection and Recognition System based on Deep Learning"**. The project is a multi-stage ALPR pipeline that processes an input image to perform:

1. **Vehicle Detection and Classification**
2. **License Plate Detection**
3. **Character Recognition**

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: Vehicle Detection and Classification](#stage-1-vehicle-detection-and-classification)
  - [Stage 2: License Plate Detection](#stage-2-license-plate-detection)
  - [Stage 3: Character Recognition](#stage-3-character-recognition)
- [Datasets and Models](#datasets-and-models)
- [Citation](#citation)

## Overview

This project implements a scalable, modular ALPR system designed to:
- Detect and classify vehicles (car, truck, bus, motorcycle).
- Detect license plates within the detected vehicles.
- Recognize characters from the detected license plates.

The pipeline leverages deep learning models for object detection, segmentation, and optical character recognition (OCR).

## System Architecture

The ALPR pipeline is divided into three stages:
1. **Vehicle Detection and Classification:** Uses object detection models to locate and classify vehicles.
2. **License Plate Detection:** Detects and segments license plates from the identified vehicle regions.
3. **Character Recognition:** Extracts and recognizes the text from the license plates.

Each stage is independent, allowing for scalability and easy replacement of components.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BrunoCColaco/License-Plate-Detection-and-Recognition-System-based-on-Deep-Learning
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. [Download](https://iselpt-my.sharepoint.com/:f:/g/personal/a45037_alunos_isel_pt/EnWO1v8KJkxFuFTPnVknsFEBztm3LcqWXJOLEns3AJ1ofA?e=NfWoqy) pretrained models and place them in the `models/` directory.
   

## Usage

To run the ALPR pipeline, use the `ALPR` class and call the `process` method with the following parameters:

```python
from alpr import ALPR

# Initialize the ALPR system
alpr_system = ALPR()

# Process images
alpr_system.process(
    images=['path/to/image1.jpg', 'path/to/image2.jpg'],
    vehicle_model='yolo',  # Choose 'faster', 'yolo', or 'ssd'
    plate_task='bbox',     # Choose 'bbox' for detection or 'seg' for segmentation
    output_dir='output/'   # Directory to save the results
)
```

### Parameters
- **`images`**: List of image file paths to process.
- **`vehicle_model`**: Specifies the model for vehicle detection (`'faster'` for Faster R-CNN, `'yolo'` for YOLOv8 [default], or `'ssd'` for SSD).
- **`plate_task`**: Task for license plate detection (`'bbox'` for bounding box detection or `'seg'` for segmentation).
- **`output_dir`**: Directory to save the output results.

## Pipeline Stages

### Stage 1: Vehicle Detection and Classification

- Models: Faster R-CNN, SSD, YOLOv8 (trained on the COCO dataset).
- Objective: Detect vehicles in the image and classify them into categories: car, truck, bus, motorcycle.
- Output: Bounding boxes of detected vehicles with their classes.

### Stage 2: License Plate Detection

- Model: YOLOv8 (segmentation variant).
- Objective: Detect license plates from the cropped vehicle regions.
- Output: Bounding boxes and segmented masks of license plates.

### Stage 3: Character Recognition

- OCR Tool: PyTesseract (OCR engine).
- Preprocessing: Includes orientation correction of plates for better recognition accuracy.


## Datasets and Models

All datasets and models can be downloaded from [here](https://iselpt-my.sharepoint.com/:f:/g/personal/a45037_alunos_isel_pt/EnWO1v8KJkxFuFTPnVknsFEBztm3LcqWXJOLEns3AJ1ofA?e=k6zqVN)


## Citation

If you use this repository, please cite:
```
@mastersthesis{bruno2024,
  title={Development of License Plate Detection and Recognition System based on Deep Learning},
  author={Bruno Colaço},
  year={2024},
  school={Instituto Superior de Engenharia de Lisboa}
}
```

