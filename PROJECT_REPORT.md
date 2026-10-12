# AI Plant Disease Detection System — Project Report Notes

## Abstract

Plant diseases can reduce crop quality and productivity. Early identification of suspicious leaf symptoms can help growers decide when to inspect a crop more carefully. This project implements an image-classification prototype that uses transfer learning with MobileNetV2 to classify plant-leaf photos into categories available in the PlantVillage dataset. A Flask-based web application provides an image upload screen, top-three model predictions, confidence scores, and general guidance. The model is evaluated using a held-out test split, and the system saves standard classification metrics and a confusion matrix.

The application is an educational decision-support tool. Its output should not be treated as a definitive diagnosis because controlled dataset images may differ from photographs captured outdoors.

## Problem statement

Manual identification of visible plant-leaf disease symptoms can be difficult for users without agricultural training. The project aims to build a simple web application that applies a trained image-classification model to a leaf image and presents a readable prediction with model scores.

## Objectives

1. Prepare labeled plant-leaf image data for supervised learning.
2. Use transfer learning to build a multi-class image classifier.
3. Create a responsive web interface for image upload and prediction.
4. Display top-three predicted categories and their confidence scores.
5. Evaluate the model using test accuracy, precision, recall, F1-score, and a confusion matrix.
6. State the system's limitations and encourage expert confirmation of disease concerns.

## Scope

The prototype classifies images only among categories present in the dataset used for training. It does not identify arbitrary plants or diseases outside those labels. It does not connect to farm sensors, prescribe chemicals, or guarantee field-level performance.

## Existing system and motivation

Visual assessment by a person can require crop-specific experience, and symptoms may overlap across diseases or be affected by lighting, nutrition, pests, and weather. A computer-vision classifier can provide a fast preliminary category match, but should support rather than replace expert assessment.

## Proposed system

The proposed system uses a MobileNetV2 convolutional neural network initialized with ImageNet weights. It resizes leaf images, applies modest augmentation while training, learns a classification head, and optionally fine-tunes the final backbone layers. The best validation checkpoint is evaluated on a test subset and served by a Flask application.

## System modules

### 1. Dataset preparation

Reads class-labeled folders, gathers supported image files, creates stratified train/validation/test splits, and converts images into 224 × 224 RGB tensors.

### 2. Model training

Uses MobileNetV2 as the feature-extraction backbone, global average pooling, dropout, and a softmax output layer. Sparse categorical cross-entropy is used for optimization. Early stopping and learning-rate reduction help training converge.

### 3. Model evaluation

Reports test loss and accuracy, per-class precision/recall/F1-score, and a confusion matrix. Metrics must be generated from the actual training run; they should not be fabricated or assumed in the report.

### 4. Web application

Flask serves the frontend and prediction endpoint. The frontend validates file type and size, displays an image preview, submits the image, and renders the model's top-three predictions and guidance.

### 5. Result interpretation

The largest softmax output is shown as the top class. The score is a model confidence score and is not a calibrated probability of a real-world diagnosis. General guidance is intentionally conservative and asks users to seek local expertise before taking high-impact action.

## Technology stack

- Programming language: Python
- Deep learning: TensorFlow / Keras, MobileNetV2
- Image processing: Pillow, TensorFlow image utilities
- Data handling: NumPy, Pandas
- Evaluation: scikit-learn, Matplotlib
- Backend: Flask
- Frontend: HTML, CSS, JavaScript
- Dataset: PlantVillage

## Hardware and software requirements

- Python 3.10 or 3.11 (64-bit)
- 8 GB RAM recommended
- CPU; GPU acceleration is helpful for faster training
- Internet access during initial setup to install packages and retrieve pretrained weights
- PlantVillage images stored in class-named folders
- Modern desktop or mobile browser

## Workflow

1. Download and extract the dataset into `data/PlantVillage/`.
2. Install dependencies from `requirements.txt` in a virtual environment.
3. Run `python train.py` and wait for training/evaluation to finish.
4. Review files created in `models/` and `reports/`.
5. Run `python app.py`.
6. Open `http://127.0.0.1:5000`, upload a leaf photo, and review the result.

## Evaluation measures

- **Accuracy:** fraction of test images classified correctly.
- **Precision:** fraction of predicted class instances that are correct.
- **Recall:** fraction of actual class instances correctly found.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** shows which classes are commonly confused by the model.

For multi-class image classification, per-class precision, recall, and F1-score should be examined in addition to overall accuracy. A random image split may also overestimate performance if related or near-duplicate images cross the split boundary. A research-grade assessment should use a leaf-group-aware split where available.

## Limitations

- Predictions are limited to training labels.
- PlantVillage images may have cleaner backgrounds and different lighting from outdoor photos.
- Visible symptoms can overlap between diseases and non-disease stresses.
- Softmax scores may look high even when an image is unfamiliar or out of scope.
- Model training time depends heavily on hardware.
- Guidance in the demo is general and is not a treatment recommendation.

## Future enhancements

- Use official group-aware splits and evaluate on a separate set of real field photos.
- Add image-quality and out-of-distribution checks.
- Add explainability visualizations such as Grad-CAM.
- Store user history only with explicit consent and appropriate privacy controls.
- Create an offline mobile interface and support regional languages.
- Consult local agronomists to review class descriptions and crop-specific guidance.

## Conclusion

The project demonstrates an end-to-end AI workflow: labeled image data, transfer-learning model training, quantitative evaluation, a Flask inference API, and a responsive web interface. Its practical value depends on careful model evaluation and responsible interpretation of predictions. For agricultural decisions, the model should be used as an initial aid and not as the sole diagnostic authority.
