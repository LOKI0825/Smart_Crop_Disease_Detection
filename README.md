# LeafLens AI — Plant Disease Detection System

A full-stack, student-project implementation that classifies plant leaf images using transfer learning with MobileNetV2. The Flask website accepts a photo and displays the model's top three class predictions, confidence scores, and general next-step guidance.

> **Important:** This is an educational prototype, not a definitive agricultural diagnosis. The model only knows the classes represented in its training dataset. PlantVillage images commonly have controlled backgrounds, so performance on real field photos may be lower. Confirm high-impact decisions with a qualified local agricultural expert.

## Features

- Responsive web UI with image preview, drag-and-drop upload, and upload validation.
- Flask backend with `/predict` and `/api/health` endpoints.
- MobileNetV2 transfer-learning training pipeline.
- Stratified train / validation / test split (70% / 15% / 15%).
- Evaluation outputs: test accuracy/loss, per-class precision/recall/F1 CSV, and confusion-matrix image.
- Model predictions: top class, top-three matches, confidence scores, crop/category labels, and conservative general guidance.
- Image upload size limit of 10 MB; uploaded images are processed in memory and not saved to disk.

## 1. Requirements

- Python 3.10 or 3.11 recommended (64-bit).
- Internet connection for the first training run to download MobileNetV2 pretrained weights.
- PlantVillage dataset downloaded separately (roughly 1 GB or more depending on the distribution).
- 8 GB RAM recommended; a GPU is helpful but not mandatory. Full CPU training may take a long time.

## 2. Get the dataset

Use one of these sources:

- [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [PlantVillage dataset — original research repository](https://github.com/spMohanty/PlantVillage-Dataset)
- [PlantVillage dataset on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)
- [PlantVillage dataset on Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

For this project, use a folder-based download where each class is a folder and the images are directly inside that folder. Extract or copy it so the layout looks like this:

```text
plant_disease_detection/
├── app.py
├── train.py
├── data/
│   └── PlantVillage/
│       ├── Apple___Apple_scab/
│       │   ├── image_001.JPG
│       │   └── ...
│       ├── Apple___healthy/
│       ├── Tomato___Early_blight/
│       ├── Tomato___healthy/
│       └── ...
├── models/
├── reports/
├── src/
├── static/
└── templates/
```

The actual class names vary slightly between dataset distributions. That is okay: the script detects classes automatically. If your downloaded ZIP has an extra wrapper folder, the script tries to find the class-folder level automatically.

**Note:** Do not put all images in one folder. There must be one subfolder per label, with images inside the label folders.

## 3. Open in VS Code and install packages

Open the `plant_disease_detection` folder in VS Code. In the integrated terminal, run:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, open a Command Prompt terminal and use:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If dependency installation fails, check the Python version first; Python 3.10 or 3.11 is recommended.

## 4. Train the AI model

From the project root, run:

```bash
python train.py
```

The default command uses `data/PlantVillage`, trains a MobileNetV2 classification head for up to 10 epochs, then fine-tunes a small part of the network for up to 3 more epochs. Early stopping may finish training sooner.

Optional arguments:

```bash
python train.py --data-dir data/PlantVillage --epochs 8 --fine-tune-epochs 2 --batch-size 16
```

Use a smaller batch size (for example, `--batch-size 8`) if memory is limited. To skip fine-tuning:

```bash
python train.py --fine-tune-epochs 0
```

Training may take from several minutes to many hours depending on hardware and how much data is used. The pretrained MobileNetV2 weights are downloaded by TensorFlow on the first run, so an internet connection is needed then.

On successful completion, these files are created:

```text
models/
├── plant_disease_model.keras
└── classes.json
reports/
├── metrics.json
├── classification_report.csv
└── confusion_matrix.png
```

The held-out test metrics are saved to `reports/metrics.json`; do not claim a particular accuracy until you train and evaluate your own model.

## 5. Run the website

After training, run:

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Upload a clear plant-leaf photo and select **Analyze leaf**.

The website can start before training so you can review the UI, but prediction is intentionally unavailable until both `models/plant_disease_model.keras` and `models/classes.json` exist.

## Run the included utility tests

From the project root, run:

```bash
python -m unittest discover -s tests -v
```

These tests check label formatting and generic guidance helpers; they do not replace training-time model evaluation.

## API endpoints

### `GET /api/health`

Returns whether model files are present and how many classes are in `classes.json`.

### `POST /predict`

Send a multipart form field named `leaf_image` containing a JPG, PNG, WEBP, or BMP image. The response includes the predicted category, confidence score, top-three predictions, general guidance, and disclaimer. The maximum upload size is 10 MB.

Example with curl:

```bash
curl -X POST -F "leaf_image=@leaf.jpg" http://127.0.0.1:5000/predict
```

## Project structure

```text
plant_disease_detection/
├── app.py                     # Flask routes and upload validation
├── train.py                   # Dataset preparation, training, and evaluation
├── PROJECT_REPORT.md          # Project report/synopsis notes
├── requirements.txt           # Python dependencies
├── src/
│   └── predictor.py           # Load model, make predictions, format results
├── templates/
│   └── index.html              # Web interface
├── tests/
│   └── test_predictor_utils.py # Small offline utility tests
├── static/
│   ├── style.css               # Responsive styling
│   └── app.js                  # Preview, drag-and-drop, API call, result UI
├── data/                       # Put downloaded dataset here
├── models/                     # Generated model and class labels
└── reports/                    # Generated evaluation reports
```

## Methodology

1. Organize labeled leaf photos by class folder.
2. Split image paths into stratified 70% training, 15% validation, and 15% test sets.
3. Resize inputs to 224 × 224 pixels.
4. Apply modest training-time augmentation (flip, rotation, zoom, and contrast).
5. Train a MobileNetV2 model with ImageNet pretrained features, then optionally fine-tune its last layers.
6. Select a checkpoint based on validation accuracy and evaluate on the held-out test subset.
7. Serve the trained model through Flask.

Because a folder-based random split may place near-duplicate or related leaf images into different subsets in some dataset versions, test scores can be optimistic. For a research-grade evaluation, use the official leaf-group-aware split and keep all images from the same leaf/group in a single split.

## Troubleshooting

- **Dataset not found:** check `data/PlantVillage` and make sure it contains class folders, not only another ZIP file. You can override the location with `--data-dir`.
- **Not enough memory:** close other applications and reduce `--batch-size` to 8 or 16.
- **Model not trained:** run `python train.py` successfully before using the prediction function.
- **Model file missing or class mismatch:** keep `plant_disease_model.keras` and `classes.json` together in `models/`. Retrain if class folders change.
- **TensorFlow installation errors:** use a fresh Python 3.10 or 3.11 virtual environment, then reinstall `requirements.txt`.
- **Poor real-world predictions:** capture clear images in natural light, use a leaf that fills much of the photo, and remember that performance on PlantVillage-style images does not guarantee field accuracy.

## Dataset citation

If you use PlantVillage for a college project or report, cite the original paper:

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science, 7, 1419. https://doi.org/10.3389/fpls.2016.01419
