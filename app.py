"""Flask web application for plant leaf disease classification."""
from __future__ import annotations

import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from src.predictor import ModelNotReadyError, predict_pil_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "classes.json"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


def model_status() -> tuple[bool, int]:
    """Return whether trained artifacts exist and how many classes are saved."""
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return False, 0
    try:
        import json

        with CLASSES_PATH.open("r", encoding="utf-8") as file:
            names = json.load(file)
        return True, len(names) if isinstance(names, list) else 0
    except (OSError, ValueError):
        return False, 0


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    ready, class_count = model_status()
    return render_template("index.html", model_ready=ready, class_count=class_count)


@app.get("/api/health")
def health():
    ready, class_count = model_status()
    return jsonify({
        "status": "ready" if ready else "model_not_trained",
        "model_available": ready,
        "class_count": class_count,
        "message": "Model artifacts found." if ready else "Train the model by following README.md.",
    })


@app.post("/predict")
def predict():
    uploaded = request.files.get("leaf_image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a leaf image before running the analysis."}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Unsupported file type. Upload a JPG, PNG, WEBP, or BMP image."}), 400

    try:
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({"error": "The uploaded file is empty. Please choose another image."}), 400
        with Image.open(io.BytesIO(image_bytes)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "This file could not be read as an image. Please upload a valid leaf photo."}), 400

    try:
        result = predict_pil_image(image)
        return jsonify(result)
    except ModelNotReadyError as exc:
        return jsonify({"error": str(exc), "code": "MODEL_NOT_READY"}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The image could not be analyzed. Check the model files and server log, then try again."}), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "The image is larger than 10 MB. Please resize it and try again."}), 413


if __name__ == "__main__":
    # Local development only. For deployment, use a production WSGI server.
    app.run(host="127.0.0.1", port=5000, debug=True)
