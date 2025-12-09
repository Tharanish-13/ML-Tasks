# app/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .utils import preprocess_image_bytes, load_model_by_name, load_auxiliary
import numpy as np
import io

app = FastAPI(title="Animal Classifier API")

# serve static frontend
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

scaler, label_map = None, None
try:
    scaler, label_map = load_auxiliary()
    print("Loaded scaler and label_map")
except Exception as e:
    print("Auxiliary files not loaded:", e)

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")

@app.post("/predict")
async def predict(model: str = Form(...), file: UploadFile = File(...)):
    """
    POST form-data:
      - model: 'logistic' | 'knn' | 'gaussian'
      - file: image
    Returns JSON: { prediction: label, class_id: int, confidences: {...} }
    """
    try:
        data = await file.read()
        img_bytes = io.BytesIO(data)
        x = preprocess_image_bytes(img_bytes)  # shape (1, D)
        if scaler is None:
            return JSONResponse({"error": "Scaler/label map not found. Train models and restart server."}, status_code=500)
        x_scaled = scaler.transform(x)
        model_obj = load_model_by_name(model)
        pred_id = int(model_obj.predict(x_scaled)[0])
        label = label_map.get(pred_id, str(pred_id))
        # if model has predict_proba:
        confidences = {}
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(x_scaled)[0]
            # build dict of label: prob
            confidences = {label_map.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
        return {"prediction": label, "class_id": pred_id, "confidences": confidences}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
