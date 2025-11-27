# app.py — merged: new model core + old API surface (health, predict)
import os
import time
import re
import html
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
import tf_keras as keras

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers.models.bert.modeling_tf_bert import TFBertModel

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "artifacts/final_model.keras"
TOKENIZER_PATH = "artifacts/tokenizer"
MAX_LEN = 65

# Suppress noisy TF logs if desired
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ------------------------------
# CLEANING
# ------------------------------
url_re   = re.compile(r'https?://\S+|www\.\S+')
html_rex = re.compile(r'<.*?>')
emoji_re = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

def clean_text_adv(s: str) -> str:
    s = html.unescape(str(s)).lower()
    s = url_re.sub(' ', s)
    s = html_rex.sub(' ', s)
    s = emoji_re.sub(' ', s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ------------------------------
# Globals (bundle)
# ------------------------------
model_bundle = None  # will hold dict: model, tokenizer, max_len, load_time, error/info

# ------------------------------
# Resource loader (used at startup)
# ------------------------------
def load_resources():
    """
    Load model and tokenizer. Returns a dict with keys:
    - model: keras model or None
    - tokenizer: tokenizer or None
    - max_len: int
    - load_time: seconds
    - error: None or str
    - model_path_exists: bool
    - tokenizer_path_exists: bool
    """
    bundle = {
        "model": None,
        "tokenizer": None,
        "max_len": MAX_LEN,
        "load_time": None,
        "error": None,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "tokenizer_path_exists": os.path.exists(TOKENIZER_PATH),
    }
    t0 = time.time()
    try:
        # Load model
        if bundle["model_path_exists"]:
            # Use custom_objects for TFBertModel if model contains it
            print("🔹 Loading model from", MODEL_PATH)
            try:
                model = keras.models.load_model(
                    MODEL_PATH,
                    custom_objects={"TFBertModel": TFBertModel},
                )
            except Exception as e:
                # Try load without custom_objects fallback (capture error)
                print("⚠️ load_model raised, capturing error:", e)
                raise
            bundle["model"] = model
            print("✅ Model loaded")
        else:
            raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

        # Load tokenizer (prefer local)
        if bundle["tokenizer_path_exists"]:
            print("🔹 Loading tokenizer from", TOKENIZER_PATH)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, local_files_only=True)
            bundle["tokenizer"] = tokenizer
            tokenizer_source = "local"
            print("✅ Tokenizer loaded (local)")
        else:
            # Try to load tokenizer online as fallback
            print("⚠️ Tokenizer local dir not found; trying online fallback")
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
            bundle["tokenizer"] = tokenizer
            tokenizer_source = "online"
            print("✅ Tokenizer loaded (online)")

        bundle["tokenizer_source"] = tokenizer_source
        bundle["load_time"] = time.time() - t0
        bundle["error"] = None
        return bundle

    except Exception as e:
        bundle["load_time"] = time.time() - t0
        bundle["error"] = str(e)
        print("❌ Error loading resources:", bundle["error"])
        # Optionally print traceback for debugging in logs
        import traceback
        traceback.print_exc()
        return bundle

# ------------------------------
# Predict util (uses model_bundle)
# ------------------------------
def predict_duplicate_pair(q1: str, q2: str):
    if model_bundle is None or model_bundle.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    max_len = model_bundle["max_len"]

    q1_clean = clean_text_adv(q1)
    q2_clean = clean_text_adv(q2)

    enc = tokenizer(
        q1_clean,
        q2_clean,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )

    # ensure keys exist
    input_ids = enc.get("input_ids")
    attention_mask = enc.get("attention_mask")
    token_type_ids = enc.get("token_type_ids")

    # Some tokenizers backbones may not return token_type_ids; prepare zeros if needed
    if token_type_ids is None:
        token_type_ids = tf.zeros_like(input_ids)

    pred = model.predict([input_ids, attention_mask, token_type_ids], verbose=0)
    prob = float(np.asarray(pred).ravel()[0])
    label = "duplicate" if prob >= 0.5 else "not duplicate"
    return prob, label

# ------------------------------
# FastAPI: lifespan to load resources once at startup
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    print("⏳ Starting app, loading resources...")
    model_bundle = load_resources()
    if model_bundle.get("model") is None:
        print("⚠️ Model not available at startup:", model_bundle.get("error"))
    else:
        print("✅ Resources loaded successfully")
    yield
    print("🔄 Shutting down app")

app = FastAPI(title="Duplicate Question Detector API", version="1.0.0", lifespan=lifespan)

# ------------------------------
# Schemas
# ------------------------------
class PredictRequest(BaseModel):
    question1: str
    question2: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: dict | None = None

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/", tags=["root"])
def root():
    return {"message": "Duplicate Question Detector API is running!", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    ok = (model_bundle is not None) and (model_bundle.get("model") is not None)
    info = None
    if model_bundle:
        info = {
            "model_path_exists": model_bundle.get("model_path_exists"),
            "tokenizer_path_exists": model_bundle.get("tokenizer_path_exists"),
            "load_time": model_bundle.get("load_time"),
            "tokenizer_source": model_bundle.get("tokenizer_source", None),
            "error": model_bundle.get("error"),
        }
        # Try to include simple model shape info if model present
        m = model_bundle.get("model")
        if m is not None:
            try:
                info["model_inputs"] = [str(inp.shape) for inp in m.inputs] if hasattr(m, "inputs") else None
                info["model_outputs"] = [str(out.shape) for out in m.outputs] if hasattr(m, "outputs") else None
            except Exception:
                pass
    status = "healthy" if ok else "unhealthy"
    return HealthResponse(status=status, model_loaded=ok, model_info=info)

@app.post("/predict", tags=["predict"])
def predict(req: PredictRequest):
    start = time.time()
    try:
        prob, label = predict_duplicate_pair(req.question1, req.question2)
        elapsed = time.time() - start
        return {
            "question1": req.question1,
            "question2": req.question2,
            "probability": prob,
            "prediction": label,
            "processing_time": elapsed,
            "weights_loaded": model_bundle.get("model") is not None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ------------------------------
# Run (when executed directly)
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Uvicorn server (app:app)")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
