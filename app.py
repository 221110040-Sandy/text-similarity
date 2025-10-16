# model_api.py - FastAPI Backend for Text Similarity (Functional + LIST inputs)

# ========= Environment (set sebelum import apa pun) =========
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # pakai tf.keras (penting jika keras==3 terpasang)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tokenizer thread safety
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # suppress TF INFO logs
# NOTE: Kalau full offline, biarkan "1" dan siapkan artifacts lokal (minilm-tf + tokenizer).
# Jika perlu download sekali dari internet, set "0" sementara.
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import re
from typing import Dict, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ========= Global storage =========
model_bundle = None

# ========= Config (harus sama dengan training/inference) =========
MAX_LEN = 59
BILSTM_NUMBER = 100
ATTENTION_NUMBER = 100
DENSE_SIZES = [64, 32]
DROPOUT = 0.4
L2_LAMBDA = 1e-4

ART_DIR = "artifacts"
TOKENIZER_DIR = os.path.join(ART_DIR, "tokenizer")
MINI_DIR = os.path.join(ART_DIR, "minilm-tf")  # optional TF backbone cache untuk offline
WEIGHT_BEST = os.path.join(ART_DIR, "best_model.weights.h5")  # atau final_model.weights.h5
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"


# ========= Utils =========
def clean_text(text):
    """Text cleaning function - same as training script."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_neural_model():
    """
    Load model & tokenizer (Functional + LIST inputs) identik dengan skrip STS Dokumen.
    Return bundle dict untuk runtime.
    """
    print("🧠 Loading neural model...")
    print(f"📁 CWD: {os.getcwd()}")
    t0 = time.time()

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from transformers import AutoTokenizer, TFAutoModel, AutoConfig

        # ---- Functional architecture (nama layer sama persis training) ----
        inp_ids = keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
        inp_msk = keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")
        inp_seg = keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="token_type_ids")

        # ---- BERT backbone (frozen) ----
        bert = None
        if os.path.exists(MINI_DIR):
            try:
                cfg = AutoConfig.from_pretrained(
                    MINI_DIR,
                    local_files_only=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                bert = TFAutoModel.from_pretrained(
                    MINI_DIR, config=cfg, local_files_only=True, name="bert_backbone"
                )
                print("✅ BERT loaded from artifacts/minilm-tf (offline)")
            except Exception as e:
                print(f"⚠️ Failed to load BERT from {MINI_DIR}: {e}")

        if bert is None:
            # Jika tidak ada backbone lokal & offline aktif → stop dengan pesan jelas
            if os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1":
                raise FileNotFoundError(
                    "TRANSFORMERS_OFFLINE=1, tapi artifacts/minilm-tf tidak ada. "
                    "Sediakan backbone lokal atau set TRANSFORMERS_OFFLINE=0 untuk fallback online."
                )
            cfg = AutoConfig.from_pretrained(
                MODEL_NAME, output_hidden_states=False, output_attentions=False
            )
            bert = TFAutoModel.from_pretrained(
                MODEL_NAME, config=cfg, from_pt=True, name="bert_backbone"
            )
            print("✅ BERT loaded online (fallback)")

        bert.trainable = False
        # Panggil backbone dengan keyword args (bukan list)
        bert_out = bert(
            input_ids=inp_ids,
            attention_mask=inp_msk,
            token_type_ids=inp_seg,
            training=False,
        ).last_hidden_state  # (B, T, H)

        # ---- LN -> BiLSTM (mask-aware) ----
        x = layers.LayerNormalization(epsilon=1e-6, name="ln_after_bert")(bert_out)
        mask_bool = tf.cast(inp_msk, tf.bool)
        x = layers.Bidirectional(
            layers.LSTM(BILSTM_NUMBER, return_sequences=True), name="bilstm"
        )(x, mask=mask_bool)  # (B,T,H')

        # ---- Additive attention ----
        score = layers.Dense(ATTENTION_NUMBER, activation="tanh", name="attn_tanh")(x)  # (B,T,A)
        score = layers.Dense(1, name="attn_score")(score)                               # (B,T,1)

        mask_exp = tf.expand_dims(mask_bool, axis=-1)                                   # (B,T,1)
        neg_inf = tf.fill(tf.shape(score), tf.cast(-1e9, score.dtype))                  # (B,T,1)
        score_m = tf.where(mask_exp, score, neg_inf)                                    # (B,T,1)

        alphas = layers.Softmax(axis=1, name="attn_softmax")(score_m)                   # (B,T,1)
        context = tf.reduce_sum(x * alphas, axis=1)                                     # (B,H')

        # ---- Dense head + L2 regularizer (match training) ----
        REG = keras.regularizers.l2(L2_LAMBDA)
        h = context
        for i, size in enumerate(DENSE_SIZES, 1):
            h = layers.Dense(size, activation="relu", kernel_regularizer=REG, name=f"dense_hidden{i}")(h)
            h = layers.Dropout(DROPOUT, name=f"dropout{i}")(h)

        out = layers.Dense(1, activation="sigmoid", name="similarity")(h)
        model = keras.Model(inputs=[inp_ids, inp_msk, inp_seg], outputs=out, name="sts_cross_encoder")

        # ---- Warm-up (shape sanity) ----
        _ = model(
            [
                tf.zeros((1, MAX_LEN), dtype=tf.int32),
                tf.zeros((1, MAX_LEN), dtype=tf.int32),
                tf.zeros((1, MAX_LEN), dtype=tf.int32),
            ],
            training=False,
        )

        # ---- Load weights (.h5 weights-only) ----
        weights_loaded = False
        if os.path.exists(WEIGHT_BEST):
            try:
                model.load_weights(WEIGHT_BEST)
                weights_loaded = True
                print(f"✅ Weights loaded: {WEIGHT_BEST}")
            except Exception as e:
                try:
                    model.load_weights(WEIGHT_BEST, by_name=True, skip_mismatch=True)
                    weights_loaded = True
                    print(f"✅ Weights loaded by_name (skip_mismatch): {WEIGHT_BEST}")
                except Exception as e2:
                    print(f"⚠️ Failed to load weights: {e2}")
        else:
            print(f"⚠️ Weight file not found: {WEIGHT_BEST}")

        # ---- Tokenizer (prefer offline) ----
        local_tok = os.path.exists(TOKENIZER_DIR)
        if not local_tok and os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1":
            raise FileNotFoundError(
                f"Tokenizer dir not found: {TOKENIZER_DIR} (offline mode). "
                f"Sediakan tokenizer lokal atau set TRANSFORMERS_OFFLINE=0."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_DIR if local_tok else MODEL_NAME,
            use_fast=True,
            local_files_only=local_tok,
        )
        # extend max length untuk windowing manual
        tokenizer.model_max_length = 10_000_000
        if hasattr(tokenizer, "init_kwargs"):
            tokenizer.init_kwargs["model_max_length"] = 10_000_000

        load_time = time.time() - t0
        print(f"✅ Model ready in {load_time:.2f}s (weights_loaded={weights_loaded})")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "max_len": MAX_LEN,
            "weights_loaded": weights_loaded,
            "load_time": load_time,
            "tokenizer_source": "artifacts/tokenizer" if local_tok else "online",
        }

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def predict_neural_similarity(text1: str, text2: str) -> float:
    """Predict pairwise similarity menggunakan neural model."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import tensorflow as tf

        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        max_len = model_bundle["max_len"]

        # Clean & truncate by words (aman)
        def clean_and_truncate(text, max_words=120):
            cleaned = clean_text(text)
            words = cleaned.split()
            return " ".join(words[:max_words])

        text1_clean = clean_and_truncate(text1)
        text2_clean = clean_and_truncate(text2)

        # Tokenize
        encoded = tokenizer(
            text1_clean,
            text2_clean,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="tf",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        inputs = [
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded.get("token_type_ids", tf.zeros_like(encoded["input_ids"])),
        ]

        # Predict
        prediction = model(inputs, training=False)
        similarity = float(prediction.numpy().ravel()[0])
        return similarity

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============ Document Similarity (Sliding Window + BERTScore-like) ============
def make_windows(text: str, tokenizer, per_side: int, stride: int):
    """Create sliding windows dari raw text (berbasis token piece)."""
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    if not ids:
        return [""]

    wins = []
    s = 0
    n = len(ids)
    while s < n:
        e = min(s + per_side, n)
        if s >= e:
            break
        win = tokenizer.decode(ids[s:e], clean_up_tokenization_spaces=True).strip()
        wins.append(win)
        if e == n:
            break
        s += stride
    return wins


def score_matrix_batch(model, tokenizer, winsA, winsB, max_len=MAX_LEN, batch_size=512):
    """Score semua pasangan window dalam batch."""
    import tensorflow as tf
    import numpy as np

    m, n = len(winsA), len(winsB)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float32)

    # All pairs
    pairs = [(i, j) for i in range(m) for j in range(n)]
    a_texts = [winsA[i] for (i, _) in pairs]
    b_texts = [winsB[j] for (_, j) in pairs]

    # Encode sekali untuk semua
    enc = tokenizer(
        a_texts,
        b_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="tf",
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    ids = enc["input_ids"]
    msk = enc["attention_mask"]
    seg = enc.get("token_type_ids", tf.zeros_like(ids))

    # Predict in batches
    probs = []
    for s in range(0, ids.shape[0], batch_size):
        e = min(s + batch_size, ids.shape[0])
        batch_input = [
            ids[s:e],
            msk[s:e],
            seg[s:e],
        ]
        y = model(batch_input, training=False)
        probs.append(tf.reshape(y, [-1]).numpy())

    import numpy as np
    probs = np.concatenate(probs)

    # Sim matrix m x n
    S = np.zeros((m, n), dtype=np.float32)
    for (i, j), score in zip(pairs, probs):
        S[i, j] = float(score)

    return S


def bertscore_from_matrix(S):
    """Hitung P/R/F1 ala BERTScore dari matriks similarity."""
    import numpy as np

    if S.size == 0:
        return {"P": 0.0, "R": 0.0, "F1": 0.0}

    row_max = S.max(axis=1) if S.shape[0] > 0 else np.array([], dtype=np.float32)
    R = float(row_max.mean()) if row_max.size else 0.0

    col_max = S.max(axis=0) if S.shape[1] > 0 else np.array([], dtype=np.float32)
    P = float(col_max.mean()) if col_max.size else 0.0

    F1 = 0.0 if (P + R) == 0.0 else (2.0 * P * R) / (P + R)
    return {"P": P, "R": R, "F1": F1}


def aggregate_bertscore(S, symmetric=True):
    """Aggregate P/R/F1 (symmetrical atau directional)."""
    if S.size == 0:
        return {"doc_score": 0.0, "detail": {"AtoB": {}, "BtoA": {}, "symmetric": symmetric}}

    AtoB = bertscore_from_matrix(S)

    if symmetric:
        BtoA = bertscore_from_matrix(S.T)
        doc_score = 0.5 * (AtoB["F1"] + BtoA["F1"])
        return {"doc_score": float(doc_score), "detail": {"AtoB": AtoB, "BtoA": BtoA, "symmetric": True}}
    else:
        return {"doc_score": float(AtoB["F1"]), "detail": {"AtoB": AtoB, "symmetric": False}}


def score_two_documents(doc1: str, doc2: str, per_side: int, stride: int, topk: int, symmetric: bool):
    """Pipeline lengkap untuk dua dokumen."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        max_len = model_bundle["max_len"]

        # 1) Windowing
        winsA = make_windows(doc1, tokenizer, per_side, stride)
        winsB = make_windows(doc2, tokenizer, per_side, stride)
        print(f"Document windowing: A={len(winsA)} windows, B={len(winsB)} windows")

        # 2) Scoring pasangan window
        S = score_matrix_batch(model, tokenizer, winsA, winsB, max_len, batch_size=512)

        # 3) Aggregate
        agg = aggregate_bertscore(S, symmetric=symmetric)

        # 4) Evidence teratas
        m, n = S.shape
        flat = [(i, j, float(S[i, j])) for i in range(m) for j in range(n)]
        flat.sort(key=lambda x: x[2], reverse=True)

        evidence = [{
            "i": int(i),
            "j": int(j),
            "score": float(s),
            "windowA": winsA[i][:200],
            "windowB": winsB[j][:200],
        } for i, j, s in flat[:min(topk, len(flat))]]

        return {
            "doc_score": agg["doc_score"],
            "detail": agg["detail"],
            "top_evidence": evidence,
            "shape": {"m": len(winsA), "n": len(winsB)},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document similarity error: {str(e)}")


# ========= FastAPI (lifespan) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    try:
        model_bundle = load_neural_model()
        print("🚀 FastAPI server ready with loaded model!")
    except Exception as e:
        print(f"⚠️ Failed to load neural model: {e}")
        print("📝 Server cannot run without neural model")
        model_bundle = None

    yield
    print("🔄 FastAPI server shutting down")


app = FastAPI(
    title="Text Similarity API",
    description="Neural text similarity using MiniLM + BiLSTM + Attention (Functional, weights-only).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Schemas =========
class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class DocumentSimilarityRequest(BaseModel):
    doc1: str
    doc2: str
    per_side_len: int = 28  # (59-3)//2
    stride: int = 28
    topk_evidence: int = 5
    use_symmetric: bool = True


class SimilarityResponse(BaseModel):
    similarity: float
    processing_time: float
    method: str
    weights_loaded: Optional[bool] = None
    error: Optional[str] = None


class DocumentSimilarityResponse(BaseModel):
    doc_score: float
    processing_time: float
    detail: Dict
    top_evidence: list
    shape: Dict
    method: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None


# ========= Endpoints =========
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check + diagnostics."""
    if model_bundle is None:
        artifacts_info = {}
        if os.path.exists(ART_DIR):
            artifacts_info["artifacts_dir_exists"] = True
            artifacts_info["artifacts_contents"] = os.listdir(ART_DIR)
        else:
            artifacts_info["artifacts_dir_exists"] = False

        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_info={
                "error": "Model failed to load",
                "artifacts_info": artifacts_info,
                "working_directory": os.getcwd(),
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_info={
            "weights_loaded": model_bundle["weights_loaded"],
            "max_len": model_bundle["max_len"],
            "load_time": model_bundle["load_time"],
            "tokenizer_source": model_bundle.get("tokenizer_source", "unknown"),
        },
    )


@app.post("/predict", response_model=SimilarityResponse)
async def predict_similarity(request: SimilarityRequest):
    """Prediksi similarity untuk dua teks pendek menggunakan Neural model."""
    start_time = time.time()
    try:
        if model_bundle is None:
            raise HTTPException(status_code=503, detail="Neural model not available")

        sim = predict_neural_similarity(request.text1, request.text2)
        dt = time.time() - start_time
        return SimilarityResponse(
            similarity=sim,
            processing_time=dt,
            method="Neural (MiniLM + BiLSTM + Attention)",
            weights_loaded=model_bundle["weights_loaded"],
        )

    except HTTPException:
        raise
    except Exception as e:
        dt = time.time() - start_time
        return SimilarityResponse(similarity=0.0, processing_time=dt, method="Error", error=str(e))


@app.post("/predict-document", response_model=DocumentSimilarityResponse)
async def predict_document_similarity(request: DocumentSimilarityRequest):
    """Prediksi similarity dokumen panjang dengan sliding window + BERTScore-like."""
    start_time = time.time()
    try:
        if model_bundle is None:
            raise HTTPException(status_code=503, detail="Model not available")

        result = score_two_documents(
            doc1=request.doc1,
            doc2=request.doc2,
            per_side=request.per_side_len,
            stride=request.stride,
            topk=request.topk_evidence,
            symmetric=request.use_symmetric,
        )
        dt = time.time() - start_time
        return DocumentSimilarityResponse(
            doc_score=result["doc_score"],
            processing_time=dt,
            detail=result["detail"],
            top_evidence=result["top_evidence"],
            shape=result["shape"],
            method="Document Similarity (Sliding Window + BERTScore)",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Text Similarity API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"},
    }


if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print("📁 Ensure 'artifacts/' contains:")
    print("   - best_model.weights.h5   (or final_model.weights.h5)")
    print("   - tokenizer/              (HF tokenizer files)")
    print("   - minilm-tf/              (optional if offline; TF backbone cache)")
    print("🌐 API:  http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")

    uvicorn.run(
        "model_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # True untuk dev hot-reload (disarankan 1 worker)
        workers=1,
    )
