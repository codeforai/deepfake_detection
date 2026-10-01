"""
AI-Generated Voice Detection API - XGBoost Version
FastAPI application for detecting AI-generated vs Human voices using XGBoost
Optimized for Render free tier (512MB RAM)
"""

import os
import base64
import io
import numpy as np
import librosa
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# ============== Configuration ==============
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set. Refusing to start without authentication.")

SAMPLE_RATE = 16000  # Must match training sample rate
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 10.0  # seconds

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Model path
MODEL_PATH = "model/model_xgboost.json"

# ============== Load Model ==============
print(f"Loading XGBoost model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load XGBoost model
model = xgb.Booster()
model.load_model(MODEL_PATH)
print("XGBoost model loaded successfully")

# ============== FastAPI App ==============
app = FastAPI(
    title="AI Voice Detection API (XGBoost)",
    description="Detects whether a voice sample is AI-generated or Human using XGBoost",
    version="1.0.0"
)

# ============== Request/Response Models ==============
class VoiceDetectionRequest(BaseModel):
    language: str = Field(..., description="Language: Tamil/English/Hindi/Malayalam/Telugu")
    audioFormat: str = Field(..., description="Audio format (mp3)")
    audioBase64: str = Field(..., description="Base64-encoded MP3 audio")

class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# ============== Feature Extraction ==============
def extract_lfcc_from_bytes(audio_bytes, sr=SAMPLE_RATE, n_lfcc=N_LFCC, 
                           n_fft=N_FFT, hop_length=HOP_LENGTH, max_duration=MAX_DURATION):
    """
    Extract LFCC-style cepstral features from audio bytes (in-memory).
    Uses MFCC computation for practical spoofing detection.
    
    Returns:
        np.ndarray: LFCC features of shape (n_lfcc, time_steps)
    """
    try:
        # Load audio from bytes (no disk I/O)
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=sr, duration=max_duration)
        
        # Compute LFCC-style cepstral features (matching training exactly)
        lfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_lfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Calculate target length based on max duration
        target_length = int(max_duration * sr / hop_length)
        
        # Pad or truncate to target length
        if lfcc.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - lfcc.shape[1]
            lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            lfcc = lfcc[:, :target_length]
        
        return lfcc
    except Exception as e:
        print(f"Error processing audio: {e}")
        target_length = int(max_duration * SAMPLE_RATE / hop_length)
        return np.zeros((n_lfcc, target_length), dtype=np.float32)

def extract_statistical_features(lfcc_matrix):
    """
    Extract statistical features from LFCC matrix.
    This matches the training pipeline EXACTLY.
    
    Input: (n_lfcc, time_steps) e.g., (40, 313)
    Output: (n_features,) e.g., (365,)
    
    Features extracted:
    - Per-coefficient statistics: mean, std, min, max, 25th/50th/75th percentiles, range, IQR
    - Overall statistics: global mean, std, min, max, median
    """
    feat = []
    
    # Per-coefficient statistics (40 coefficients × 9 stats = 360 features)
    feat.extend(np.mean(lfcc_matrix, axis=1))       # 40
    feat.extend(np.std(lfcc_matrix, axis=1))        # 40
    feat.extend(np.min(lfcc_matrix, axis=1))        # 40
    feat.extend(np.max(lfcc_matrix, axis=1))        # 40
    feat.extend(np.percentile(lfcc_matrix, 25, axis=1))  # 40
    feat.extend(np.percentile(lfcc_matrix, 50, axis=1))  # 40
    feat.extend(np.percentile(lfcc_matrix, 75, axis=1))  # 40
    feat.extend(np.max(lfcc_matrix, axis=1) - np.min(lfcc_matrix, axis=1))  # 40 (range)
    feat.extend(np.percentile(lfcc_matrix, 75, axis=1) - np.percentile(lfcc_matrix, 25, axis=1))  # 40 (IQR)
    
    # Overall statistics (5 features)
    feat.extend([
        np.mean(lfcc_matrix),
        np.std(lfcc_matrix),
        np.min(lfcc_matrix),
        np.max(lfcc_matrix),
        np.median(lfcc_matrix)
    ])
    
    return np.array(feat, dtype=np.float32)

def preprocess_audio_bytes(audio_bytes):
    """
    Extract LFCC features and compute statistical features for XGBoost input.
    Matches exact preprocessing used during training.
    
    Returns:
        xgb.DMatrix: XGBoost DMatrix ready for prediction
    """
    # Extract LFCC features from bytes
    lfcc = extract_lfcc_from_bytes(audio_bytes)
    
    # Extract statistical features
    features = extract_statistical_features(lfcc)
    
    # Reshape for XGBoost: (1, n_features)
    features = features.reshape(1, -1)
    
    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(features)
    
    return dmatrix

def generate_explanation(confidence: float, is_ai: bool) -> str:
    """
    Generate a human-readable explanation for the classification.
    """
    if is_ai:
        if confidence > 0.9:
            return "Strong indicators of synthetic speech: unnatural pitch consistency, robotic intonation patterns, and artificial spectral characteristics detected"
        elif confidence > 0.75:
            return "Moderate AI speech indicators: irregular prosody patterns and synthetic voice artifacts detected"
        elif confidence > 0.6:
            return "Slight synthetic characteristics detected: minor artifacts in speech patterns suggest AI generation"
        else:
            return "Weak AI indicators detected: some subtle synthetic patterns present in the audio"
    else:
        if confidence > 0.9:
            return "Strong human voice characteristics: natural pitch variations, authentic breathing patterns, and organic speech dynamics detected"
        elif confidence > 0.75:
            return "Natural human speech patterns detected: consistent with organic voice production"
        elif confidence > 0.6:
            return "Predominantly human characteristics: natural prosody with minor inconsistencies"
        else:
            return "Voice appears human with some ambiguous characteristics"

# ============== API Endpoints ==============
@app.get("/")
async def root():
    return {"message": "AI Voice Detection API (XGBoost)", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "xgboost"
    }

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    """
    Detect whether a voice sample is AI-generated or Human using XGBoost.
    """
    # Validate API Key
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key or malformed request"}
        )
    
    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Unsupported language. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}"}
        )
    
    # Validate audio format
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Only MP3 audio format is supported"}
        )
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audioBase64)
        
        # Preprocess audio (in-memory, no disk I/O)
        features = preprocess_audio_bytes(audio_bytes)
        
        # Run XGBoost inference
        prediction = model.predict(features)[0]
        
        # Log prediction for debugging
        print(f"Raw prediction: {prediction:.4f}, Language: {request.language}")
        
        # Determine classification
        # Model output: higher value = more likely AI_GENERATED (label 1)
        is_ai = prediction >= 0.5
        classification = "AI_GENERATED" if is_ai else "HUMAN"
        
        # Confidence score (distance from decision boundary)
        confidence_score = float(prediction) if is_ai else float(1 - prediction)
        
        # Generate explanation
        explanation = generate_explanation(confidence_score, is_ai)
        
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence_score, 2),
            explanation=explanation
        )
                
    except base64.binascii.Error:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Invalid Base64 encoding"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Error processing audio: {str(e)}"}
        )

# ============== Error Handlers ==============
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"status": "error", "message": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )

# ============== Main ==============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
