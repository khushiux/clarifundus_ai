from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retinatrust.config import load_config, resolve_path
from retinatrust.inference.predictor import RetinaPredictor
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.api")

predictor_instance: Optional[RetinaPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor_instance
    try:
        cfg = load_config()
        ckpt_dir = resolve_path(cfg.paths.checkpoint_dir)
        available = list(ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
        if available:
            logger.info(f"Loading RetinaPredictor on API startup from {available[0]}...")
            predictor_instance = RetinaPredictor(checkpoint_path=str(available[0]))
        else:
            logger.warning(f"No checkpoint found in {ckpt_dir}. Predictor will be unavailable until trained.")
    except Exception as e:
        logger.error(f"Error initializing predictor during startup: {e}")
    yield
    predictor_instance = None

app = FastAPI(
    title="RetinaTrust 2.0 AI API",
    description="Explainable, Uncertainty-Bounded & Conformal Ophthalmic Decision Support Service",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('online')")
    model_loaded: bool = Field(..., description="Whether trained model checkpoint is in memory")
    architecture: Optional[str] = Field(None, description="Model architecture name")
    device: Optional[str] = Field(None, description="Execution device (cuda or cpu)")
    calibration_temperature: Optional[float] = Field(None, description="Learned post-hoc scaling factor")

class PredictionResponse(BaseModel):
    predicted_grade: int = Field(..., description="Diabetic retinopathy severity grade (0 to 4)")
    grade_label: str = Field(..., description="Short grade label (e.g. 'Moderate DR')")
    grade_description: str = Field(..., description="Clinical grade definition from ICDR scale")
    estimated_confidence: float = Field(..., description="Calibrated softmax confidence score")
    predictive_entropy: float = Field(..., description="Predictive Shannon entropy in bits")
    is_uncertain: bool = Field(..., description="True if prediction flagged as ambiguous or low-confidence")
    uncertainty_reasons: List[str] = Field(default_factory=list, description="Explanations if flagged as uncertain")
    class_probabilities: Dict[str, float] = Field(..., description="Probability distribution over 5 grades")
    conformal: Dict[str, Any] = Field(..., description="Conformal Prediction finite-sample coverage set & triage")
    quality_warnings: List[str] = Field(default_factory=list, description="Technical heuristic image warnings")
    is_quality_acceptable: bool = Field(..., description="Whether image passed technical heuristics")
    calibration: Dict[str, Any] = Field(..., description="Calibration parameters and method")
    architecture: str = Field(..., description="Neural network architecture used")
    gradcam_available: bool = Field(..., description="Whether Grad-CAM++ explanation was generated")
    disclaimer: str = Field(..., description="Medical prototype legal disclaimer")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint reporting service, model, and device status."""
    if predictor_instance is not None:
        return HealthResponse(
            status="online",
            model_loaded=True,
            architecture=predictor_instance.architecture,
            device=str(predictor_instance.device),
            calibration_temperature=predictor_instance.temperature,
        )
    return HealthResponse(
        status="online",
        model_loaded=False,
        architecture=None,
        device=None,
        calibration_temperature=None,
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"description": "Invalid image file format or corrupt image"},
        413: {"description": "File size exceeds 20 MB limit"},
        503: {"description": "Model checkpoint not loaded or unavailable"},
    },
)
async def predict_retinopathy(
    file: UploadFile = File(...),
    use_ben_graham: bool = True,
):
    """Accepts a retinal fundus photograph upload (JPG/PNG <= 20MB) and returns

    calibrated DR severity, Conformal 95% safety sets, entropy, and triage guidance.
    """
    if predictor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RetinaTrust inference model is not loaded. Train a model first.",
        )

    filename = file.filename or "uploaded_image.png"
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if file_ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed formats: {sorted(list(valid_exts))}",
        )

    MAX_BYTES = 20 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds 20 MB limit.",
        )

    try:
        result = predictor_instance.predict(
            image_input=content,
            generate_cam=True,
            apply_ben_graham=use_ben_graham,
        )
        return PredictionResponse(**result.to_dict(include_images=False))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference execution error.")

@app.post(
    "/predict/report",
    responses={
        200: {"content": {"application/pdf": {}}, "description": "Downloadable Clinical PDF Report"},
        400: {"description": "Bad Request"},
        503: {"description": "Service Unavailable"},
    },
)
async def generate_screening_pdf(
    file: UploadFile = File(...),
    use_ben_graham: bool = True,
):
    """Uploads a fundus image and generates a downloadable Clinical PDF Screening Audit Report."""
    if predictor_instance is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")

    content = await file.read()
    try:
        result = predictor_instance.predict(
            image_input=content,
            generate_cam=True,
            apply_ben_graham=use_ben_graham,
        )
        pdf_bytes = result.generate_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="RetinaTrust_Clinical_Report.pdf"'},
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not generate PDF report.")
