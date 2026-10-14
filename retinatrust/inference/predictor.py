import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from retinatrust.config import AppConfig, load_config, resolve_path
from retinatrust.data.fundus_preprocessing import ben_graham_preprocessing
from retinatrust.data.transforms import get_eval_transforms
from retinatrust.inference.conformal import ConformalPredictor, ConformalPredictionResult
from retinatrust.inference.explainability import generate_explanation
from retinatrust.inference.image_quality import check_image_quality
from retinatrust.inference.report_generator import generate_pdf_report
from retinatrust.inference.uncertainty import quantify_uncertainty
from retinatrust.models.build_model import build_model, get_device
from retinatrust.models.coral import coral_logits_to_probs
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.predictor")

GRADE_DESCRIPTIONS = {
    0: "No Diabetic Retinopathy (Healthy retina with no visible microvascular lesions)",
    1: "Mild Diabetic Retinopathy (Microaneurysms only)",
    2: "Moderate Diabetic Retinopathy (More than microaneurysms, but less than severe NPDR)",
    3: "Severe Diabetic Retinopathy (Extensive hemorrhages, venous beading, or IRMA)",
    4: "Proliferative Diabetic Retinopathy (Neovascularization, vitreous/preretinal hemorrhage)",
}

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

class PredictionResult:
    """Comprehensive result returned by RetinaPredictor."""

    def __init__(
        self,
        predicted_grade: int,
        grade_label: str,
        grade_description: str,
        confidence: float,
        predictive_entropy: float,
        is_uncertain: bool,
        uncertainty_reasons: List[str],
        class_probabilities: Dict[int, float],
        conformal_result: ConformalPredictionResult,
        quality_warnings: List[str],
        is_quality_acceptable: bool,
        temperature: float,
        is_calibrated: bool,
        architecture: str,
        gradcam_available: bool = False,
        gradcam_overlay: Optional[np.ndarray] = None,
        gradcam_heatmap: Optional[np.ndarray] = None,
        ben_graham_image: Optional[np.ndarray] = None,
        original_image_np: Optional[np.ndarray] = None,
    ):
        self.predicted_grade = predicted_grade
        self.grade_label = grade_label
        self.grade_description = grade_description
        self.confidence = confidence
        self.predictive_entropy = predictive_entropy
        self.is_uncertain = is_uncertain
        self.uncertainty_reasons = uncertainty_reasons
        self.class_probabilities = class_probabilities
        self.conformal_result = conformal_result
        self.quality_warnings = quality_warnings
        self.is_quality_acceptable = is_quality_acceptable
        self.temperature = temperature
        self.is_calibrated = is_calibrated
        self.architecture = architecture
        self.gradcam_available = gradcam_available
        self.gradcam_overlay = gradcam_overlay
        self.gradcam_heatmap = gradcam_heatmap
        self.ben_graham_image = ben_graham_image
        self.original_image_np = original_image_np

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        result = {
            "predicted_grade": self.predicted_grade,
            "grade_label": self.grade_label,
            "grade_description": self.grade_description,
            "estimated_confidence": round(self.confidence, 4),
            "predictive_entropy": round(self.predictive_entropy, 4),
            "is_uncertain": self.is_uncertain,
            "uncertainty_reasons": self.uncertainty_reasons,
            "class_probabilities": {
                f"{k} - {CLASS_NAMES.get(k, f'Grade {k}')}": round(v, 4)
                for k, v in self.class_probabilities.items()
            },
            "conformal": self.conformal_result.to_dict(),
            "quality_warnings": self.quality_warnings,
            "is_quality_acceptable": self.is_quality_acceptable,
            "calibration": {
                "is_calibrated": self.is_calibrated,
                "temperature": round(self.temperature, 4),
            },
            "architecture": self.architecture,
            "gradcam_available": self.gradcam_available,
            "disclaimer": (
                "RetinaTrust is an educational research prototype. "
                "This output is for experimental analysis only and MUST NOT "
                "be used for medical diagnosis, clinical triage, or patient care."
            ),
        }
        if include_images and self.gradcam_overlay is not None:
            result["has_gradcam_overlay"] = True
        return result

    def generate_pdf(self, scan_id: Optional[str] = None) -> bytes:
        """Compiles a publication-grade Clinical Screening PDF Report."""
        if self.original_image_np is None:
            raise ValueError("Original image array not provided in prediction result.")
        return generate_pdf_report(
            prediction_result_dict=self.to_dict(),
            original_image_np=self.original_image_np,
            gradcam_overlay_np=self.gradcam_overlay,
            ben_graham_np=self.ben_graham_image,
            reference_id=scan_id,
        )

class RetinaPredictor:
    """Production inference engine for RetinaTrust 2.0.

    Supports DenseNet-121, Swin-T, ResNet-18, Ben Graham preprocessing,
    Grad-CAM++, Conformal Prediction, and Automated Clinical PDF generation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device_name: str = "auto",
        use_ben_graham: bool = True,
    ):
        self.config: AppConfig = load_config(config_path)
        self.device = get_device(device_name)
        self.use_ben_graham = use_ben_graham

        # Resolve checkpoint path
        if checkpoint_path is not None:
            self.checkpoint_file = resolve_path(checkpoint_path)
        else:
            default_ckpt = f"{self.config.paths.checkpoint_dir}/{self.config.model.architecture}_best.pt"
            self.checkpoint_file = resolve_path(default_ckpt)
            if not self.checkpoint_file.is_file():
                ckpt_dir = resolve_path(self.config.paths.checkpoint_dir)
                available = list(ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
                if available:
                    self.checkpoint_file = available[0]

        if not self.checkpoint_file.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.checkpoint_file}\n"
                f"Please train a model first using: python -m retinatrust.training.train"
            )

        logger.info(f"Loading RetinaTrust checkpoint: {self.checkpoint_file}")
        self.checkpoint = torch.load(self.checkpoint_file, map_location=self.device)

        self.architecture = self.checkpoint.get("architecture", self.config.model.architecture)
        self.num_classes = self.checkpoint.get("num_classes", 5)
        self.input_size = tuple(self.checkpoint.get("input_size", self.config.model.input_size))
        self.norm_mean = self.checkpoint.get("norm_mean", self.config.model.norm_mean)
        self.norm_std = self.checkpoint.get("norm_std", self.config.model.norm_std)
        self.temperature = float(self.checkpoint.get("temperature", self.config.uncertainty.temperature))
        self.is_coral = bool(self.checkpoint.get("is_coral", False))

        # Build architecture
        self.model, self.target_cam_layer = build_model(
            architecture=self.architecture,
            num_classes=self.num_classes,
            pretrained=False,
            use_coral=self.is_coral,
            device=self.device,
        )
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()

        self.transform = get_eval_transforms(
            input_size=self.input_size,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            use_ben_graham=False,  # Managed explicitly in predict() for preview access
        )

        # Initialize Conformal Predictor with validated 90%/95% thresholds
        q_90 = float(self.checkpoint.get("conformal_q90", 0.88))
        q_95 = float(self.checkpoint.get("conformal_q95", 0.95))
        self.conformal_engine = ConformalPredictor(q_hat_90=q_90, q_hat_95=q_95)

        logger.info(f"RetinaPredictor ready on device {self.device} (Architecture: {self.architecture}, T: {self.temperature:.3f})")

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
        generate_cam: bool = True,
        cam_threshold: float = 0.0,
        apply_ben_graham: Optional[bool] = None,
    ) -> PredictionResult:
        """Executes full multi-signal inference pipeline."""
        run_bg = self.use_ben_graham if apply_ben_graham is None else apply_ben_graham

        # 1. Image Quality Heuristics
        quality_res = check_image_quality(image_input, config=self.config.image_quality)
        if not quality_res.is_decodable:
            raise ValueError(f"Image decode failed: {'; '.join(quality_res.errors)}")

        # 2. Extract Raw NumPy RGB Array
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
            raw_rgb = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            raw_rgb = np.array(pil_img)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            raw_rgb = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            raw_rgb = image_input.copy()
            pil_img = Image.fromarray(raw_rgb)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 3. Optional Ben Graham Illumination Preprocessing
        ben_graham_np: Optional[np.ndarray] = None
        if run_bg:
            ben_graham_np = ben_graham_preprocessing(raw_rgb, target_size=self.input_size)
            feed_img = Image.fromarray(ben_graham_np)
        else:
            feed_img = pil_img

        # 4. Preprocess for Neural Network
        tensor = self.transform(feed_img).unsqueeze(0).to(self.device)

        # 5. Neural Forward Pass
        with torch.no_grad():
            outputs = self.model(tensor)

        # 6. Resolve Probabilities (CORAL or Softmax with Temperature Scaling)
        if self.is_coral:
            probs_tensor = coral_logits_to_probs(outputs)
            probs_np = probs_tensor.squeeze().cpu().numpy()
            pred_grade = int(np.argmax(probs_np))
            confidence = float(np.max(probs_np))
            entropy = float(-np.sum(probs_np * np.log2(np.clip(probs_np, 1e-12, 1.0))))
            is_calibrated = True
            is_uncertain = (confidence < self.config.uncertainty.min_confidence_threshold)
            uncertainty_reasons = []
            if is_uncertain:
                uncertainty_reasons.append(f"Confidence below threshold ({confidence:.1%})")
            prob_dict = {i: float(probs_np[i]) for i in range(len(probs_np))}
        else:
            uncertainty_res = quantify_uncertainty(
                logits=outputs,
                temperature=self.temperature,
                config=self.config.uncertainty,
            )
            pred_grade = uncertainty_res.predicted_grade
            confidence = uncertainty_res.confidence
            entropy = uncertainty_res.entropy
            is_uncertain = uncertainty_res.is_uncertain
            uncertainty_reasons = uncertainty_res.reasons
            prob_dict = uncertainty_res.probabilities
            is_calibrated = uncertainty_res.is_calibrated
            probs_np = np.array([prob_dict[i] for i in range(5)])

        grade_label = CLASS_NAMES.get(pred_grade, f"Grade {pred_grade}")
        grade_desc = GRADE_DESCRIPTIONS.get(pred_grade, "Unknown Severity")

        # 7. Conformal Prediction Coverage Sets
        conformal_res = self.conformal_engine.predict_set(probs_np)

        # Combine warnings
        combined_warnings = list(quality_res.warnings)
        if not quality_res.is_acceptable:
            combined_warnings.extend(quality_res.errors)

        # 8. Grad-CAM++ Explainability
        cam_overlay: Optional[np.ndarray] = None
        cam_heatmap: Optional[np.ndarray] = None
        cam_available = False

        if generate_cam:
            explanation = generate_explanation(
                model=self.model,
                target_layer=self.target_cam_layer,
                input_tensor=tensor.clone(),
                original_image=raw_rgb,
                target_class=pred_grade,
                alpha=self.config.explainability.alpha,
                threshold=cam_threshold,
                use_gradcam_pp=True,
            )
            if explanation["success"]:
                cam_available = True
                cam_overlay = explanation["blended_image"]
                cam_heatmap = explanation["heatmap"]
            else:
                combined_warnings.append(f"Grad-CAM++ failed: {explanation.get('error')}")

        return PredictionResult(
            predicted_grade=pred_grade,
            grade_label=grade_label,
            grade_description=grade_desc,
            confidence=confidence,
            predictive_entropy=entropy,
            is_uncertain=is_uncertain,
            uncertainty_reasons=uncertainty_reasons,
            class_probabilities=prob_dict,
            conformal_result=conformal_res,
            quality_warnings=combined_warnings,
            is_quality_acceptable=quality_res.is_acceptable,
            temperature=self.temperature,
            is_calibrated=is_calibrated,
            architecture=self.architecture,
            gradcam_available=cam_available,
            gradcam_overlay=cam_overlay,
            gradcam_heatmap=cam_heatmap,
            ben_graham_image=ben_graham_np,
            original_image_np=raw_rgb,
        )
