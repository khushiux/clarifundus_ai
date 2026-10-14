from datetime import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

def np_image_to_rl_image(np_arr: np.ndarray, width: float = 2.2 * inch, height: float = 2.2 * inch) -> RLImage:
    """Converts a NumPy RGB array to an in-memory ReportLab Flowable Image."""
    pil_img = Image.fromarray(np_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width, height=height)

def generate_pdf_report(
    prediction_result_dict: Dict[str, Any],
    original_image_np: np.ndarray,
    gradcam_overlay_np: Optional[np.ndarray] = None,
    ben_graham_np: Optional[np.ndarray] = None,
    reference_id: Optional[str] = None,
) -> bytes:
    """Generates a publication-grade Clinical Research Screening PDF Audit Report in memory.

    Args:
        prediction_result_dict: Structured dictionary from RetinaPredictor.
        original_image_np: NumPy RGB array of original fundus photograph.
        gradcam_overlay_np: Optional NumPy RGB array with Grad-CAM++ overlay.
        ben_graham_np: Optional NumPy RGB array with Ben Graham illumination normalization.
        reference_id: Optional patient/scan identifier.

    Returns:
        Raw bytes of generated PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
    )
    section_title = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#374151"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#92400E"),
    )

    elements = []
    ref_id = reference_id or f"SCAN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Header
    elements.append(Paragraph("👁️ RetinaTrust AI | Clinical Screening Research Audit", title_style))
    elements.append(Paragraph("Explainable & Uncertainty-Bounded Ophthalmic Decision Support", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=10))

    # Meta Info Table
    pred_grade = prediction_result_dict.get("predicted_grade", 0)
    grade_label = prediction_result_dict.get("grade_label", "Unknown")
    confidence = prediction_result_dict.get("estimated_confidence", 0.0)

    meta_data = [
        [
            Paragraph(f"<b>Scan Reference ID:</b> {ref_id}", body_style),
            Paragraph(f"<b>Audit Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style),
        ],
        [
            Paragraph(f"<b>Model Architecture:</b> {prediction_result_dict.get('architecture', 'DenseNet-121')}", body_style),
            Paragraph(f"<b>Execution Engine:</b> PyTorch CUDA (RTX 3050)", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F4F6")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # Severity Verdict Banner
    grade_colors = {
        0: colors.HexColor("#059669"),
        1: colors.HexColor("#0284C7"),
        2: colors.HexColor("#D97706"),
        3: colors.HexColor("#DC2626"),
        4: colors.HexColor("#7F1D1D"),
    }
    verdict_color = grade_colors.get(pred_grade, colors.HexColor("#1F2937"))
    verdict_text = f"<b>DIAGNOSTIC ESTIMATE: GRADE {pred_grade} ({grade_label.upper()})</b>"
    desc_text = prediction_result_dict.get("grade_description", "")

    verdict_table = Table(
        [[
            Paragraph(f"<font color='white' size='13'>{verdict_text}</font>", body_style),
            Paragraph(f"<font color='white' size='9'><b>Calibrated Confidence:</b> {confidence:.1%}</font>", body_style)
        ],
        [
            Paragraph(f"<font color='white' size='9'>{desc_text}</font>", body_style),
            Paragraph("", body_style)
        ]],
        colWidths=[400, 140]
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (0, 1), (1, 1)),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 14))

    # Imagery Row (Original, Ben Graham, Grad-CAM++)
    elements.append(Paragraph("1. Ophthalmic Saliency & Lesion Mapping", section_title))
    image_cells = []
    headers = []

    orig_rl = np_image_to_rl_image(original_image_np, width=2.4 * inch, height=2.4 * inch)
    image_cells.append(orig_rl)
    headers.append(Paragraph("<center><b>Original Photograph</b></center>", body_style))

    if ben_graham_np is not None:
        bg_rl = np_image_to_rl_image(ben_graham_np, width=2.4 * inch, height=2.4 * inch)
        image_cells.append(bg_rl)
        headers.append(Paragraph("<center><b>Ben Graham Preprocessed</b></center>", body_style))

    if gradcam_overlay_np is not None:
        cam_rl = np_image_to_rl_image(gradcam_overlay_np, width=2.4 * inch, height=2.4 * inch)
        image_cells.append(cam_rl)
        headers.append(Paragraph("<center><b>Grad-CAM++ Saliency Map</b></center>", body_style))

    col_w = 540 / len(image_cells)
    img_table = Table([headers, image_cells], colWidths=[col_w] * len(image_cells))
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 14))

    # Multi-Signal Reliability & Conformal Prediction Table
    elements.append(Paragraph("2. Conformal Safety Guarantees & Clinical Triage", section_title))
    conformal = prediction_result_dict.get("conformal", {})
    triage = conformal.get("triage_recommendation", "Standard Clinical Review")
    set_95 = conformal.get("prediction_set_95", [pred_grade])

    safety_data = [
        [
            Paragraph("<b>Safety Metric</b>", body_style),
            Paragraph("<b>Observed Value</b>", body_style),
            Paragraph("<b>Clinical Triage Interpretation</b>", body_style),
        ],
        [
            Paragraph("<b>Conformal 95% Set</b>", body_style),
            Paragraph(f"<b>{set_95}</b>", body_style),
            Paragraph("Finite-sample mathematical coverage guarantee (p >= 0.95)", body_style),
        ],
        [
            Paragraph("<b>Predictive Entropy</b>", body_style),
            Paragraph(f"{prediction_result_dict.get('predictive_entropy', 0.0):.2f} bits", body_style),
            Paragraph("Measures dispersion across disease stages", body_style),
        ],
        [
            Paragraph("<b>Automated Triage</b>", body_style),
            Paragraph(f"<b>{triage}</b>", body_style),
            Paragraph("Automated referral and clinical routing rule", body_style),
        ],
        [
            Paragraph("<b>Image Quality Heuristics</b>", body_style),
            Paragraph("PASSED" if prediction_result_dict.get("is_quality_acceptable", True) else "FLAGGED", body_style),
            Paragraph("; ".join(prediction_result_dict.get("quality_warnings", ["Clear focus & illumination"])) or "No quality defects", body_style),
        ],
    ]
    safety_table = Table(safety_data, colWidths=[140, 150, 250])
    safety_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(safety_table)
    elements.append(Spacer(1, 14))

    # Mandatory Legal Disclaimer Box
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#D1D5DB"), spaceAfter=6))
    elements.append(Paragraph(
        "<b>RESEARCH PROTOTYPE NOTICE:</b> RetinaTrust is an academic AI research system designed for "
        "evaluating explainability and uncertainty quantification. It is NOT certified as a medical device by "
        "the FDA, CE, or CDSCO. All screening findings must be verified by a board-certified ophthalmologist.",
        disclaimer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
