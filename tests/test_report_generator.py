import numpy as np
import pytest
from retinatrust.inference.report_generator import generate_pdf_report

def test_generate_pdf_report():
    dummy_result = {
        "predicted_grade": 2,
        "grade_label": "Moderate DR",
        "grade_description": "More than microaneurysms, but less than severe NPDR",
        "estimated_confidence": 0.8421,
        "predictive_entropy": 0.65,
        "architecture": "DenseNet-121",
        "is_quality_acceptable": True,
        "quality_warnings": [],
        "conformal": {
            "prediction_set_95": [1, 2],
            "set_size_95": 2,
            "triage_recommendation": "Non-Urgent Clinical Ophthalmology Follow-Up",
        },
    }

    orig_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    cam_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    bg_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

    pdf_bytes = generate_pdf_report(
        prediction_result_dict=dummy_result,
        original_image_np=orig_img,
        gradcam_overlay_np=cam_img,
        ben_graham_np=bg_img,
        reference_id="TEST-SCAN-001",
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF magic byte signature is %PDF
    assert pdf_bytes.startswith(b"%PDF")
