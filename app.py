from datetime import datetime
import io
import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from retinatrust.config import load_config, resolve_path
from retinatrust.inference.predictor import RetinaPredictor, CLASS_NAMES, GRADE_DESCRIPTIONS

st.set_page_config(
    page_title="RetinaTrust 2.0 | Explainable & Uncertainty-Aware DR Research",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clinical CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.90rem;
        margin-bottom: 1.2rem;
    }
    .grade-card {
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    .grade-0 { background-color: #059669; }
    .grade-1 { background-color: #0284C7; }
    .grade-2 { background-color: #D97706; }
    .grade-3 { background-color: #DC2626; }
    .grade-4 { background-color: #7F1D1D; }
    .conformal-card {
        background-color: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-left: 5px solid #4F46E5;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        color: #312E81;
        margin-bottom: 1rem;
    }
    .uncertainty-alert {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #991B1B;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .quality-warning {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #B45309;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_cached_predictor(checkpoint_path: str) -> Optional[RetinaPredictor]:
    """Caches RetinaPredictor in memory to avoid reloading weights per query."""
    try:
        if os.path.exists(checkpoint_path):
            return RetinaPredictor(checkpoint_path=checkpoint_path)
    except Exception as e:
        st.error(f"Failed to initialize predictor: {e}")
    return None

@st.cache_data
def get_curated_samples():
    """Returns curated representative samples covering grades 0-4 from bundled data/samples or manifests."""
    cfg = load_config()
    sample_dir = resolve_path("data/samples")
    samples = []

    # 1. Prioritize self-contained bundled samples (for seamless cloud deployment)
    if sample_dir.is_dir():
        for f in sorted(sample_dir.glob("*.*")):
            parts = f.stem.split("_")
            if len(parts) >= 4 and parts[1] == "grade":
                try:
                    g = int(parts[2])
                    is_aptos = (parts[0].lower() == "aptos")
                    cid = "_".join(parts[3:])
                    ds_name = "APTOS 2019 (In-Domain Validation)" if is_aptos else "IDRiD (Cross-Domain Benchmark Test)"
                    label_name = f"APTOS Val — Grade {g}: {CLASS_NAMES[g]} ({cid[:8]})" if is_aptos else f"IDRiD Test (OOD) — Grade {g}: {CLASS_NAMES[g]} ({cid})"
                    samples.append({
                        "name": label_name,
                        "dataset": ds_name,
                        "id": cid,
                        "grade": g,
                        "description": GRADE_DESCRIPTIONS[g],
                        "path": str(f),
                    })
                except Exception:
                    continue

    if samples:
        return samples

    # 2. Fallback to local manifest paths
    val_path = resolve_path(f"{cfg.paths.manifest_dir}/val_manifest.csv")
    idrid_path = resolve_path(f"{cfg.paths.manifest_dir}/idrid_test_manifest.csv")

    if val_path.is_file():
        val_df = pd.read_csv(val_path)
        for g in range(5):
            sub = val_df[val_df["label"] == g]
            if not sub.empty:
                r = sub.iloc[0]
                samples.append({
                    "name": f"APTOS Val — Grade {g}: {CLASS_NAMES[g]} ({r['id_code'][:8]})",
                    "dataset": "APTOS 2019 (In-Domain Validation)",
                    "id": str(r["id_code"]),
                    "grade": int(g),
                    "description": GRADE_DESCRIPTIONS[g],
                    "path": str(r["image_path"]),
                })
    if idrid_path.is_file():
        idrid_df = pd.read_csv(idrid_path)
        for g in [0, 2, 4]:
            sub = idrid_df[idrid_df["label"] == g]
            if not sub.empty:
                r = sub.iloc[0]
                samples.append({
                    "name": f"IDRiD Test (OOD) — Grade {g}: {CLASS_NAMES[g]} ({r['id_code']})",
                    "dataset": "IDRiD (Cross-Domain Benchmark Test)",
                    "id": str(r["id_code"]),
                    "grade": int(g),
                    "description": GRADE_DESCRIPTIONS[g],
                    "path": str(r["image_path"]),
                })
    return samples

def main():
    cfg = load_config()
    ckpt_dir = resolve_path(cfg.paths.checkpoint_dir)
    available_ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
    selected_ckpt = str(available_ckpts[0]) if available_ckpts else str(ckpt_dir / "resnet18_best.pt")

    # Initialize Session State
    if "active_image" not in st.session_state:
        st.session_state["active_image"] = None
    if "active_source" not in st.session_state:
        st.session_state["active_source"] = None
    if "ground_truth_grade" not in st.session_state:
        st.session_state["ground_truth_grade"] = None

    # Sidebar Controls
    st.sidebar.title("👁️ RetinaTrust 2.0")
    st.sidebar.markdown("**Ophthalmic Decision Support Research**")
    st.sidebar.info(
        "Hybrid Deep Learning with Conformal Prediction Guarantees, "
        "Ben Graham Preprocessing, and Grad-CAM++."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Clinical Inference Controls")
    use_ben_graham = st.sidebar.checkbox(
        "Enable Ben Graham Illumination Preprocessing",
        value=True,
        help="Subtracted local Gaussian mean removes camera vignetting and highlights faint microaneurysms."
    )
    cam_threshold = st.sidebar.slider(
        "Grad-CAM++ Saliency Cutoff",
        min_value=0.0,
        max_value=0.8,
        value=0.15,
        step=0.05,
        help="Filters low-intensity background noise to reveal primary microvascular lesion clusters."
    )

    # Main Header
    st.markdown('<div class="main-header">RetinaTrust 2.0: AI Screening Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Signal Retinopathy Research with Conformal Finite-Sample Safety Sets, '
        'Grad-CAM++ Lesion Mapping, and Automated Clinical Audit Export.</div>',
        unsafe_allow_html=True,
    )

    # Medical Disclaimer
    st.markdown(
        '<div class="disclaimer-box">'
        '⚠️ <strong>RESEARCH PROTOTYPE NOTICE:</strong> This software is an educational research artifact. '
        'It is <strong>NOT</strong> a certified medical device and must <strong>NEVER</strong> be used for '
        'independent clinical diagnosis or patient care without board-certified ophthalmologist review.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Model Check
    if not os.path.exists(selected_ckpt):
        st.error("⚠️ Model Checkpoint Not Found")
        st.warning(
            f"No trained checkpoint was found at:\n`{selected_ckpt}`\n\n"
            "RetinaTrust strictly prohibits fabricating fake predictions."
        )
        st.code("python -m retinatrust.training.train --smoke-test", language="powershell")
        return

    predictor = get_cached_predictor(selected_ckpt)
    if predictor is None:
        st.error("Failed to load model weights.")
        return

    st.sidebar.success(f"Model Loaded ({predictor.device})")
    st.sidebar.markdown(f"**Architecture:** `{predictor.architecture}`")
    st.sidebar.markdown(f"**Calibration Temp ($T$):** `{predictor.temperature:.4f}`")

    # Image Input Section with Tabs
    st.subheader("Select Retinal Fundus Photograph")
    tab_sample, tab_upload = st.tabs([
        "🧪 Curated Validation Samples (Grades 0–4)",
        "📤 Upload Custom Fundus Image",
    ])

    curated_samples = get_curated_samples()

    with tab_sample:
        st.markdown(
            "Select a pre-verified retinal fundus image from the validation / OOD benchmark sets to test the system across all 5 clinical severity levels:"
        )
        sample_names = [s["name"] for s in curated_samples]
        selected_sample_name = st.selectbox(
            "Choose a clinical benchmark case:",
            options=sample_names,
            index=0 if sample_names else None,
            help="Includes benchmark cases from APTOS 2019 (in-domain) and IDRiD (cross-domain test)."
        )

        chosen_sample = next((s for s in curated_samples if s["name"] == selected_sample_name), None)

        if chosen_sample:
            scol1, scol2 = st.columns([3, 1])
            with scol1:
                st.markdown(f"**Dataset Origin:** `{chosen_sample['dataset']}`")
                st.markdown(f"**Patient / Image ID:** `{chosen_sample['id']}`")
                st.markdown(f"**Ground Truth Diagnosis:** **Grade {chosen_sample['grade']} ({CLASS_NAMES[chosen_sample['grade']]})**")
                st.caption(f"_{chosen_sample['description']}_")

                st.write("")
                # Prominent Load Button explicitly requested
                if st.button("📥 Load & Analyze Selected Sample", type="primary", use_container_width=True, key="btn_load_sample"):
                    p = Path(chosen_sample["path"])
                    if p.is_file():
                        try:
                            st.session_state["active_image"] = Image.open(p).convert("RGB")
                            st.session_state["active_source"] = chosen_sample["name"]
                            st.session_state["ground_truth_grade"] = chosen_sample["grade"]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading sample image: {e}")
                    else:
                        st.error(f"Sample file not found at: {p}")

            with scol2:
                p = Path(chosen_sample["path"])
                if p.is_file():
                    try:
                        thumb = Image.open(p).convert("RGB")
                        st.image(thumb, caption="Sample Preview", width=160)
                    except Exception:
                        pass

    with tab_upload:
        st.markdown("Upload your own fundus photograph (JPG, PNG, JPEG, BMP - Max 20MB):")
        uploaded_file = st.file_uploader(
            "Upload Retinal Fundus Photograph",
            type=["jpg", "jpeg", "png", "bmp", "tif"],
            help="Images are processed entirely in memory without persistent file retention.",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            col_up_btn, col_up_thumb = st.columns([3, 1])
            with col_up_btn:
                st.markdown(f"**Filename:** `{uploaded_file.name}`")
                st.markdown(f"**Size:** `{len(uploaded_file.getvalue()) / 1024:.1f} KB`")
                if st.button("📥 Load & Analyze Uploaded Image", type="primary", use_container_width=True, key="btn_load_upload"):
                    try:
                        st.session_state["active_image"] = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                        st.session_state["active_source"] = f"Uploaded File: {uploaded_file.name}"
                        st.session_state["ground_truth_grade"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error decoding uploaded image: {e}")
            with col_up_thumb:
                try:
                    up_thumb = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    st.image(up_thumb, caption="Upload Preview", width=160)
                except Exception:
                    pass

    # Check Active Loaded Image
    input_pil = st.session_state.get("active_image")
    if input_pil is None:
        st.info("👆 Please click **'📥 Load & Analyze Selected Sample'** above or upload your own retinal photograph to run the screening.")
        return

    # Active Image Banner with Clear Button
    st.markdown("---")
    c_banner, c_reset = st.columns([4, 1])
    with c_banner:
        gt_text = ""
        if st.session_state.get("ground_truth_grade") is not None:
            g = st.session_state["ground_truth_grade"]
            gt_text = f" &nbsp;|&nbsp; 🎯 <strong>Ground Truth:</strong> Grade {g} ({CLASS_NAMES[g]})"
        st.markdown(
            f'<div style="background-color: #F0FDF4; border: 1px solid #86EFAC; border-left: 5px solid #22C55E; padding: 0.65rem 1rem; border-radius: 6px; color: #166534; font-size: 0.95rem;">'
            f'<strong>👁️ Currently Analyzing:</strong> {st.session_state.get("active_source", "Active Retinal Photograph")}{gt_text}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c_reset:
        if st.button("🔄 Reset / Clear", use_container_width=True, help="Clear active image and select another"):
            st.session_state["active_image"] = None
            st.session_state["active_source"] = None
            st.session_state["ground_truth_grade"] = None
            st.rerun()

    # Run Prediction Pipeline
    with st.spinner("Executing Ben Graham normalization, forward pass, Conformal sets, and Grad-CAM++..."):
        try:
            result = predictor.predict(
                image_input=input_pil,
                generate_cam=True,
                cam_threshold=cam_threshold,
                apply_ben_graham=use_ben_graham,
            )
        except Exception as e:
            st.error(f"Inference pipeline error: {e}")
            return

    st.markdown("---")

    # Quality Warnings
    if result.quality_warnings:
        warning_text = "<br>• ".join(result.quality_warnings)
        st.markdown(
            f'<div class="quality-warning"><strong>⚠️ Technical Image Quality Heuristics:</strong><br>• {warning_text}</div>',
            unsafe_allow_html=True,
        )

    # Uncertainty Alert
    if result.is_uncertain:
        reasons_text = "<br>• ".join(result.uncertainty_reasons)
        st.markdown(
            f'<div class="uncertainty-alert">'
            f'<strong>🚩 UNCERTAINTY WARNING (Requires Human Clinical Review):</strong><br>'
            f'Model flagged this prediction as ambiguous.<br>• {reasons_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Main Visual Layout
    col_images, col_metrics = st.columns([1, 1], gap="large")

    with col_images:
        st.subheader("1. Ophthalmic Saliency & Lesions")
        tab_side, tab_bg, tab_cam = st.tabs(["Side-by-Side View", "Ben Graham Preprocessed", "Grad-CAM++ Overlay"])

        with tab_side:
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(input_pil, caption="Original Fundus Photograph", use_container_width=True)
            with img_c2:
                if result.gradcam_available and result.gradcam_overlay is not None:
                    st.image(
                        result.gradcam_overlay,
                        caption=f"Grad-CAM++ (Target: {result.grade_label})",
                        use_container_width=True,
                    )
                else:
                    st.warning("Grad-CAM++ explanation not available.")

        with tab_bg:
            if result.ben_graham_image is not None:
                st.image(
                    result.ben_graham_image,
                    caption="Ben Graham Illumination-Normalized View (Vignetting Removed)",
                    use_container_width=True,
                )
            else:
                st.info("Ben Graham preprocessing disabled.")

        with tab_cam:
            if result.gradcam_available and result.gradcam_overlay is not None:
                st.image(result.gradcam_overlay, caption="High-Resolution Grad-CAM++ Overlay", use_container_width=True)

        st.caption(
            "ℹ️ Heatmap uses higher-order partial derivatives (Grad-CAM++) to map distinct "
            "microvascular abnormalities across the retina without averaging them out."
        )

    with col_metrics:
        st.subheader("2. Estimated Severity & Conformal Safety")

        grade = result.predicted_grade
        grade_card_class = f"grade-{min(4, max(0, grade))}"

        st.markdown(
            f'<div class="grade-card {grade_card_class}">'
            f'<h2 style="margin:0; color:white;">Grade {grade}: {result.grade_label}</h2>'
            f'<p style="margin:5px 0 0 0; font-size:1.05rem; opacity:0.95;">{result.grade_description}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Conformal Prediction Safety Card & Benchmark Ground-Truth Concordance
        conf = result.conformal_result

        # Ground-Truth Benchmark Concordance (When Sample is Loaded)
        gt = st.session_state.get("ground_truth_grade")
        if gt is not None:
            is_exact = (grade == gt)
            in_set_95 = (gt in conf.prediction_set_95) if conf and hasattr(conf, "prediction_set_95") else False
            if is_exact:
                st.markdown(
                    f'<div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-left: 4px solid #10B981; padding: 0.65rem 1rem; border-radius: 6px; color: #065F46; font-size: 0.92rem; margin-bottom: 0.9rem;">'
                    f'<strong>✅ Benchmark Concordance:</strong> Model predicted Grade {grade} ({result.grade_label}), '
                    f'<strong>matching ground-truth clinical annotation!</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                set_badge = "✅ Ground-truth included in 95% Conformal Set" if in_set_95 else "⚠️ Outside 95% Conformal Set"
                st.markdown(
                    f'<div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #3B82F6; padding: 0.65rem 1rem; border-radius: 6px; color: #1E40AF; font-size: 0.92rem; margin-bottom: 0.9rem;">'
                    f'<strong>📊 Benchmark Comparison:</strong> Ground Truth is <strong>Grade {gt} ({CLASS_NAMES[gt]})</strong> | '
                    f'Model predicted <strong>Grade {grade} ({result.grade_label})</strong>.<br>'
                    f'<em>{set_badge}</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        set_str = ", ".join([f"Grade {g}" for g in conf.prediction_set_95]) if conf and hasattr(conf, "prediction_set_95") else "N/A"
        set_size = conf.set_size_95 if conf and hasattr(conf, "set_size_95") else len(conf.prediction_set_95) if conf else 0
        triage_rec = conf.triage_recommendation if conf and hasattr(conf, "triage_recommendation") else "Clinical Review Recommended"
        st.markdown(
            f'<div class="conformal-card">'
            f'<strong>🛡️ Conformal Prediction (95% Mathematical Coverage Guarantee):</strong><br>'
            f'<strong>Plausible Diagnostic Set:</strong> <code>{{{set_str}}}</code> (Cardinality: {set_size})<br>'
            f'<strong>Automated Clinical Triage:</strong> <em>{triage_rec}</em>'
            f'</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Model Confidence", f"{result.confidence:.1%}")
        m2.metric("Predictive Entropy", f"{result.predictive_entropy:.2f} bits")
        m3.metric("Calibration Temp (T)", f"{result.temperature:.3f}")

        st.markdown("#### Calibrated Class Probability Distribution")
        df_probs = pd.DataFrame({
            "Grade": [f"Grade {k} ({CLASS_NAMES[k]})" for k in range(5)],
            "Probability": [result.class_probabilities[k] for k in range(5)],
        })
        st.bar_chart(df_probs.set_index("Grade"), color="#1E3A8A")

        # Download Clinical Audit PDF Button
        st.markdown("---")
        st.markdown("#### 📄 Clinical Audit & Reporting")
        try:
            pdf_bytes = result.generate_pdf(scan_id=f"AUDIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            st.download_button(
                label="📥 Download Clinical Screening Report (PDF)",
                data=pdf_bytes,
                file_name=f"RetinaTrust_Screening_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                help="Generates a downloadable, publication-grade clinical screening PDF audit document.",
            )
        except Exception as e:
            st.warning(f"Could not prepare PDF download: {e}")

if __name__ == "__main__":
    main()
