import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import plotly.graph_objects as go
import plotly.express as px

from utils.image_quality import compute_ela_analysis, save_ela_visualization
from utils.ocr_extraction import extract_document_text, write_ocr_results, initialize_ocr_engine
from utils.qr_detection import find_qr_codes, draw_qr_annotations
from utils.watermark_check import locate_watermark, score_watermark_authenticity
from utils.layout_check import analyze_document_structure, save_layout_overlay
from utils.validation import compute_final_verdict, persist_analysis_log
from utils.ml_model import classify_document, load_classifier_model
from utils.config import (
    ELA_OUTPUT_DIR, OCR_OUTPUT_DIR, QR_OUTPUT_DIR, 
    LAYOUT_OUTPUT_DIR, WATERMARK_TEMPLATE_PATH
)
from utils.logger import logger

# Page configuration
st.set_page_config(
    page_title="Document Forgery Detection",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'final_verdict' not in st.session_state:
    st.session_state.final_verdict = None


def safe_delete_file(file_path: str, max_retries: int = 5, delay: float = 0.1) -> bool:
    """Safely delete a file on Windows, handling file lock issues.
    
    Args:
        file_path: Path to the file to delete
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        True if file was deleted successfully, False otherwise
    """
    if not os.path.exists(file_path):
        return True
    
    for attempt in range(max_retries):
        try:
            os.remove(file_path)
            return True
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                logger.warning(f"Failed to delete {file_path} after {max_retries} attempts: {e}")
                return False
    return False


def convert_pdf_to_image(pdf_path: str, page_number: int = 0) -> Optional[Image.Image]:
    """Convert a PDF page to a PIL Image.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to convert (0-indexed, default: first page)
        
    Returns:
        PIL Image object or None if conversion fails
    """
    try:
        pdf_document = fitz.open(pdf_path)
        if page_number >= len(pdf_document):
            logger.warning(f"Page {page_number} not found in PDF, using first page")
            page_number = 0
        
        page = pdf_document[page_number]
        # Render page to a pixmap (image) with high DPI for quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_document.close()
        return img
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return None


def run_analysis_pipeline(image_path: str) -> Tuple[Dict, Dict]:
    """Execute all 7 analysis stages on a document image.
    
    Args:
        image_path: Path to uploaded document image
        
    Returns:
        Tuple of (stage_results_dict, final_verdict_dict)
    """
    results = {}
    
    # Stage 1: Error Level Analysis
    with st.spinner("Analyzing image quality (ELA)..."):
        ela_map, anomaly = compute_ela_analysis(image_path)
        if ela_map is not None:
            ela_path = ELA_OUTPUT_DIR / "ela_result.png"
            save_ela_visualization(ela_map, str(ela_path))
            results["image_quality"] = {
                "anomaly_score": anomaly,
                "visualization_path": str(ela_path)
            }
        else:
            results["image_quality"] = {"anomaly_score": 0.0, "visualization_path": None}
    
    # Stage 2: OCR Text Extraction
    with st.spinner("Extracting text (OCR)..."):
        ocr_engine = initialize_ocr_engine()
        ocr_data = extract_document_text(image_path, ocr_engine)
        ocr_path = OCR_OUTPUT_DIR / "ocr_result.txt"
        write_ocr_results(ocr_data, str(ocr_path))
        results["ocr"] = {
            **ocr_data,
            "output_path": str(ocr_path)
        }
    
    # Stage 3: QR Code Detection
    with st.spinner("Detecting QR codes..."):
        qr_data = find_qr_codes(image_path)
        if qr_data["detected"]:
            qr_path = QR_OUTPUT_DIR / "qr_annotated.png"
            draw_qr_annotations(image_path, qr_data, str(qr_path))
            qr_data["visualization_path"] = str(qr_path)
        results["qr"] = qr_data
    
    # Stage 4: Watermark Verification
    with st.spinner("Checking watermark..."):
        watermark_data = locate_watermark(image_path, WATERMARK_TEMPLATE_PATH)
        authenticity = score_watermark_authenticity(watermark_data)
        watermark_data["authenticity_score"] = authenticity
        results["watermark"] = watermark_data
    
    # Stage 5: Layout Analysis
    with st.spinner("Analyzing layout..."):
        layout_data = analyze_document_structure(image_path)
        if layout_data["valid"]:
            layout_path = LAYOUT_OUTPUT_DIR / "layout_visualization.png"
            save_layout_overlay(image_path, layout_data, str(layout_path))
            layout_data["visualization_path"] = str(layout_path)
        results["layout"] = layout_data
    
    # Stage 6: ML Classification
    with st.spinner("Running ML model..."):
        model, device = load_classifier_model()
        ml_result = classify_document(image_path, model, device)
        results["ml_model"] = ml_result
    
    # Stage 7: Final Validation
    with st.spinner("Computing final verdict..."):
        verdict = compute_final_verdict(results)
        persist_analysis_log(results, verdict)
        results["validation_log_path"] = str(Path("outputs/logs/last_run.json"))
    
    return results, verdict


def render_premium_validation_dashboard(verdict: Dict) -> None:
    """Render a premium dashboard-style UI for Stage 7 validation summary.
    
    Args:
        verdict: Final validation result dictionary
    """
    if not verdict:
        st.warning("No validation data available")
        return
    
    verdict_text = verdict.get("verdict", "UNKNOWN")
    score = verdict.get("overall_score", 0.0)
    confidence = verdict.get("confidence", "LOW")
    stage_scores = verdict.get("stage_scores", {})
    
    # Color scheme based on verdict
    color_map = {
        "GENUINE": {"bg": "#10b981", "text": "#ffffff", "border": "#059669", "icon": "✅"},
        "LIKELY GENUINE": {"bg": "#3b82f6", "text": "#ffffff", "border": "#2563eb", "icon": "✓"},
        "SUSPICIOUS": {"bg": "#f59e0b", "text": "#ffffff", "border": "#d97706", "icon": "⚠️"},
        "FAKE": {"bg": "#ef4444", "text": "#ffffff", "border": "#dc2626", "icon": "❌"},
        "ERROR": {"bg": "#6b7280", "text": "#ffffff", "border": "#4b5563", "icon": "⚠️"}
    }
    
    colors = color_map.get(verdict_text, color_map["ERROR"])
    confidence_colors = {
        "HIGH": "#10b981",
        "MEDIUM": "#f59e0b",
        "LOW": "#ef4444"
    }
    
    # Main Verdict Card
    st.markdown("""
    <style>
    .verdict-card {
        background: linear-gradient(135deg, """ + colors["bg"] + """ 0%, """ + colors["border"] + """ 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 2px solid """ + colors["border"] + """;
    }
    .verdict-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: """ + colors["text"] + """;
        margin: 0;
        text-align: center;
    }
    .verdict-subtitle {
        font-size: 1.2rem;
        color: """ + colors["text"] + """;
        opacity: 0.9;
        text-align: center;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid """ + colors["bg"] + """;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Verdict Card
    st.markdown(f"""
    <div class="verdict-card">
        <h1 class="verdict-title">{colors["icon"]} {verdict_text}</h1>
        <p class="verdict-subtitle">Confidence: <strong>{confidence}</strong> | Overall Score: <strong>{score:.2f}/100</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #6b7280; font-size: 0.9rem; margin: 0; text-transform: uppercase;">Overall Score</h3>
            <h2 style="color: {colors['bg']}; font-size: 2.5rem; margin: 0.5rem 0;">{score:.1f}</h2>
            <p style="color: #9ca3af; margin: 0;">out of 100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_color = confidence_colors.get(confidence, "#6b7280")
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #6b7280; font-size: 0.9rem; margin: 0; text-transform: uppercase;">Confidence</h3>
            <h2 style="color: {confidence_color}; font-size: 2.5rem; margin: 0.5rem 0;">{confidence}</h2>
            <p style="color: #9ca3af; margin: 0;">assessment level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        verdict_icon = colors["icon"]
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #6b7280; font-size: 0.9rem; margin: 0; text-transform: uppercase;">Verdict</h3>
            <h2 style="color: {colors['bg']}; font-size: 2.5rem; margin: 0.5rem 0;">{verdict_icon}</h2>
            <p style="color: #9ca3af; margin: 0;">{verdict_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gauge Meter
    st.subheader("📊 Authenticity Score Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Authenticity Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': colors["bg"]},
            'steps': [
                {'range': [0, 30], 'color': "#fee2e2"},
                {'range': [30, 60], 'color': "#fef3c7"},
                {'range': [60, 80], 'color': "#dbeafe"},
                {'range': [80, 100], 'color': "#d1fae5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#1f2937"}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stage Scores Breakdown
    st.subheader("📋 Stage-wise Analysis Breakdown")
    
    # Stage names mapping
    stage_names = {
        "ela": "🔍 Image Quality (ELA)",
        "ocr": "📝 OCR Text Extraction",
        "qr": "📱 QR Code Detection",
        "watermark": "💧 Watermark Verification",
        "layout": "📐 Layout Analysis",
        "ml": "🤖 ML Classification"
    }
    
    # Create table data
    table_data = []
    for stage_key, stage_score in stage_scores.items():
        stage_name = stage_names.get(stage_key, stage_key.upper())
        # Determine color based on score
        if stage_score >= 80:
            score_color = "#10b981"
            status = "Excellent"
        elif stage_score >= 60:
            score_color = "#3b82f6"
            status = "Good"
        elif stage_score >= 40:
            score_color = "#f59e0b"
            status = "Fair"
        else:
            score_color = "#ef4444"
            status = "Poor"
        
        table_data.append({
            "Stage": stage_name,
            "Score": f"{stage_score:.2f}",
            "Status": status,
            "Color": score_color
        })
    
    # Display as styled table
    for row in table_data:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{row['Stage']}**")
        with col2:
            st.markdown(f"<span style='color: {row['Color']}; font-weight: 700; font-size: 1.1rem;'>{row['Score']}/100</span>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<span style='background: {row['Color']}20; color: {row['Color']}; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem;'>{row['Status']}</span>", unsafe_allow_html=True)
        
        # Progress bar
        progress_color = row['Color']
        st.markdown(f"""
        <div style="background: #e5e7eb; height: 8px; border-radius: 4px; margin: 0.5rem 0 1rem 0; overflow: hidden;">
            <div style="background: {progress_color}; height: 100%; width: {float(row['Score'])}%; transition: width 0.3s;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Radar Chart
    if len(stage_scores) > 0:
        st.subheader("🎯 Stage Performance Radar Chart")
        
        categories = [stage_names.get(k, k.upper()) for k in stage_scores.keys()]
        values = list(stage_scores.values())
        
        # Convert hex color to rgba for transparency
        def hex_to_rgba(hex_color, alpha=0.5):
            """Convert hex color to rgba string."""
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Stage Scores',
            line_color=colors["bg"],
            fillcolor=hex_to_rgba(colors["bg"], 0.5)
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#1f2937"}
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Collapsible JSON View (using checkbox since we can't nest expanders)
    show_json = st.checkbox("🔧 Show Debug: Raw JSON Data", value=False)
    if show_json:
        st.json(verdict)


def render_verdict_display(verdict: Dict) -> None:
    """Display final verdict with appropriate styling.
    
    Args:
        verdict: Final validation result dictionary
    """
    st.header("🎯 Final Verdict")
    
    verdict_text = verdict["verdict"]
    score = verdict["overall_score"]
    confidence = verdict["confidence"]
    
    if verdict_text == "GENUINE":
        st.success(f"**{verdict_text}** - Confidence: {confidence} | Score: {score:.2f}/100")
    elif verdict_text == "FAKE":
        st.error(f"**{verdict_text}** - Confidence: {confidence} | Score: {score:.2f}/100")
    else:
        st.warning(f"**{verdict_text}** - Confidence: {confidence} | Score: {score:.2f}/100")


def render_stage_results(results: Dict) -> None:
    """Render collapsible sections for each analysis stage.
    
    Args:
        results: Dictionary containing all stage results
    """
    st.header("📊 Stage-wise Results")
    
    # Stage 1: Image Quality
    with st.expander("🔍 Stage 1: Image Quality Analysis (ELA)", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            vis_path = results.get("image_quality", {}).get("visualization_path")
            if vis_path and Path(vis_path).exists():
                st.image(vis_path, caption="ELA Visualization", use_container_width=True)
        with col2:
            anomaly = results.get("image_quality", {}).get("anomaly_score", 0.0)
            st.metric("Anomaly Score", f"{anomaly:.2f}")
            st.caption("Lower score = more authentic")
    
    # Stage 2: OCR
    with st.expander("📝 Stage 2: OCR Text Extraction", expanded=False):
        ocr = results.get("ocr", {})
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text_area("Extracted Text", ocr.get("text", "No text found"), 
                        height=150, disabled=True)
        with col2:
            st.metric("Confidence", f"{ocr.get('confidence', 0.0):.2%}")
            st.metric("Word Count", ocr.get("word_count", 0))
    
    # Stage 3: QR Codes
    with st.expander("📱 Stage 3: QR Code Detection", expanded=False):
        qr = results.get("qr", {})
        col1, col2 = st.columns([2, 1])
        with col1:
            if qr.get("detected") and qr.get("visualization_path"):
                vis_path = qr["visualization_path"]
                if Path(vis_path).exists():
                    st.image(vis_path, caption="QR Code Detection", use_container_width=True)
            else:
                st.info("No QR codes detected")
        with col2:
            st.metric("QR Codes Found", qr.get("count", 0))
            for i, code in enumerate(qr.get("codes", [])):
                st.caption(f"QR {i+1}: {code.get('data', 'N/A')[:30]}...")
    
    # Stage 4: Watermark
    with st.expander("💧 Stage 4: Watermark Verification", expanded=False):
        wm = results.get("watermark", {})
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Watermark Detected", "Yes" if wm.get("detected") else "No")
        with col2:
            auth = wm.get("authenticity_score", 0.0)
            st.metric("Authenticity Score", f"{auth:.2f}/100")
            if wm.get("location"):
                st.caption(f"Location: {wm['location']}")
    
    # Stage 5: Layout
    with st.expander("📐 Stage 5: Layout Analysis", expanded=False):
        layout = results.get("layout", {})
        col1, col2 = st.columns([2, 1])
        with col1:
            if layout.get("valid") and layout.get("visualization_path"):
                vis_path = layout["visualization_path"]
                if Path(vis_path).exists():
                    st.image(vis_path, caption="Layout Visualization", use_container_width=True)
        with col2:
            score = layout.get("score", 0.0)
            st.metric("Layout Score", f"{score:.2%}")
            features = layout.get("features", {})
            st.caption(f"Text Regions: {features.get('text_regions', 0)}")
            st.caption(f"Alignment: {features.get('alignment_score', 0):.2%}")
    
    # Stage 6: ML Model
    with st.expander("🤖 Stage 6: ML Model Classification", expanded=False):
        ml = results.get("ml_model", {})
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Prediction", ml.get("class", "UNKNOWN"))
            st.metric("Confidence", f"{ml.get('confidence', 0.0):.2%}")
        with col2:
            probs = ml.get("probabilities", {})
            st.caption(f"Genuine: {probs.get('genuine', 0):.2%}")
            st.caption(f"Fake: {probs.get('fake', 0):.2%}")
    
    # Stage 7: Validation Summary
    with st.expander("✅ Stage 7: Final Validation Summary", expanded=True):
        render_premium_validation_dashboard(st.session_state.final_verdict)


# Sidebar: File upload
st.sidebar.header("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image or PDF file",
    type=['png', 'jpg', 'jpeg', 'pdf'],
    help="Upload a document image or PDF to analyze"
)

# Main content area
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Uploaded Document")
        
        # Check if file is PDF or image
        file_extension = uploaded_file.name.split('.')[-1].lower() if uploaded_file.name else ''
        is_pdf = file_extension == 'pdf'
        image = None
        
        if is_pdf:
            # Save PDF temporarily and convert to image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                tmp_pdf.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_pdf.name
            
            # Convert first page of PDF to image
            image = convert_pdf_to_image(tmp_pdf_path, page_number=0)
            if image is None:
                st.error("❌ Failed to convert PDF to image")
                safe_delete_file(tmp_pdf_path)
            else:
                st.image(image, caption="Uploaded Document (PDF - First Page)", use_container_width=True)
                # Clean up temporary PDF file
                safe_delete_file(tmp_pdf_path)
        else:
            # Handle image files
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Failed to open image: {str(e)}")
                image = None
    
    if image is not None and st.button("🚀 Analyze Document", type="primary", use_container_width=True):
        # Save uploaded file temporarily
        if is_pdf:
            # Save PDF and convert to image for analysis
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                tmp_pdf.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_pdf.name
            
            # Convert PDF to image
            converted_image = convert_pdf_to_image(tmp_pdf_path, page_number=0)
            if converted_image is None:
                st.error("❌ Failed to convert PDF to image for analysis")
                safe_delete_file(tmp_pdf_path)
            else:
                # Save converted image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                    converted_image.save(tmp_img.name)
                    tmp_path = tmp_img.name
                
                # Close the converted image to release file handle
                converted_image.close()
                
                # Clean up PDF file
                safe_delete_file(tmp_pdf_path)
                
                try:
                    results, verdict = run_analysis_pipeline(tmp_path)
                    st.session_state.analysis_results = results
                    st.session_state.final_verdict = verdict
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    logger.error(f"Analysis pipeline failed: {e}")
                    st.error(f"❌ Error during processing: {str(e)}")
                finally:
                    # Add a small delay to ensure all file handles are closed
                    time.sleep(0.1)
                    safe_delete_file(tmp_path)
        else:
            # Save image file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Note: Don't close 'image' here as it's used for display above
            # The saved file handle is already closed by the context manager
            
            try:
                results, verdict = run_analysis_pipeline(tmp_path)
                st.session_state.analysis_results = results
                st.session_state.final_verdict = verdict
                st.success("✅ Analysis complete!")
            except Exception as e:
                logger.error(f"Analysis pipeline failed: {e}")
                st.error(f"❌ Error during processing: {str(e)}")
            finally:
                # Add a small delay to ensure all file handles are closed
                time.sleep(0.1)
                safe_delete_file(tmp_path)
    
    # Display results if available
    if st.session_state.final_verdict:
        st.divider()
        render_verdict_display(st.session_state.final_verdict)
        st.divider()
        render_stage_results(st.session_state.analysis_results)
