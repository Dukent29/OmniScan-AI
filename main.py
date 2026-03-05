# FILE: main.py
# ROLE: The User Interface (Streamlit). Connects AI and MongoDB.

import streamlit as st
import os
from database_manager import save_analysis, get_all_records, delete_record, save_training_entry
from dataset_manager import add_dataset_entries, add_dataset_entry, ALLOWED_TYPES

# --- PAGE CONFIG ---
st.set_page_config(page_title="NFS AI Classifier", layout="wide")

# --- MODERN DARK THEME CSS ---
st.markdown(
    """
    <style>
    :root{
        --bg: #0B0F17;
        --surface: rgba(255,255,255,0.04);
        --surface-2: rgba(255,255,255,0.06);
        --border: rgba(255,255,255,0.10);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.62);
        --muted-2: rgba(255,255,255,0.45);
        --accent: #6EE7FF;
        --accent-2: #A78BFA;
        --danger: #FF5C7A;
        --success: #34D399;
        --warning: #FBBF24;
        --radius: 16px;
    }

    /* App background + main width */
    .stApp { background: var(--bg); color: var(--text); }
.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
}

    /* Titles */
    h1, h2, h3, .stMarkdown, .stText { color: var(--text); }
    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.25rem 0 0.1rem 0;
        color: var(--text);
    }
    .section-sub { color: var(--muted); margin-bottom: 0.9rem; }

    /* Subtle separator */
    .soft-sep { border-top: 1px solid var(--border); margin: 16px 0; opacity: 0.7; }

    /* Result card */
    .result-card{
        background: radial-gradient(1200px 600px at 10% 0%, rgba(110,231,255,0.10), transparent 40%),
                    radial-gradient(1200px 600px at 90% 20%, rgba(167,139,250,0.10), transparent 45%),
                    var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(6px);
    }
    .result-title{ font-size: 1.05rem; font-weight: 800; color: var(--text); }
    .result-line{ font-size: 0.95rem; color: var(--muted); margin-top: 6px; }
    .result-line b{ color: var(--text); font-weight: 700; }

    /* Streamlit containers (border=True) -> make them look like cards */
    div[data-testid="stContainer"][style]{
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 14px 14px 10px 14px;
    }

    /* Inputs */
    div[data-testid="stFileUploader"] section,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div,
    div[data-testid="stRadio"] div[role="radiogroup"]{
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
    }

    /* Expander header */
    details summary{
        background: transparent !important;
        color: var(--text) !important;
        font-weight: 750;
    }

    /* Chips */
    .chip{
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.04);
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 650;
        white-space: nowrap;
    }
    .chip.good{
        border-color: rgba(52,211,153,0.35);
        background: rgba(52,211,153,0.10);
        color: rgba(167,243,208,0.95);
    }
    .chip.warn{
        border-color: rgba(251,191,36,0.35);
        background: rgba(251,191,36,0.10);
        color: rgba(253,230,138,0.95);
    }

    /* Progress bar */
    .bar{
        width:100%;
        height:10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow:hidden;
        border: 1px solid var(--border);
        margin-bottom: 6px;
    }
    .bar > span{
        display:block;
        height:100%;
        width:0%;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }

    /* Buttons */
    div[data-testid="stButton"] > button{
        border-radius: 12px;
        font-weight: 750;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.04);
        color: var(--text);
        padding: 0.55rem 0.9rem;
    }
    div[data-testid="stButton"] > button:hover{
        border-color: rgba(110,231,255,0.45);
        background: rgba(110,231,255,0.08);
    }
    div[data-testid="stButton"] > button[kind="primary"]{
        background: linear-gradient(90deg, rgba(110,231,255,0.95), rgba(167,139,250,0.95));
        color: #061018;
        border: none;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover{
        filter: brightness(1.02);
    }
    div[data-testid="stButton"] > button[kind="secondary"]{
        border: 1px solid rgba(255,92,122,0.55);
        background: rgba(255,92,122,0.10);
        color: rgba(255,184,196,0.95);
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover{
        background: rgba(255,92,122,0.16);
    }

    /* Make captions less loud */
    .stCaption { color: var(--muted-2) !important; }

    </style>
    """,
    unsafe_allow_html=True,
)


def _build_fun_note(type_label: str, detail_label: str, match_score: float) -> str:
    low = (detail_label or "").strip().lower()
    if "baby yoda" in low or "grogu" in low:
        return f"Force Match unlocked for Grogu: {match_score:.2f}%"
    if type_label == "hand-signs":
        return f"Gesture sync score: {match_score:.2f}%"
    if type_label == "vehicles":
        return f"Garage match score: {match_score:.2f}%"
    return f"Visual match score: {match_score:.2f}%"


def _pct_from_any(value) -> float:
    """
    Converts: '0.934', 0.934, '93.4%', '93.4' -> 93.4
    If input is 0..1 we assume it's probability and scale to 0..100.
    """
    if value is None:
        return 0.0
    s = str(value).strip().replace("%", "")
    try:
        x = float(s)
    except ValueError:
        return 0.0
    if 0.0 <= x <= 1.0:
        return x * 100.0
    return x


def _bar(pct: float) -> str:
    pct = max(0.0, min(100.0, float(pct or 0.0)))
    return f"""
    <div class="bar"><span style="width:{pct:.0f}%"></span></div>
    <span class="chip">{pct:.0f}%</span>
    """


# --- TOASTS / LATEST RESULT ---
if "ui_feedback" in st.session_state:
    st.success(st.session_state["ui_feedback"])
    del st.session_state["ui_feedback"]

if "ui_result" in st.session_state:
    result = st.session_state["ui_result"]
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Latest AI Analysis</div>
            <div class="result-line"><b>Class:</b> {result['type']}</div>
            <div class="result-line"><b>Name (Label):</b> {result['label']}</div>
            <div class="result-line"><b>Description:</b> {result['description']}</div>
            <div class="result-line">
                <b>Type %:</b> {result['type_score']} |
                <b>Label %:</b> {result['label_score']} |
                <b>Match %:</b> {result['match_score']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(result["fun_note"])
    del st.session_state["ui_result"]


# --- HEADER ---
st.title("AI Image Multi-Classifier")
st.write("Upload an image to identify: Humans, Fictional Characters, Plants, Vehicles, or Animals.")


def render_workspace() -> None:
    st.markdown('<div class="section-title">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Analyze image and add training samples</div>', unsafe_allow_html=True)

    st.subheader("Analysis")
    source_mode = st.radio(
        label="source",
        options=["File", "Camera"],
        horizontal=True,
        label_visibility="collapsed",
        key="analysis_mode",
    )

    if source_mode == "File":
        input_file = st.file_uploader(
            label="upload",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed",
            key="analysis_upload",
        )
    else:
        input_file = st.camera_input(
            label="camera",
            label_visibility="collapsed",
            key="analysis_camera",
        )

    if input_file is not None:
        os.makedirs("temp_uploads", exist_ok=True)

        source_name = input_file.name if getattr(input_file, "name", None) else "camera_capture.jpg"
        file_path = os.path.join("temp_uploads", source_name)
        with open(file_path, "wb") as f:
            f.write(input_file.getbuffer())

        st.image(file_path, caption="Image to analyze", width=320)

        if st.button("Start AI Recognition", type="primary"):
            with st.spinner("AI is processing..."):
                from vision_engine import analyze_image, get_label_description

                type_label, type_confidence, detail_label, detail_confidence = analyze_image(file_path)
                detail_description = get_label_description(type_label, detail_label)
                file_size = os.path.getsize(file_path)

                save_analysis(
                    source_name,
                    file_size,
                    type_label,
                    type_confidence,
                    detail_label,
                    detail_confidence,
                    detail_description,
                    source_path=file_path,
                )

                match_score = round(((type_confidence + detail_confidence) / 2.0) * 100, 2)
                fun_note = _build_fun_note(type_label, detail_label, match_score)

                st.session_state["ui_feedback"] = (
                    f"Analysis saved. Type: {type_label} | Label: {detail_label} | "
                    f"Confidence: {round(type_confidence * 100, 2)}%"
                )
                st.session_state["ui_result"] = {
                    "type": type_label,
                    "label": detail_label,
                    "description": detail_description or "No description found for this label yet.",
                    "type_score": round(type_confidence * 100, 2),
                    "label_score": round(detail_confidence * 100, 2),
                    "match_score": match_score,
                    "fun_note": fun_note,
                }
                st.rerun()

    st.markdown('<div class="soft-sep"></div>', unsafe_allow_html=True)

    st.subheader("Add Training Data")
    dataset_images = st.file_uploader(
        "Add images to dataset",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="dataset_uploader_multi",
    )

    if not dataset_images:
        st.info("Select one or multiple images to start.")
        return

    st.caption(f"{len(dataset_images)} file(s) selected.")
    entries = []
    for i, img in enumerate(dataset_images):
        with st.expander(f"Metadata: {img.name}", expanded=(i == 0)):
            class_name = st.selectbox("Type", options=ALLOWED_TYPES, key=f"type_{i}")
            label = st.text_input("Label", key=f"label_{i}", placeholder="Example: open hand, Batman, lion, motorcycle")
            description = st.text_area("Description", key=f"desc_{i}", placeholder="Short context for training/reference.")
        entries.append(
            {
                "image_bytes": img.getvalue(),
                "original_name": img.name,
                "class_name": class_name,
                "label": label,
                "description": description,
            }
        )

    if st.button("Add All To Dataset", type="primary"):
        try:
            saved_paths = add_dataset_entries(entries)
            for entry, saved_path in zip(entries, saved_paths):
                save_training_entry(
                    file_name=saved_path.name,
                    file_size=saved_path.stat().st_size,
                    class_name=entry["class_name"],
                    label=entry.get("label", ""),
                    description=entry.get("description", ""),
                    source_path=str(saved_path),
                )
            st.session_state["ui_feedback"] = (
                f"Saved {len(saved_paths)} image(s) to dataset. Next step: run `python train_model.py`."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to add dataset entries: {exc}")


def render_history() -> None:
    st.markdown('<div class="section-title">History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Structured database records and actions</div>', unsafe_allow_html=True)

    history = get_all_records()
    if not history:
        st.info("The database is currently empty.")
        return

    st.caption("Recent analyses (newest first)")
    st.markdown('<div class="soft-sep"></div>', unsafe_allow_html=True)

    # Feed-like rows (more modern than fake table)
    for record in history:
        analysis = record.get("Analysis", {})
        record_id = str(record["_id"])
        source_path = record.get("SourcePath")

        predicted_type = analysis.get("Type")
        predicted_label = analysis.get("Label") or predicted_type

        # Convert stored values to %
        type_pct = _pct_from_any(analysis.get("SuccessRate"))
        label_pct = _pct_from_any(analysis.get("LabelConfidence"))
        match_pct = _pct_from_any(analysis.get("MatchScore"))

        # Confidence tags (tiny but useful)
        tag_class = "good" if match_pct >= 80 else ("warn" if match_pct >= 55 else "")
        class_chip = f'<span class="chip">{predicted_type or "unknown"}</span>'
        label_chip = f'<span class="chip">{(predicted_label or "unknown")}</span>'
        match_chip = f'<span class="chip {tag_class}">match {match_pct:.0f}%</span>' if match_pct else '<span class="chip">match —</span>'

        with st.container(border=True):
            top = st.columns([0.16, 0.52, 0.32], vertical_alignment="center")

            # Thumbnail
            with top[0]:
                if source_path and os.path.exists(source_path):
                    st.image(source_path, width=72)
                else:
                    st.markdown('<span class="chip">no image</span>', unsafe_allow_html=True)

            # Main info
            with top[1]:
                st.markdown(
                    f"**{record.get('Name') or 'Unnamed file'}**",
                )
                st.caption(f"{record.get('Date')}")
                st.markdown(f"{class_chip} {label_chip} {match_chip}", unsafe_allow_html=True)
                desc = analysis.get("Description") or ""
                if desc.strip():
                    st.write(desc)

            # Actions
            with top[2]:
                a = st.columns([1, 1], vertical_alignment="center")
                use_clicked = a[0].button("Use", key=f"use_{record_id}", help="Use this result for training", type="primary")
                del_clicked = a[1].button("Delete", key=f"del_{record_id}", help="Delete this record", type="secondary")

                # Score bars under actions
                st.caption("Type confidence")
                st.markdown(_bar(type_pct), unsafe_allow_html=True)
                st.caption("Label confidence")
                st.markdown(_bar(label_pct), unsafe_allow_html=True)
                st.caption("Match")
                st.markdown(_bar(match_pct), unsafe_allow_html=True)

            if use_clicked:
                if not source_path or not os.path.exists(source_path):
                    st.warning("Cannot add to training: source image file is not available for this record.")
                elif predicted_type not in ALLOWED_TYPES:
                    st.warning(f"Type '{predicted_type}' is not in allowed dataset classes: {ALLOWED_TYPES}")
                else:
                    with open(source_path, "rb") as fh:
                        image_bytes = fh.read()

                    add_dataset_entry(
                        image_bytes=image_bytes,
                        original_name=record.get("Name") or os.path.basename(source_path),
                        class_name=predicted_type,
                        label=predicted_label,
                        description=f"Validated from app history record {record_id}.",
                    )
                    save_training_entry(
                        file_name=record.get("Name") or os.path.basename(source_path),
                        file_size=len(image_bytes),
                        class_name=predicted_type,
                        label=predicted_label,
                        description=f"Validated from app history record {record_id}.",
                        source_path=source_path,
                    )
                    st.session_state["ui_feedback"] = (
                        "Validated sample added to dataset. Run `python train_model.py` to update the model."
                    )
                    st.rerun()

            if del_clicked:
                delete_record(record["_id"])
                st.toast(f"Deleted {record.get('Name')}")
                st.rerun()

            with st.expander("Correct & Add", expanded=False):
                corrected_type = st.selectbox(
                    "Correct Type",
                    options=ALLOWED_TYPES,
                    index=ALLOWED_TYPES.index(predicted_type) if predicted_type in ALLOWED_TYPES else 0,
                    key=f"corr_type_{record_id}",
                )
                corrected_label = st.text_input(
                    "Correct Label",
                    value=predicted_label or "",
                    key=f"corr_label_{record_id}",
                )
                corrected_desc = st.text_area(
                    "Description",
                    value="",
                    key=f"corr_desc_{record_id}",
                    placeholder="Example: left hand showing L sign on plain background.",
                )
                replacement_file = st.file_uploader(
                    "Optional: upload replacement image if original source is missing",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    key=f"corr_file_{record_id}",
                )

                if st.button("Add Corrected Sample", key=f"corr_add_{record_id}", type="primary"):
                    os.makedirs("temp_uploads", exist_ok=True)
                    fallback_path = os.path.join("temp_uploads", str(record.get("Name") or ""))

                    usable_path = source_path if source_path and os.path.exists(source_path) else None
                    if usable_path is None and os.path.exists(fallback_path):
                        usable_path = fallback_path

                    if replacement_file is not None:
                        image_bytes = replacement_file.getvalue()
                        original_name = replacement_file.name
                        used_source_path = None
                    elif usable_path is not None:
                        with open(usable_path, "rb") as fh:
                            image_bytes = fh.read()
                        original_name = record.get("Name") or os.path.basename(usable_path)
                        used_source_path = usable_path
                    else:
                        st.warning("No source image found. Upload a replacement image to add corrected sample.")
                        image_bytes = None
                        original_name = None
                        used_source_path = None

                    if image_bytes is not None:
                        add_dataset_entry(
                            image_bytes=image_bytes,
                            original_name=original_name,
                            class_name=corrected_type,
                            label=(corrected_label or corrected_type),
                            description=corrected_desc,
                        )
                        save_training_entry(
                            file_name=original_name,
                            file_size=len(image_bytes),
                            class_name=corrected_type,
                            label=(corrected_label or corrected_type),
                            description=corrected_desc,
                            source_path=used_source_path or source_path,
                        )
                        st.session_state["ui_feedback"] = (
                            "Corrected sample added to dataset. Run `python train_model.py` to update the model."
                        )
                        st.rerun()


# --- MAIN LAYOUT ---
left_col, right_col = st.columns([0.95, 1.55], gap="large")
with left_col:
    with st.container(border=True):
        render_workspace()

with right_col:
    with st.container(border=True):
        render_history()