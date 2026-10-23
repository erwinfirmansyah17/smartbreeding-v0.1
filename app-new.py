"""SmartBreeding: fungsi appp.py dengan tampilan sage dan emas."""
import os
import glob
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import base64
from html import escape

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ModuleNotFoundError:
    REMBG_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ModuleNotFoundError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False


LOGO_PATH = os.path.join(os.path.dirname(__file__), "SmartBreeding_V.01-removebg-preview.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = os.path.join(os.path.dirname(__file__), "SmartBreeding V.01.png")

st.set_page_config(
    page_title="SmartBreeding V.01",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)


THEME_CSS = """
<style>
:root { --green:#3e5956; --gold:#b18c36; --ink:#344f4c; --muted:#768780;
        --canvas:#e4eeec; --field:#edf3f1; --line:#d4e1dc; --surface:#ffffff; }
.stApp { background:var(--canvas); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1320px; padding:3.4rem 3rem 3rem; }
h1,h2,h3,p,label { color:var(--ink); }
h1 { font-size:2.2rem!important; font-weight:500!important; letter-spacing:-1px; }
h3 { font-size:1.15rem!important; font-weight:500!important; }
.eyebrow { color:var(--muted); font-size:10px; letter-spacing:2.4px; margin:12px 0 24px; }
.gold { color:var(--gold); }
.sb-brand,.brand { display:flex; align-items:center; gap:12px; padding:8px 0 28px; }
.sb-brand-mark { width:58px; flex-shrink:0; }
.sb-logo,.brand img { width:58px; height:58px; object-fit:contain; }
.sb-brand-title,.brand strong { font-size:20px; font-weight:550; letter-spacing:-.5px; }
.sb-brand-sub,.brand small { display:block; font-size:10px; opacity:.7; margin-top:6px; letter-spacing:1px; }
[data-testid="stSidebar"] { background:var(--green); border-right:0; }
[data-testid="stSidebar"] * { color:#edf4f0; }
[data-testid="stSidebar"] .stButton button { background:transparent; border:0; border-radius:0; padding:13px 16px; justify-content:flex-start; box-shadow:none; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background:#ffffff0c; border-left:2px solid #d5b76d; }
[data-testid="stSidebar"] .stButton button[kind="primary"] p { color:#e6c983; }
[data-testid="stSidebar"] .stButton button:hover { background:#ffffff12; }
[data-testid="stSidebar"] hr { border-color:#ffffff18; }
.sb-hero { padding:8px 0 30px; margin-bottom:28px; border-bottom:1px solid var(--line); }
.sb-hero-title { color:var(--ink); font-size:34px; font-weight:500; letter-spacing:-1px; }
.sb-hero-sub { color:var(--muted); font-size:14px; line-height:1.9; margin-top:12px; max-width:760px; }
.sb-card { padding:14px 0 12px; min-height:100px; }
.sb-card-title { font-size:15px; font-weight:550; margin-bottom:10px; color:var(--ink); }
.sb-muted,.sb-metric-note { color:var(--muted); font-size:12px; line-height:1.9; }
.sb-section { color:var(--ink); font-size:12px; font-weight:600; letter-spacing:1px; margin:38px 0 18px; }
.sb-metric { padding:8px 0 24px; border-bottom:1px solid var(--line); }
.sb-metric-label { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:1.8px; }
.sb-metric-value { font-size:42px; font-weight:400; margin:10px 0 4px; letter-spacing:-1.5px; color:var(--ink); }
.sb-pill { display:inline-block; color:var(--gold); font-size:26px; font-weight:400; padding:0 0 16px; }
.sb-input-title { font-size:11px; color:var(--muted); margin-bottom:10px; }
[data-testid="stForm"] { background:transparent; border:0; padding:12px 0 20px; }
[data-testid="stMetric"] { background:transparent; padding:16px 0; border-bottom:1px solid var(--line); }
[data-testid="stMetricValue"] { font-weight:400; }
[data-testid="stTextInputRootElement"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background:var(--field)!important; border-radius:6px!important; border-color:transparent!important; color:var(--ink)!important; }
input,textarea { color:var(--ink)!important; }
.stButton button,.stDownloadButton button,.stFormSubmitButton button {
    border-radius:6px; min-height:42px; border:1px solid var(--line); color:var(--ink); background:transparent; box-shadow:none; }
.stButton button[kind="primary"],.stFormSubmitButton button { background:var(--green); border-color:var(--green); color:#fff; }
.stButton button:hover,.stDownloadButton button:hover { border-color:var(--gold); color:var(--gold); }
.stButton button:focus-visible,.stDownloadButton button:focus-visible { outline:2px solid var(--gold); outline-offset:3px; }
[data-testid="stFileUploader"] { padding:8px 0; }
[data-testid="stFileUploaderDropzone"] { background:var(--field); border:1px dashed var(--line); border-radius:6px; }
[data-testid="stImage"] { border-radius:6px; overflow:hidden; }
[data-testid="stDataFrame"] { border-radius:4px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:28px; background:transparent; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius:0; }
.stTabs [aria-selected="true"] { color:var(--gold); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--gold); }
[data-testid="stAlert"] { border-radius:4px; }
.login-hero { padding:12px 44px 20px 0; min-height:530px; border-right:1px solid var(--line); }
.login-hero h1 { color:var(--ink); font-family:Georgia,serif; font-size:48px!important; font-weight:400!important; line-height:1.18; max-width:440px; margin:22px 0; }
.login-hero p { color:var(--muted); line-height:1.9; max-width:350px; font-size:14px; }
.login-hero.register-hero { border-right:0; border-left:1px solid var(--line); padding:12px 0 20px 44px; }
.landscape { margin:20px 0; width:100%; max-width:420px; height:135px; }
hr { border-color:var(--line); }
@media(max-width:768px) {
    .block-container { padding:3rem 1.2rem 2rem; }
    .login-hero { min-height:auto; border-right:0; border-bottom:1px solid var(--line); padding:0 0 22px; }
    .login-hero.register-hero { border-left:0; padding:22px 0 0; border-bottom:0; border-top:1px solid var(--line); }
    .login-hero h1 { font-size:36px!important; }
    .sb-hero-title { font-size:28px; }
    .sb-metric-value { font-size:34px; }
}

/* Explicit foregrounds keep nested Markdown text readable on dark surfaces. */
.stButton button[kind="primary"] p,.stFormSubmitButton button p,
.stButton button[kind="primary"] svg,.stFormSubmitButton button svg { color:#fff!important; }
[data-testid="stSidebar"] { border-radius:0 28px 28px 0; }
[data-testid="stSidebar"] .stButton button { border-radius:24px; }
[data-testid="stSidebar"] .stButton button p { color:#edf4f0!important; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background:#e4eeec; border-left:0; }
[data-testid="stSidebar"] .stButton button[kind="primary"] p { color:#344f4c!important; }
.st-key-dashboard_summary { background:var(--surface); border-radius:26px; padding:24px 28px; margin-bottom:14px; }
.st-key-dashboard_summary .sb-metric { border:0; padding:4px 0; }
.st-key-dashboard_summary .sb-metric-value { font-size:32px; }
.st-key-dashboard_list { background:var(--surface); border-radius:24px; padding:10px 24px 20px; }
.st-key-dashboard_list hr { margin:8px 0; border-top-style:dashed; }
.st-key-dashboard_list .stButton button { border-radius:24px; background:#c89c2c; border:0; }
.st-key-dashboard_list .stButton button p { color:#263d37!important; font-weight:600; font-size:12px; }
.dash-row { min-height:96px; padding:18px 0; }
.dash-row small { display:block; font-size:10px; color:var(--muted); letter-spacing:1px; margin-bottom:8px; }
.dash-row strong { font-size:18px; color:var(--ink); font-weight:500; }
.dash-row p { font-size:12px; color:var(--muted); margin:6px 0 0; line-height:1.7; }
.dash-panel { background:#3e5956; border-radius:24px; padding:30px 26px; min-height:436px; color:#f6faf8; }
.dash-panel h3,.dash-panel p,.dash-panel strong,.dash-panel span { color:#f6faf8!important; }
.dash-panel small { color:#d7e5df; font-size:10px; letter-spacing:1.5px; }
.dash-panel h3 { font-size:23px!important; margin:16px 0 24px; line-height:1.4; }
.dash-panel p { font-size:12px; line-height:1.9; }
.dash-flow { padding:18px 0 24px; }
.dash-step { padding:12px 0 12px 20px; border-left:1px solid #b9a45f; font-size:13px; }
.dash-step b { color:#ecd086; font-weight:500; margin-right:12px; }
.dash-setting { border-top:1px solid #ffffff30; display:flex; justify-content:space-between; padding-top:22px; margin-top:14px; }
.dash-setting strong { display:block; font-size:24px; font-weight:400; margin-top:7px; }
@media(max-width:768px) { .st-key-dashboard_summary,.st-key-dashboard_list { padding:18px; } .dash-panel { min-height:auto; } }

/* Motion uses opacity/transform so it does not shift the surrounding layout. */
@keyframes sb-enter { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
@keyframes sb-from-left { from { opacity:0; transform:translateX(-24px); } to { opacity:1; transform:translateX(0); } }
@keyframes sb-from-right { from { opacity:0; transform:translateX(24px); } to { opacity:1; transform:translateX(0); } }
.st-key-auth_login { animation:sb-from-right 520ms cubic-bezier(.22,1,.36,1) both; }
.st-key-auth_register { animation:sb-from-left 520ms cubic-bezier(.22,1,.36,1) both; }
.login-hero { animation:sb-from-left 580ms cubic-bezier(.22,1,.36,1) both; }
.login-hero.register-hero { animation-name:sb-from-right; }
.st-key-page_dashboard,.st-key-page_dataset,.st-key-page_deteksi,
.st-key-page_pengaturan,.st-key-page_profil { animation:sb-enter 420ms cubic-bezier(.22,1,.36,1) both; }
.st-key-dashboard_summary { animation:sb-enter 440ms 40ms cubic-bezier(.22,1,.36,1) both; }
.st-key-dashboard_list { animation:sb-enter 480ms 90ms cubic-bezier(.22,1,.36,1) both; }
.dash-panel { animation:sb-enter 500ms 140ms cubic-bezier(.22,1,.36,1) both; }
.stButton button,.stDownloadButton button,.stFormSubmitButton button {
    transition:background-color 180ms ease,border-color 180ms ease,color 180ms ease,transform 180ms ease,box-shadow 180ms ease;
}
.stButton button:active,.stDownloadButton button:active,.stFormSubmitButton button:active { transform:scale(.98); }
[data-testid="stTextInputRootElement"],[data-baseweb="select"] > div,[data-baseweb="textarea"] {
    transition:box-shadow 180ms ease,background-color 180ms ease;
}
[data-testid="stTextInputRootElement"]:focus-within,[data-baseweb="textarea"]:focus-within {
    box-shadow:0 0 0 2px #b18c3640;
}
[data-testid="stImage"] img { transition:transform 320ms cubic-bezier(.22,1,.36,1); }
.stTabs [data-baseweb="tab"] { transition:color 180ms ease,background-color 180ms ease; }
@media(hover:hover) and (pointer:fine) {
    .stButton button:hover,.stDownloadButton button:hover,.stFormSubmitButton button:hover { transform:translateY(-2px); box-shadow:0 5px 14px #243c3710; }
    [data-testid="stSidebar"] .stButton button:hover { transform:translateX(3px); box-shadow:none; }
    [data-testid="stImage"]:hover img { transform:scale(1.025); }
}
@media(prefers-reduced-motion:reduce) {
    .stApp *,.stApp *::before,.stApp *::after { animation:none!important; transition:none!important; scroll-behavior:auto!important; }
    .stApp button,.stApp [data-testid="stImage"] img { transform:none!important; }
}
</style>
"""


def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    if st.session_state.get("settings", {}).get("theme") == "Dark":
        st.markdown("""<style>
        :root { --ink:#e4eeec; --muted:#a8beb5; --canvas:#192f2b;
                --field:#29423e; --line:#39534b; --gold:#d3b46b; --surface:#243c37; }
        </style>""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "authenticated": False,
        "page": "dashboard",
        "user_name": "Akun Sapi",
        "user_role": "Akun",
        "gender": "Perempuan",
        "address": "Lab Monitoring Sapi",
        "profile_image": None,
        "show_register": False,
        "active_account": None,
        "accounts": {
            "smartbreeding": "smartbreeding123",
        },
        "account_data": {},
        "settings": {
            "conf_threshold": 0.25,
            "iou_threshold": 0.50,
            "theme": "Light",
            "notifications": True,
            "account_username": "smartbreeding",
            "account_password": "smartbreeding123",
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "cattle_data" not in st.session_state:
        st.session_state.cattle_data = pd.DataFrame(
            [
                {"ID Sapi": "S-001", "Nama": "Dewi", "Jenis": "Sapi Perah", "Kelamin": "Betina", "Umur": 2, "BB (kg)": 420, "Kondisi": "Sehat"},
                {"ID Sapi": "S-002", "Nama": "Bimo", "Jenis": "Sapi Potong", "Kelamin": "Jantan", "Umur": 3, "BB (kg)": 500, "Kondisi": "Sehat"},
                {"ID Sapi": "S-003", "Nama": "Citra", "Jenis": "Sapi Perah", "Kelamin": "Betina", "Umur": 4, "BB (kg)": 470, "Kondisi": "Cekal"},
            ]
        )

    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None


def get_default_account_state(username=None):
    username = username or "akun"
    return {
        "user_name": username,
        "user_role": "Akun",
        "gender": "Perempuan",
        "address": "Lab Monitoring Sapi",
        "profile_image": None,
        "page": "dashboard",
        "detection_history": [],
        "latest_result": None,
    }


def sync_active_account_state():
    active_account = st.session_state.get("active_account")
    if not active_account:
        return
    st.session_state.account_data[active_account] = {
        "user_name": st.session_state.get("user_name", active_account),
        "user_role": st.session_state.get("user_role", "Akun"),
        "gender": st.session_state.get("gender", "Perempuan"),
        "address": st.session_state.get("address", "Lab Monitoring Sapi"),
        "profile_image": st.session_state.get("profile_image"),
        "page": st.session_state.get("page", "dashboard"),
        "detection_history": list(st.session_state.get("detection_history", [])),
        "latest_result": st.session_state.get("latest_result"),
    }


def load_account_state(username):
    state_store = st.session_state.setdefault("account_data", {})
    account_state = state_store.setdefault(username, get_default_account_state(username))
    st.session_state.active_account = username
    st.session_state.user_name = account_state.get("user_name", username)
    st.session_state.user_role = account_state.get("user_role", "Akun")
    st.session_state.gender = account_state.get("gender", "Perempuan")
    st.session_state.address = account_state.get("address", "Lab Monitoring Sapi")
    st.session_state.profile_image = account_state.get("profile_image")
    st.session_state.page = account_state.get("page", "dashboard")
    st.session_state.detection_history = list(account_state.get("detection_history", []))
    st.session_state.latest_result = account_state.get("latest_result")


def get_default_profile_image():
    return None


def display_profile_avatar(size=48):
    image_bytes = st.session_state.get("profile_image")
    if image_bytes:
        return st.image(Image.open(__import__('io').BytesIO(image_bytes)), width=size)
    return st.markdown(f"<div style='width:{size}px;height:{size}px;border-radius:50%;background:#1f77b4;color:white;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:18px;'> {st.session_state.user_name[:1].upper()} </div>", unsafe_allow_html=True)


def find_model_path():
    base_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(base_dir, "model", "best.pt"),
        os.path.join(base_dir, "weights", "best.pt"),
        os.path.join(base_dir, "best.pt"),
        r"D:\Kuliah\RISET\YOLO_Training\YOLO11s_Cow_Belly\weights\best.pt",
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            return path
    return None


MODEL_PATH = find_model_path()
MODEL_ERROR = None


@st.cache_resource
def load_model():
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Package 'ultralytics' belum terpasang.")
    if MODEL_PATH is None:
        raise FileNotFoundError("Model YOLO tidak ditemukan. Letakkan file best.pt di folder model atau weights.")
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_rembg_session():
    if not REMBG_AVAILABLE:
        raise ImportError("Package 'rembg' belum terpasang.")
    return new_session("u2net")


def remove_background(image: Image.Image, session):
    if not REMBG_AVAILABLE:
        return image
    image_rgba = image.convert("RGBA")
    return remove(image_rgba, session=session)


try:
    model = load_model()
except Exception as exc:
    model = None
    MODEL_ERROR = str(exc)


def draw_text_outline(img, text, position, text_color, font_scale=0.55, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_measurement_yolo(img, x1, y1, x2, y2, class_name, confidence, color):
    width = x2 - x1
    height = y2 - y1
    diagonal = np.sqrt(width ** 2 + height ** 2)
    area = width * height

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.line(img, (x1, y1), (x2, y1), color, 2)
    cv2.line(img, (x1, y1), (x1, y2), color, 2)

    label = f"{class_name} {confidence:.2f}"
    draw_text_outline(img, label, (x1 + 5, y1 + 22), color)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if class_name == "Cow":
        width_text = f"W: {width}px"
    else:
        width_text = f"BW: {width}px"

    (tw, _), _ = cv2.getTextSize(width_text, font, 0.50, 2)
    draw_text_outline(img, width_text, (int((x1 + x2 - tw) / 2), y1 + 22), (255, 255, 255))

    if class_name == "Cow":
        height_text = f"H: {height}px"
    else:
        height_text = f"BH: {height}px"

    draw_text_outline(img, height_text, (x1 + 5, int((y1 + y2) / 2)), (255, 255, 255))

    if class_name == "Cow":
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        diagonal_text = f"D: {diagonal:.0f}px"
        (tw, _), _ = cv2.getTextSize(diagonal_text, font, 0.50, 2)
        draw_text_outline(img, diagonal_text, (int((x1 + x2 - tw) / 2), int((y1 + y2) / 2)), (255, 255, 0))

    if class_name == "Cow":
        area_text = f"A: {area:,.0f} px²"
    else:
        area_text = f"BA: {area:,.0f} px²"

    draw_text_outline(img, area_text, (x1 + 5, y2 - 8), (255, 0, 255))

    return {
        "Class": class_name,
        "Confidence": round(confidence, 4),
        "Width (px)": round(width, 2),
        "Height (px)": round(height, 2),
        "Diagonal (px)": round(diagonal, 2),
        "Area (px²)": round(area, 2),
    }


def process_detection(uploaded_file, conf_threshold, iou_threshold):
    image = Image.open(uploaded_file).convert("RGB")

    if REMBG_AVAILABLE:
        with st.spinner("Menghapus background..."):
            session = load_rembg_session()
            bg_removed_image = remove_background(image, session)
    else:
        bg_removed_image = image

    image_np = np.array(bg_removed_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if model is None:
        raise RuntimeError(MODEL_ERROR or "Model YOLO belum tersedia.")

    with st.spinner("Sedang melakukan deteksi..."):
        results = model.predict(source=image_bgr, conf=conf_threshold, iou=iou_threshold, imgsz=640, verbose=False)

    result = results[0]
    output_img = image_bgr.copy()
    measurement_data = []

    for box in result.boxes:
        cls = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if cls == 0:
            class_name = "Belly"
            color = (255, 0, 0)
        elif cls == 1:
            class_name = "Cow"
            color = (0, 255, 0)
        else:
            class_name = str(cls)
            color = (255, 255, 255)

        measurement_data.append(draw_measurement_yolo(output_img, x1, y1, x2, y2, class_name, confidence, color))

    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    df = pd.DataFrame(measurement_data)

    return {
        "input_image": image,
        "processed_image": bg_removed_image,
        "output_rgb": output_rgb,
        "df": df,
        "summary": {
            "cow_count": int((df["Class"] == "Cow").sum()),
            "belly_count": int((df["Class"] == "Belly").sum()),
        },
    }


DATASET_PATH = r"D:\Kuliah\RISET\Data\images"


def list_dataset_files():
    from pathlib import Path
    folder = Path(DATASET_PATH)
    if not folder.is_dir():
        return []
    excluded_names = {"3_jpeg.rf.f0eb03fbdd9f1fad8f84e384168aade7.jpg"}
    return sorted(str(path) for path in folder.rglob("*")
                  if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                  and path.name not in excluded_names)


def render_dataset_gallery(file_list, limit=None):
    sample_files = file_list if limit is None else file_list[:limit]
    if not sample_files:
        st.info("Belum ada file dataset yang tersedia.")
        return []

    preview_items = []
    row_size = 5
    target_size = (180, 180)
    for start_index in range(0, len(sample_files), row_size):
        row_files = sample_files[start_index:start_index + row_size]
        cols = st.columns(len(row_files))
        for col, file_path in zip(cols, row_files):
            try:
                image = Image.open(file_path).convert("RGB")
                image = ImageOps.pad(image, target_size, color=(255, 255, 255))
                with col:
                    st.image(image, width=180)
                preview_items.append({"Nama File": os.path.basename(file_path), "Path": file_path})
            except Exception:
                continue
    return preview_items


def build_merged_history_df():
    rows = []
    for item in st.session_state.detection_history:
        for record in item.get("rows", []):
            merged = dict(record)
            merged["timestamp"] = item.get("timestamp")
            merged["image_name"] = item.get("image_name")
            rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def login_page():
    registering = st.session_state.get("show_register", False)
    left, right = st.columns([1, 1.1] if registering else [1.1, 1], gap="large")
    form_column = left if registering else right
    brand_column = right if registering else left
    hero_class = "login-hero register-hero" if registering else "login-hero"
    with brand_column:
        logo_html = ""
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as logo_file:
                logo_html = f'<img src="data:image/png;base64,{base64.b64encode(logo_file.read()).decode()}" alt="SmartBreeding">'
        st.markdown(f'''<div class="{hero_class}">
        <div class="brand">{logo_html}<div><strong>SmartBreeding</strong><small>SMARTER FARM. BETTER FUTURE.</small></div></div>
        <div class="eyebrow gold">YOUR FARM, CONNECTED</div>
        <h1>Peternakan lebih<br>cerdas, dimulai<br>dari sini<span class="gold">.</span></h1>
        <p>Pantau ternak, analisis citra, dan kelola hasil pengukuran dalam satu ruang kerja.</p>
        <svg class="landscape" viewBox="0 0 460 150" fill="none" aria-hidden="true">
        <circle cx="356" cy="38" r="24" stroke="#c89c2c" stroke-width="1.5"/>
        <path d="M0 115 Q110 25 240 110 T480 90 M0 137 Q140 55 300 130 T480 115 M0 155 Q140 90 310 150" stroke="#91aba0" stroke-width="1.2"/>
        <path d="M95 105V64L135 36L175 64V105 M86 66L135 29L184 66 M120 98V72H150V104" stroke="#c89c2c" stroke-width="2"/>
        </svg><div class="eyebrow" style="color:var(--muted)">PRECISION ? CARE ? GROWTH</div></div>''', unsafe_allow_html=True)
    with form_column, st.container(key="auth_register" if registering else "auth_login"):
        st.markdown('<div class="eyebrow">WELCOME TO YOUR WORKSPACE</div>', unsafe_allow_html=True)
        if st.session_state.get("show_register", False):
            st.markdown("### Daftar Akun Baru")
            st.caption("Buat akun baru untuk masuk ke SmartBreeding.")

            with st.form("register_form"):
                full_name = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap")
                username = st.text_input("Username Baru", placeholder="Masukkan username baru")
                password = st.text_input("Password", type="password", placeholder="Masukkan password")
                confirm_password = st.text_input("Konfirmasi Password", type="password", placeholder="Ulangi password")
                submitted = st.form_submit_button("Buat Akun", use_container_width=True)

            if submitted:
                if not full_name or not username or not password:
                    st.error("Semua field wajib diisi.")
                elif username in st.session_state.accounts:
                    st.error("Username sudah terdaftar, pilih username lain.")
                elif password != confirm_password:
                    st.error("Password dan konfirmasi password tidak sama.")
                else:
                    previous_user = st.session_state.get("active_account")
                    if previous_user and previous_user != username:
                        sync_active_account_state()

                    st.session_state.accounts[username] = password
                    st.session_state.settings["account_username"] = username
                    st.session_state.settings["account_password"] = password
                    st.session_state.user_name = full_name or username
                    st.session_state.user_role = "Akun"
                    st.session_state.gender = "Perempuan"
                    st.session_state.address = "Lab Monitoring Sapi"
                    st.session_state.profile_image = None
                    st.session_state.detection_history = []
                    st.session_state.latest_result = None
                    st.session_state.active_account = username
                    st.session_state.account_data.setdefault(username, get_default_account_state(username))
                    st.session_state.account_data[username].update({
                        "user_name": st.session_state.user_name,
                        "user_role": st.session_state.user_role,
                        "gender": st.session_state.gender,
                        "address": st.session_state.address,
                        "profile_image": None,
                        "page": "dashboard",
                        "detection_history": [],
                        "latest_result": None,
                    })
                    st.session_state.authenticated = True
                    st.session_state.page = "dashboard"
                    st.session_state.show_register = False
                    st.success("Akun berhasil dibuat dan login berhasil.")
                    st.rerun()

            if st.button("Sudah punya akun? Masuk", use_container_width=True):
                st.session_state.show_register = False
                st.rerun()
        else:
            st.markdown("## Selamat datang kembali.")
            st.caption("Silahkan masukkan username dan password untuk melanjutkan.")

            with st.form("login_form"):
                username = st.text_input(
                    "Username",
                    value=st.session_state.settings["account_username"],
                    placeholder="Masukkan username",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    value=st.session_state.settings["account_password"],
                    placeholder="Masukkan password",
                )
                submitted = st.form_submit_button("Masuk", use_container_width=True)

            if submitted:
                stored_password = st.session_state.accounts.get(username)
                if stored_password == password:
                    previous_user = st.session_state.get("active_account")
                    if previous_user and previous_user != username:
                        sync_active_account_state()

                    st.session_state.settings["account_username"] = username
                    st.session_state.settings["account_password"] = password
                    st.session_state.authenticated = True
                    load_account_state(username)
                    st.session_state.user_role = "Akun"
                    st.session_state.page = "dashboard"
                    st.success("Login berhasil.")
                    st.rerun()
                else:
                    st.error("Username atau password salah.")

            st.info("Akun demo: **smartbreeding / smartbreeding123**")

            if st.button("Belum punya akun? Daftar sekarang", use_container_width=True):
                st.session_state.show_register = True
                st.rerun()


def render_sidebar():
    with st.sidebar:
        logo_html = ""
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as image_file:
                logo_base64 = base64.b64encode(image_file.read()).decode()
            logo_html = f"""
                <img src="data:image/png;base64,{logo_base64}" style="width:100%;height:100%;object-fit:contain;display:block;border-radius:50%;background:transparent;" />
            """

        st.markdown(
            f"""
            <div class="sb-brand">
                <div class="sb-brand-mark">
                    <div class="sb-logo">{logo_html}</div>
                </div>
                <div class="sb-brand-text">
                    <div class="sb-brand-title">SmartBreeding</div>
                    <div class="sb-brand-sub">V.01 • Monitoring Sapi</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        col_avatar, col_info = st.columns([0.8, 2.2])
        with col_avatar:
            image_bytes = st.session_state.get("profile_image")
            if image_bytes:
                st.image(Image.open(__import__("io").BytesIO(image_bytes)), width=46)
            else:
                initial = st.session_state.user_name[:1].upper()
                st.markdown(
                    f"""
                    <div style="width:46px;height:46px;border-radius:50%;background:white;color:#087a5a;
                                display:flex;align-items:center;justify-content:center;font-weight:900;
                                font-size:18px;box-shadow:0 4px 14px rgba(0,0,0,.14);">{initial}</div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_info:
            st.markdown(f"**{st.session_state.user_name}**")
            st.caption("Akun aktif")

        st.markdown("---")

        menu_items = [
            ("dashboard", "Dashboard"),
            ("dataset", "Dataset"),
            ("deteksi", "Deteksi"),
            ("pengaturan", "Pengaturan"),
        ]

        for page_key, label in menu_items:
            if st.button(label, key=f"nav_{page_key}", type="primary" if st.session_state.page == page_key else "secondary", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")
        st.caption("SMARTBREEDING V.01")
        if st.button("Keluar", use_container_width=True):
            sync_active_account_state()
            st.session_state.authenticated = False
            st.session_state.active_account = None
            st.session_state.page = "login"
            st.rerun()


def dashboard_page():
    dataset_files = list_dataset_files()
    history = build_merged_history_df()
    st.markdown(f'<div class="eyebrow">SMARTBREEDING / DASHBOARD</div><div class="sb-hero-title">Selamat datang, {escape(st.session_state.user_name)}.</div><p class="sb-muted">Satu ruang untuk monitoring, analisis, dan pengukuran ternak Anda.</p>', unsafe_allow_html=True)
    with st.container(key="dashboard_summary"):
        for col, (label, value, note) in zip(st.columns(3), [
            ("Dataset", len(dataset_files), "Citra tersedia"),
            ("Analisis", len(st.session_state.detection_history), "Sesi deteksi"),
            ("Riwayat", len(history), "Objek terukur"),
        ]):
            with col:
                st.markdown(f'<div class="sb-metric"><div class="sb-metric-label">{label}</div><div class="sb-metric-value">{value}</div><div class="sb-metric-note">{note}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st.markdown('<div class="sb-section">RUANG KERJA</div>', unsafe_allow_html=True)
        with st.container(key="dashboard_list"):
            actions = [
                ("01 / DATASET", "Galeri citra sapi", "Telusuri gambar untuk monitoring dan pengujian model.", "Buka Dataset", "dataset"),
                ("02 / ANALISIS", "Deteksi Cow & Belly", "Hapus background, deteksi objek, dan ukur morfologi sapi.", "Mulai Deteksi", "deteksi"),
                ("03 / PREFERENSI", "Pengaturan workspace", "Kelola profil, akun, confidence, IoU, dan tema.", "Pengaturan", "pengaturan"),
            ]
            for index, (label, title, description, button, page) in enumerate(actions):
                content, action = st.columns([2.5, 1], vertical_alignment="center")
                with content:
                    st.markdown(f'<div class="dash-row"><small>{label}</small><strong>{title}</strong><p>{description}</p></div>', unsafe_allow_html=True)
                with action:
                    if st.button(button, key=f"quick_{page}", use_container_width=True):
                        st.session_state.page = page
                        st.rerun()
                if index < len(actions) - 1:
                    st.divider()
    with right:
        st.markdown('<div class="sb-section">ALUR ANALISIS</div>', unsafe_allow_html=True)
        settings = st.session_state.settings
        st.markdown(f'''<div class="dash-panel"><small>PRECISION LIVESTOCK</small>
        <h3>Dari citra menjadi<br>informasi terukur.</h3>
        <div class="dash-flow"><div class="dash-step"><b>01</b> Background removal</div>
        <div class="dash-step"><b>02</b> Deteksi Cow & Belly</div>
        <div class="dash-step"><b>03</b> Pengukuran morfologi</div></div>
        <p>Width, height, diagonal, dan area objek dalam pixel. Hasil analisis dapat diunduh sebagai CSV.</p>
        <div class="dash-setting"><div><small>CONFIDENCE</small><strong>{settings['conf_threshold']:.2f}</strong></div>
        <div><small>IOU THRESHOLD</small><strong>{settings['iou_threshold']:.2f}</strong></div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">PREVIEW DATASET</div>', unsafe_allow_html=True)
    render_dataset_gallery(dataset_files, limit=5)
    st.markdown('<div class="sb-section">RIWAYAT TERBARU</div>', unsafe_allow_html=True)
    if not history.empty:
        st.dataframe(history.tail(10), use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada riwayat deteksi. Mulai analisis untuk melihat hasil pengukuran di sini.")


def dataset_page():
    st.markdown(
        """
        <div class="sb-hero">
            <div class="sb-hero-title">Dataset Sapi</div>
            <div class="sb-hero-sub">Galeri citra yang digunakan untuk monitoring dan pengujian model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_files = list_dataset_files()

    if not dataset_files:
        st.warning(f"Dataset belum tersedia pada folder: {DATASET_PATH}")
        return

    c1, c2 = st.columns([1, 3])
    c1.metric("Total Citra", len(dataset_files))
    c2.info("Dataset ditampilkan dalam bentuk thumbnail untuk memudahkan pemeriksaan visual.")

    st.markdown('<div class="sb-section">Galeri Dataset</div>', unsafe_allow_html=True)
    st.caption(f"Sumber: {DATASET_PATH}")
    search = st.text_input("Cari gambar", placeholder="Cari nama file dataset...")
    filtered = [path for path in dataset_files if search.casefold() in os.path.basename(path).casefold()]
    if not filtered:
        st.info("Tidak ada gambar yang sesuai pencarian.")
        return
    st.caption(f"{len(filtered)} gambar ditemukan")
    page = st.selectbox("Halaman", range(1, (len(filtered) + 19) // 20 + 1))
    render_dataset_gallery(filtered[(page - 1) * 20:page * 20], limit=None)


def detection_page():
    st.markdown(
        """
        <div class="sb-hero">
            <div class="sb-hero-title">Analisis Sapi</div>
            <div class="sb-hero-sub">
                Upload gambar, hapus background, lalu deteksi Cow dan Belly secara otomatis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(MODEL_ERROR or "Model belum siap. Silakan cek file model.")
        return

    st.markdown('<div class="sb-section">1. Input Gambar</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Pilih gambar sapi",
        type=["jpg", "jpeg", "png"],
        key="detection_upload",
        help="Format yang didukung: JPG, JPEG, PNG",
    )

    col_setting_1, col_setting_2, col_action = st.columns([1, 1, 1.2])
    with col_setting_1:
        st.markdown('<div class="sb-input-title">Confidence</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="sb-pill">{st.session_state.settings["conf_threshold"]:.2f}</span>',
            unsafe_allow_html=True,
        )
    with col_setting_2:
        st.markdown('<div class="sb-input-title">IoU Threshold</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="sb-pill">{st.session_state.settings["iou_threshold"]:.2f}</span>',
            unsafe_allow_html=True,
        )
    with col_action:
        submitted = st.button("Mulai Analisis", type="primary", use_container_width=True)

    if submitted:
        if uploaded_file is None:
            st.warning("Silakan pilih gambar terlebih dahulu.")
        else:
            try:
                conf_threshold = st.session_state.settings["conf_threshold"]
                iou_threshold = st.session_state.settings["iou_threshold"]

                with st.status("SmartBreeding sedang menganalisis gambar...", expanded=True) as status:
                    st.write("🌿 Menghapus background...")
                    result = process_detection(uploaded_file, conf_threshold, iou_threshold)
                    st.write("🎯 Menjalankan deteksi Cow dan Belly...")
                    st.write("📏 Menghitung ukuran objek...")
                    status.update(label="Analisis selesai", state="complete", expanded=False)

                st.session_state.latest_result = result
                st.session_state.detection_history.append(
                    {
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_name": getattr(uploaded_file, "name", "camera_image.jpg"),
                        "rows": result["df"].to_dict("records"),
                        "summary": result["summary"],
                    }
                )
                sync_active_account_state()
                st.success("Analisis berhasil dilakukan.")
            except Exception as exc:
                st.error(f"Gagal memproses deteksi: {exc}")

    st.markdown('<div class="sb-section">2. Hasil Analisis</div>', unsafe_allow_html=True)
    result = st.session_state.latest_result

    if result is None:
        st.info("Belum ada hasil analisis. Pilih gambar lalu tekan **Mulai Analisis**.")
    else:
        df = result["df"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="sb-card-title">Gambar Asli</div>'
                '<div class="sb-muted">Citra sebelum background removal.</div>',
                unsafe_allow_html=True,
            )
            st.image(result["input_image"], use_container_width=True)

        with col2:
            st.markdown(
                '<div class="sb-card-title">Hasil Deteksi</div>'
                '<div class="sb-muted">Background telah diproses dan bounding box ditampilkan.</div>',
                unsafe_allow_html=True,
            )
            st.image(result["output_rgb"], use_container_width=True)

        cow_count = result["summary"]["cow_count"]
        belly_count = result["summary"]["belly_count"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Cow", cow_count)
        m2.metric("Belly", belly_count)
        m3.metric("Total Objek", cow_count + belly_count)

        if not df.empty:
            st.markdown('<div class="sb-section">Data Hasil Pengukuran</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil CSV",
                csv,
                file_name="hasil_deteksi.csv",
                mime="text/csv",
                use_container_width=True,
            )

            cow_data = df[df["Class"] == "Cow"]
            belly_data = df[df["Class"] == "Belly"]

            st.markdown('<div class="sb-section">Ringkasan Morfologi</div>', unsafe_allow_html=True)

            if not cow_data.empty:
                cow = cow_data.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cow Width", f"{cow['Width (px)']:.0f} px")
                c2.metric("Cow Height", f"{cow['Height (px)']:.0f} px")
                c3.metric("Cow Diagonal", f"{cow['Diagonal (px)']:.0f} px")
                c4.metric("Cow Area", f"{cow['Area (px²)']:,.0f} px²")

            if not belly_data.empty:
                belly = belly_data.iloc[0]
                b1, b2, b3 = st.columns(3)
                b1.metric("Belly Width", f"{belly['Width (px)']:.0f} px")
                b2.metric("Belly Height", f"{belly['Height (px)']:.0f} px")
                b3.metric("Belly Area", f"{belly['Area (px²)']:,.0f} px²")
        else:
            st.warning("Tidak ada objek terdeteksi pada gambar ini.")

    st.markdown('<div class="sb-section">3. Riwayat Analisis</div>', unsafe_allow_html=True)

    if not st.session_state.detection_history:
        st.info("Belum ada riwayat deteksi.")
    else:
        history_df = build_merged_history_df()
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        merged_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Semua Riwayat CSV",
            merged_csv,
            file_name="riwayat_semua_deteksi.csv",
            mime="text/csv",
            use_container_width=True,
        )


def profile_page():
    st.markdown("""<div class="sb-hero"><div class="sb-hero-title">Profil</div><div class="sb-hero-sub">Kelola informasi pengguna SmartBreeding.</div></div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 2.3])

    with col_left:
        uploaded_photo = st.file_uploader("Upload foto profil", type=["png", "jpg", "jpeg"])
        if uploaded_photo is not None:
            st.session_state.profile_image = uploaded_photo.getvalue()

        image_bytes = st.session_state.get("profile_image")
        if image_bytes:
            st.image(Image.open(__import__('io').BytesIO(image_bytes)), width=180)
        else:
            st.markdown(
                f"<div style='width:180px;height:180px;border-radius:50%;background:#1f77b4;color:white;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:54px;'>{st.session_state.user_name[:1].upper()}</div>",
                unsafe_allow_html=True,
            )

    with col_right:
        with st.form("profile_form"):
            name = st.text_input("Nama", value=st.session_state.user_name)
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=["Laki-laki", "Perempuan"].index(st.session_state.gender))
            address = st.text_area("Alamat", value=st.session_state.address)
            updated = st.form_submit_button("Simpan Profil")

        if updated:
            st.session_state.user_name = name
            st.session_state.user_role = "Akun"
            st.session_state.gender = gender
            st.session_state.address = address
            st.success("Profil berhasil diperbarui.")

    st.markdown("---")
    st.subheader("Statistik")
    c1, c2, c3 = st.columns(3)
    c1.metric("Deteksi", len(st.session_state.detection_history))
    c2.metric("Status", "Aktif")
    c3.metric("Akun", st.session_state.user_name)


def settings_page():
    st.markdown("""<div class="sb-hero"><div class="sb-hero-title">Pengaturan</div><div class="sb-hero-sub">Atur profil, akun, model, dan tema aplikasi.</div></div>""", unsafe_allow_html=True)

    profile_tab, account_tab, model_tab = st.tabs(["Pengaturan Profil", "Pengaturan Akun", "Pengaturan Model"])

    with profile_tab:
        with st.form("form_profile_settings"):
            st.write("### Profil")
            new_name = st.text_input("Nama", value=st.session_state.user_name)
            new_gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=["Laki-laki", "Perempuan"].index(st.session_state.gender))
            new_address = st.text_area("Alamat", value=st.session_state.address)
            submitted = st.form_submit_button("Simpan")

            if submitted:
                st.session_state.user_name = new_name
                st.session_state.user_role = "Akun"
                st.session_state.gender = new_gender
                st.session_state.address = new_address
                sync_active_account_state()
                st.success("Profil berhasil diperbarui.")
                st.rerun()

    with account_tab:
        with st.form("form_account_settings"):
            st.write("### Akun")
            current_username = st.session_state.get("active_account") or st.session_state.settings.get("account_username")
            st.text_input("Username", value=current_username, disabled=True, help="Username tidak dapat diubah.")
            new_password = st.text_input("Password Baru", type="password", value=st.session_state.settings["account_password"], help="Password yang digunakan untuk login berikutnya.")
            new_notifications = st.checkbox("Aktifkan notifikasi", value=st.session_state.settings["notifications"])
            submitted = st.form_submit_button("Simpan")

            if submitted:
                if not new_password.strip():
                    st.warning("Password baru tidak boleh kosong.")
                else:
                    if current_username:
                        st.session_state.accounts[current_username] = new_password
                    st.session_state.settings["account_username"] = current_username
                    st.session_state.settings["account_password"] = new_password
                    st.session_state.settings["notifications"] = new_notifications
                    st.success("Password berhasil diganti. Login berikutnya akan menggunakan password baru.")
                    st.rerun()

    with model_tab:
        with st.form("form_model_settings"):
            st.write("### Model")
            new_conf_threshold = st.slider("Confidence Threshold", 0.10, 1.00, st.session_state.settings["conf_threshold"], 0.05)
            new_iou_threshold = st.slider("IoU Threshold", 0.10, 1.00, st.session_state.settings["iou_threshold"], 0.05)
            new_theme = st.selectbox("Tema", ["Light", "Dark", "System Default"], index=["Light", "Dark", "System Default"].index(st.session_state.settings["theme"]))
            submitted = st.form_submit_button("Simpan")

            if submitted:
                st.session_state.settings["conf_threshold"] = new_conf_threshold
                st.session_state.settings["iou_threshold"] = new_iou_threshold
                st.session_state.settings["theme"] = new_theme
                apply_theme()
                st.success("Pengaturan model berhasil disimpan.")
                st.rerun()

    apply_theme()


def main_app():
    render_sidebar()

    with st.container(key=f"page_{st.session_state.page}"):
        if st.session_state.page == "dashboard":
            dashboard_page()
        elif st.session_state.page == "dataset":
            dataset_page()
        elif st.session_state.page == "deteksi":
            detection_page()
        elif st.session_state.page == "profil":
            profile_page()
        elif st.session_state.page == "pengaturan":
            settings_page()
        else:
            dashboard_page()


init_state()
apply_theme()

if not st.session_state.authenticated:
    login_page()
else:
    main_app()
