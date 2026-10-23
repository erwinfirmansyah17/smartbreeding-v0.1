import os
import glob
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import base64

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


def apply_theme():
    settings = st.session_state.get("settings", {})
    theme = settings.get("theme", "Light")

    if theme == "System Default":
        theme = "Light"

    if theme == "Dark":
        page_bg = "linear-gradient(180deg,#07111f 0%,#0b1728 55%,#0d1f25 100%)"
        card_bg = "rgba(15,23,42,.86)"
        card_border = "rgba(148,163,184,.20)"
        primary_text = "#f8fafc"
        secondary_text = "#cbd5e1"
        sidebar_bg = "linear-gradient(180deg,#071827 0%,#0d2f38 100%)"
        soft_bg = "rgba(255,255,255,.06)"
    else:
        page_bg = "linear-gradient(180deg,#f3f8ff 0%,#f6fffa 50%,#eef8f2 100%)"
        card_bg = "rgba(255,255,255,.94)"
        card_border = "#dfe9e5"
        primary_text = "#14213d"
        secondary_text = "#667085"
        sidebar_bg = "linear-gradient(180deg,#075f49 0%,#0b7b5a 48%,#0b4f68 100%)"
        soft_bg = "#f8fbfa"

    css = f"""
    <style>
    :root {{
        --sb-green:#0b7a59;
        --sb-green-2:#16a75c;
        --sb-blue:#1769e8;
        --sb-red:#e34242;
        --sb-text:{primary_text};
        --sb-muted:{secondary_text};
        --sb-card:{card_bg};
        --sb-border:{card_border};
    }}

    .stApp {{
        background:{page_bg};
        color:var(--sb-text);
    }}

    .block-container {{
        max-width: 1320px;
        padding-top: 1.15rem;
        padding-bottom: 3rem;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background:{sidebar_bg};
        border-right:1px solid rgba(255,255,255,.12);
    }}

    [data-testid="stSidebar"] * {{
        color:white;
    }}

    [data-testid="stSidebar"] .stButton button {{
        width:100%;
        min-height:46px;
        border:none;
        border-radius:13px;
        color:white;
        font-weight:700;
        background:rgba(255,255,255,.08);
        transition:.18s ease;
        text-align:left;
    }}

    [data-testid="stSidebar"] .stButton button:hover {{
        background:rgba(255,255,255,.17);
        transform:translateY(-1px);
    }}

    .sb-brand {{
        display:flex;
        align-items:center;
        gap:14px;
        padding:10px 8px 14px 0;
        width:100%;
    }}

    .sb-brand-mark {{
        display:flex;
        align-items:center;
        justify-content:center;
        min-width:84px;
        min-height:84px;
        flex-shrink:0;
    }}

    .sb-logo {{
        width:clamp(54px, 5vw, 78px);
        height:clamp(54px, 5vw, 78px);
        border-radius:50%;
        background:transparent;
        border:2px solid rgba(255,255,255,.78);
        display:flex;
        align-items:center;
        justify-content:center;
        overflow:hidden;
        box-shadow:0 6px 18px rgba(0,0,0,.14);
        padding:0;
        max-width:100%;
        max-height:100%;
    }}

    .sb-logo img {{
        display:block;
        width:100%;
        height:100%;
        object-fit:contain;
        border-radius:50%;
        background:transparent;
    }}

    .sb-brand-text {{
        display:flex;
        flex-direction:column;
        justify-content:center;
        line-height:1.08;
        margin-left:2px;
        flex:1 1 auto;
        min-width:0;
    }}

    .sb-brand-title {{
        font-size:clamp(16px, 2vw, 22px);
        font-weight:900;
        line-height:1.08;
        color:white;
        white-space:normal;
    }}

    .sb-brand-sub {{
        margin-top:4px;
        font-size:clamp(9px, 1.2vw, 12px);
        color:#d9fff0;
        font-weight:600;
        white-space:normal;
    }}

    @media (max-width: 768px) {{
        .sb-brand {{
            gap:10px;
            padding:8px 4px 12px 0;
        }}

        .sb-brand-mark {{
            min-width:58px;
            min-height:58px;
        }}

        .sb-logo {{
            width:clamp(46px, 9vw, 58px);
            height:clamp(46px, 9vw, 58px);
        }}
    }}

    @media (max-width: 520px) {{
        .sb-brand {{
            flex-wrap:nowrap;
            justify-content:flex-start;
        }}

        .sb-brand-text {{
            margin-left:0;
        }}
    }}

    .login-brand-wrap {{
        display:flex;
        justify-content:center;
        align-items:center;
        width:100%;
    }}

    .login-brand-wrap img {{
        display:block;
        width:clamp(120px, 18vw, 220px);
        height:auto;
        max-width:220px;
        min-width:120px;
        margin:0 auto;
    }}

    .login-title {{
        font-size:clamp(24px, 3vw, 32px);
        font-weight:900;
        color:#14213d;
        margin-top:4px;
        text-align:center;
    }}

    .sb-hero {{
        background:linear-gradient(135deg,#087a5a 0%,#0f9c68 58%,#1769e8 145%);
        border-radius:24px;
        padding:24px 26px;
        color:white;
        box-shadow:0 14px 32px rgba(8,122,90,.18);
        margin-bottom:18px;
        position:relative;
        overflow:hidden;
    }}

    .sb-hero:after {{
        content:"";
        width:210px;height:210px;border-radius:50%;
        background:rgba(255,255,255,.08);
        position:absolute;right:-70px;top:-95px;
    }}

    .sb-hero-title {{
        font-size:31px;
        font-weight:900;
        margin:0;
        line-height:1.05;
        position:relative;z-index:1;
    }}

    .sb-hero-sub {{
        margin-top:8px;
        color:#eafff5;
        font-size:14px;
        font-weight:500;
        position:relative;z-index:1;
    }}

    .sb-card {{
        background:var(--sb-card);
        border:1px solid var(--sb-border);
        border-radius:19px;
        padding:18px;
        box-shadow:0 9px 25px rgba(15,23,42,.055);
        margin-bottom:14px;
    }}

    .sb-card-title {{
        font-size:15px;
        font-weight:850;
        color:var(--sb-text);
        margin-bottom:5px;
    }}

    .sb-muted {{
        color:var(--sb-muted);
        font-size:13px;
    }}

    .sb-section {{
        font-size:20px;
        font-weight:900;
        color:var(--sb-text);
        margin:12px 0 10px 0;
    }}

    .sb-pill {{
        display:inline-flex;
        align-items:center;
        gap:6px;
        border-radius:999px;
        padding:6px 10px;
        font-size:11px;
        font-weight:800;
        background:#e8f8ef;
        color:#087a5a;
        border:1px solid #c8eed8;
    }}

    .sb-input-title {{
        font-size:14px;
        font-weight:850;
        color:var(--sb-text);
        margin-bottom:4px;
    }}

    .sb-metric {{
        background:var(--sb-card);
        border:1px solid var(--sb-border);
        border-radius:18px;
        padding:16px 17px;
        min-height:116px;
        box-shadow:0 7px 20px rgba(15,23,42,.045);
    }}

    .sb-metric-label {{
        font-size:11px;
        letter-spacing:.06em;
        text-transform:uppercase;
        font-weight:800;
        color:var(--sb-muted);
    }}

    .sb-metric-value {{
        margin-top:8px;
        font-size:29px;
        font-weight:900;
        color:var(--sb-text);
    }}

    .sb-metric-note {{
        margin-top:3px;
        font-size:11px;
        color:var(--sb-muted);
    }}

    .sb-result-good {{
        border:1px solid #a9e6c1;
        background:linear-gradient(135deg,#f7fff9,#e9f9ef);
        border-radius:19px;
        padding:18px;
    }}

    .stButton > button {{
        border-radius:13px;
        min-height:44px;
        font-weight:800;
    }}

    .stDownloadButton > button {{
        border-radius:13px;
        min-height:44px;
        font-weight:800;
        border:1px solid #1769e8;
    }}

    [data-testid="stFileUploader"] {{
        background:var(--sb-card);
        border:1px dashed #9fc7b5;
        border-radius:17px;
        padding:6px;
    }}

    [data-testid="stMetric"] {{
        background:var(--sb-card);
        border:1px solid var(--sb-border);
        border-radius:17px;
        padding:13px 15px;
        box-shadow:0 6px 18px rgba(15,23,42,.04);
    }}

    [data-testid="stDataFrame"] {{
        border-radius:16px;
        overflow:hidden;
        border:1px solid var(--sb-border);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap:8px;
        background:{soft_bg};
        padding:6px;
        border-radius:14px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius:10px;
        font-weight:800;
    }}

    hr {{
        border-color:rgba(148,163,184,.18)!important;
    }}

    #MainMenu {{visibility:hidden;}}
    footer {{visibility:hidden;}}
    header {{background:transparent!important;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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


def list_dataset_files():
    excluded_names = {"3_jpeg.rf.f0eb03fbdd9f1fad8f84e384168aade7.jpg"}
    dataset_candidates = [
        "/content/drive/MyDrive/RISET/Dataset Sapi/Data",
        r"D:\Kuliah\RISET\Data\images",
        os.path.join(os.path.dirname(__file__), "Dataset Sapi", "Data"),
        os.path.join(os.path.dirname(__file__), "data"),
    ]

    for folder in dataset_candidates:
        if os.path.exists(folder):
            files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for file_path in glob.glob(os.path.join(folder, ext)):
                    if os.path.basename(file_path) not in excluded_names:
                        files.append(file_path)
            return sorted(files)
    return []


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
    # LOGO SMARTBREEDING
    # =========================
    logo_path = os.path.join(os.path.dirname(__file__), "SmartBreeding_V.01-removebg-preview.png")

    # Logo di tengah
    col_left, col_logo, col_right = st.columns([2, 1, 2])

    with col_logo:
        if os.path.exists(logo_path):
            st.markdown(
                f"""
                <div class="login-brand-wrap">
                    <img src="data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}" />
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        """
        <div style="
            max-width:320px;
            margin:0 auto 0 auto;
            text-align:center;
        ">
            <div class="login-title">SmartBreeding V.01</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1.1, 1.5, 1.1])
    with center:
        # st.markdown('<div class="sb-card">', unsafe_allow_html=True)

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
            st.markdown("### Masuk ke Sistem")
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

        st.markdown("</div>", unsafe_allow_html=True)

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
            ("dashboard", "Homepage"),
            ("dataset", "Dataset"),
            ("deteksi", "Deteksi"),
            ("pengaturan", "Pengaturan"),
        ]

        for page_key, label in menu_items:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
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
    st.markdown(
        f"""
        <div class="sb-hero">
            <div class="sb-hero-title">SmartBreeding V.01</div>
            <div class="sb-hero-sub">
                Selamat datang, <b>{st.session_state.user_name}</b>. Sistem monitoring,
                deteksi, dan pengukuran sapi siap digunakan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_files = list_dataset_files()
    total_dataset = len(dataset_files)
    total_sapi = len(st.session_state.cattle_data)
    total_deteksi = len(st.session_state.detection_history)
    total_history_rows = len(build_merged_history_df())

    m1, m2, m3 = st.columns(3)
    metrics = [
        ("Dataset", total_dataset, "Citra tersedia"),
        ("Analisis", total_deteksi, "Sesi deteksi"),
        ("Riwayat", total_history_rows, "Objek terukur"),
    ]
    for col, (label, value, note) in zip([m1, m2, m3], metrics):
        with col:
            st.markdown(
                f"""
                <div class="sb-metric">
                    <div class="sb-metric-label">{label}</div>
                    <div class="sb-metric-value">{value}</div>
                    <div class="sb-metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sb-section">Akses Cepat</div>', unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        st.markdown(
            '<div class="sb-card"><div class="sb-card-title">🗂️ Dataset</div>'
            '<div class="sb-muted">Lihat seluruh citra sapi yang tersedia.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Buka Dataset", key="quick_dataset", use_container_width=True):
            st.session_state.page = "dataset"
            st.rerun()

    with nav2:
        st.markdown(
            '<div class="sb-card"><div class="sb-card-title">🔎 Deteksi</div>'
            '<div class="sb-muted">Jalankan background removal dan deteksi Cow/Belly.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Mulai Deteksi", key="quick_detection", use_container_width=True):
            st.session_state.page = "deteksi"
            st.rerun()

    with nav3:
        st.markdown(
            '<div class="sb-card"><div class="sb-card-title">⚙️ Pengaturan</div>'
            '<div class="sb-muted">Atur confidence, IoU, akun, dan tema.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Buka Pengaturan", key="quick_settings", use_container_width=True):
            st.session_state.page = "pengaturan"
            st.rerun()

    st.markdown('<div class="sb-section">Fitur Utama</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    cards = [
        ("🌿 Background Removal", "Mengisolasi objek sapi menggunakan U²-Net sebelum deteksi."),
        ("🎯 Cow & Belly Detection", "Model YOLO mengenali sapi dan area belly beserta confidence."),
        ("📏 Pengukuran Morfologi", "Menghitung width, height, diagonal, dan area dalam pixel."),
    ]
    for col, (title, desc) in zip([f1, f2, f3], cards):
        with col:
            st.markdown(
                f'<div class="sb-card"><div class="sb-card-title">{title}</div>'
                f'<div class="sb-muted">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sb-section">Preview Dataset</div>', unsafe_allow_html=True)
    render_dataset_gallery(dataset_files, limit=5)

    st.markdown('<div class="sb-section">Riwayat Terbaru</div>', unsafe_allow_html=True)
    if st.session_state.detection_history:
        history_df = build_merged_history_df()
        if not history_df.empty:
            st.dataframe(history_df.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada riwayat deteksi yang tercatat.")


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
        st.warning("Dataset belum tersedia pada folder yang dipilih.")
        return

    c1, c2 = st.columns([1, 3])
    c1.metric("Total Citra", len(dataset_files))
    c2.info("Dataset ditampilkan dalam bentuk thumbnail untuk memudahkan pemeriksaan visual.")

    st.markdown('<div class="sb-section">Galeri Dataset</div>', unsafe_allow_html=True)
    render_dataset_gallery(dataset_files, limit=None)


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
