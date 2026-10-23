import os
import glob
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

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


st.set_page_config(page_title="Sistem Monitoring Sapi", layout="wide")


def apply_theme():
    settings = st.session_state.get("settings", {})
    theme = settings.get("theme", "Light")

    if theme == "System Default":
        theme = "Light"

    if theme == "Dark":
        css = """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p, .stApp div, .stApp label {
            color: #e5e7eb;
        }
        div[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
            color: white;
        }
        div[data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.08);
            color: white;
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
            font-weight: 600;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.12);
        }
        .stDataFrame, .stTable, .stMarkdownContainer, .stTabs [data-baseweb="tab-list"] {
            background: rgba(17, 24, 39, 0.7);
            color: #e5e7eb;
        }
        .main-card, .metric-box {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: none;
        }
        .hero-title { color: #f8fafc; }
        .muted { color: #cbd5e1; }
        </style>
        """
    else:
        css = """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f3f7ff 0%, #eefcf6 100%);
        }
        div[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #102a43 0%, #123d5b 100%);
            color: white;
        }
        div[data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.08);
            color: white;
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
            font-weight: 600;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.12);
        }
        .main-card {
            background: rgba(255,255,255,0.75);
            border: 1px solid #e5edf5;
            border-radius: 20px;
            padding: 1.2rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
        }
        .metric-box {
            background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%);
            border: 1px solid #d9f5e7;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.08);
        }
        .tiny-label {
            font-size: 12px;
            color: #64748b;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.4rem;
        }
        .muted {
            color: #475569;
        }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)


def init_state():
    defaults = {
        "authenticated": False,
        "page": "dashboard",
        "user_name": "Admin Sapi",
        "user_role": "Admin",
        "gender": "Perempuan",
        "address": "Lab Monitoring Sapi",
        "profile_image": None,
        "show_register": False,
        "accounts": {
            "admin": "admin123",
        },
        "settings": {
            "conf_threshold": 0.25,
            "iou_threshold": 0.50,
            "theme": "Light",
            "notifications": True,
            "account_username": "admin",
            "account_password": "admin123",
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
    st.title("Sistem Monitoring Sapi")

    if st.session_state.get("show_register", False):
        st.subheader("Halaman Daftar Akun")

        with st.form("register_form"):
            full_name = st.text_input("Nama Lengkap")
            username = st.text_input("Username Baru")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Konfirmasi Password", type="password")
            submitted = st.form_submit_button("Buat Akun")

        if submitted:
            if not full_name or not username or not password:
                st.error("Semua field harus diisi.")
            elif username in st.session_state.accounts:
                st.error("Username sudah terdaftar, silakan gunakan username lain.")
            elif password != confirm_password:
                st.error("Password dan konfirmasi password tidak sama.")
            else:
                st.session_state.accounts[username] = password
                st.session_state.settings["account_username"] = username
                st.session_state.settings["account_password"] = password
                st.session_state.user_name = full_name or username
                st.session_state.user_role = "Admin"
                st.session_state.authenticated = True
                st.session_state.page = "dashboard"
                st.session_state.show_register = False
                st.success("Akun berhasil dibuat dan login berhasil.")
                st.rerun()

        if st.button("Sudah punya akun? Masuk"):
            st.session_state.show_register = False
            st.rerun()
    else:
        st.subheader("Halaman Login")

        with st.form("login_form"):
            username = st.text_input("Username", value=st.session_state.settings["account_username"])
            password = st.text_input("Password", type="password", value=st.session_state.settings["account_password"])
            submitted = st.form_submit_button("Masuk")

        if submitted:
            stored_password = st.session_state.accounts.get(username)
            if stored_password == password:
                st.session_state.authenticated = True
                st.session_state.page = "dashboard"
                st.session_state.settings["account_username"] = username
                st.session_state.settings["account_password"] = password
                st.session_state.user_name = st.session_state.user_name if st.session_state.user_name != "Admin Sapi" else username
                st.success("Login berhasil.")
                st.rerun()
            else:
                st.error("Username atau password salah.")

        st.info("Gunakan akun demo: admin / admin123")

        if st.button("Belum punya akun? Daftar sekarang"):
            st.session_state.show_register = True
            st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='padding:12px 14px;border:1px solid rgba(255,255,255,0.12);border-radius:16px;background:rgba(255,255,255,0.06);display:flex;align-items:center;justify-content:space-between;'>
                <div style='font-size:26px;'>☰</div>
                <div style='font-size:12px;color:#dfeaf9;font-weight:700;letter-spacing:0.08em;'>MENU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        col_avatar, col_info = st.columns([1, 2])
        with col_avatar:
            image_bytes = st.session_state.get("profile_image")
            if image_bytes:
                st.image(Image.open(__import__('io').BytesIO(image_bytes)), width=42)
            else:
                st.markdown(
                    f"<div style='width:42px;height:42px;border-radius:50%;background:#1f77b4;color:white;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:16px;'>{st.session_state.user_name[:1].upper()}</div>",
                    unsafe_allow_html=True,
                )
        with col_info:
            st.markdown(f"**{st.session_state.user_name}**")
            st.caption(f"Sebagai {st.session_state.user_role}")

        st.markdown("---")

        menu_items = [
            ("dashboard", "🏠 Homepage"),
            ("dataset", "📁 Dataset"),
            ("deteksi", "🔍 Deteksi"),
            ("pengaturan", "⚙️ Pengaturan"),
        ]

        for page_key, label in menu_items:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.rerun()


def dashboard_page():
    st.markdown("<div class='hero-title'>🏠 Homepage</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 1])
    with left_col:
        st.markdown(f"## Selamat datang, {st.session_state.user_name}!")
        st.caption("Sistem monitoring sapi Anda sedang aktif dan siap digunakan.")
        st.write("Berikut ringkasan cepat fitur utama aplikasi untuk mempermudah proses monitoring dan deteksi sapi.")

    with right_col:
        st.markdown("### Status login")
        st.success("Sudah login")
        st.write(f"Nama: {st.session_state.user_name}")
        st.write(f"Role: {st.session_state.user_role}")

    dataset_files = list_dataset_files()
    total_dataset = len(dataset_files)
    total_sapi = len(st.session_state.cattle_data)
    total_deteksi = len(st.session_state.detection_history)
    total_history_rows = len(build_merged_history_df())

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", total_dataset)
    c2.metric("Deteksi", total_deteksi)
    c3.metric("Riwayat", total_history_rows)

    st.markdown("### Navigasi cepat")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("📁 Buka Dataset", use_container_width=True):
            st.session_state.page = "dataset"
            st.rerun()
    with nav2:
        if st.button("🔍 Buka Deteksi", use_container_width=True):
            st.session_state.page = "deteksi"
            st.rerun()
    with nav3:
        if st.button("⚙️ Pengaturan", use_container_width=True):
            st.session_state.page = "pengaturan"
            st.rerun()

    st.markdown("### Fitur utama aplikasi")
    st.markdown(
        """
        - Monitoring data sapi dan kondisi kesehatan.
        - Input data sapi baru secara cepat.
        - Deteksi dan prediksi objek sapi serta bagian tubuh menggunakan model YOLO.
        - Visualisasi hasil deteksi dan pengukuran morfologi.
        - Riwayat hasil deteksi yang bisa diunduh dan digabung dalam satu CSV.
        """
    )

    st.markdown("### 5 contoh dataset")
    render_dataset_gallery(dataset_files, limit=5)

    st.markdown("### Statistik riwayat deteksi")
    if st.session_state.detection_history:
        history_df = build_merged_history_df()
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada riwayat deteksi yang tercatat.")



def dataset_page():
    st.markdown("<div class='hero-title'>📁 Dataset</div>", unsafe_allow_html=True)
    dataset_files = list_dataset_files()

    if not dataset_files:
        st.warning("Dataset belum tersedia pada folder yang dipilih.")
        return

    st.markdown(f"### Preview dataset ({len(dataset_files)} data)")
    render_dataset_gallery(dataset_files, limit=None)


def detection_page():
    st.title("🔍 Halaman Deteksi")

    if model is None:
        st.warning(MODEL_ERROR or "Model belum siap. Silakan cek file model.")
        return

    st.subheader("1. Input Data")
    with st.form("input_data_form"):
        uploaded_file = st.file_uploader("Upload gambar sapi", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Mulai Deteksi")

    if submitted and uploaded_file is not None:
        try:
            conf_threshold = st.session_state.settings["conf_threshold"]
            iou_threshold = st.session_state.settings["iou_threshold"]
            result = process_detection(uploaded_file, conf_threshold, iou_threshold)
            st.session_state.latest_result = result
            st.session_state.detection_history.append(
                {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": uploaded_file.name,
                    "rows": result["df"].to_dict("records"),
                    "summary": result["summary"],
                }
            )
            st.success("Deteksi berhasil dilakukan.")
        except Exception as exc:
            st.error(f"Gagal memproses deteksi: {exc}")

    st.markdown("---")
    st.subheader("2. Hasil Deteksi")
    result = st.session_state.latest_result
    if result is None:
        st.info("Belum ada hasil deteksi. Silakan upload gambar terlebih dahulu.")
    else:
        df = result["df"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Gambar input")
            st.image(result["input_image"], use_container_width=True)
        with col2:
            st.markdown("### Hasil deteksi")
            st.image(result["output_rgb"], use_container_width=True)

        if not df.empty:
            st.markdown("### Data hasil pengukuran")
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download hasil CSV", csv, file_name="hasil_deteksi.csv", mime="text/csv")

            cow_data = df[df["Class"] == "Cow"]
            belly_data = df[df["Class"] == "Belly"]

            st.markdown("### Ringkasan")
            c1, c2, c3, c4 = st.columns(4)
            if not cow_data.empty:
                cow = cow_data.iloc[0]
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

    st.markdown("---")
    st.subheader("3. Riwayat")
    if not st.session_state.detection_history:
        st.info("Belum ada riwayat deteksi.")
    else:
        history_df = build_merged_history_df()
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        merged_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download semua riwayat CSV", merged_csv, file_name="riwayat_semua_deteksi.csv", mime="text/csv")


def profile_page():
    st.title("👤 Halaman Profil")

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
            role = st.selectbox("Sebagai", ["Admin", "User"], index=["Admin", "User"].index(st.session_state.user_role))
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=["Laki-laki", "Perempuan"].index(st.session_state.gender))
            address = st.text_area("Alamat", value=st.session_state.address)
            updated = st.form_submit_button("Simpan Profil")

        if updated:
            st.session_state.user_name = name
            st.session_state.user_role = role
            st.session_state.gender = gender
            st.session_state.address = address
            st.success("Profil berhasil diperbarui.")

    st.markdown("---")
    st.subheader("Statistik")
    c1, c2, c3 = st.columns(3)
    c1.metric("Deteksi", len(st.session_state.detection_history))
    c2.metric("Status", "Aktif")
    c3.metric("Role", st.session_state.user_role)


def settings_page():
    st.title("⚙️ Halaman Pengaturan")

    profile_tab, account_tab, model_tab = st.tabs(["Pengaturan Profil", "Pengaturan Akun", "Pengaturan Model"])

    with profile_tab:
        with st.form("form_profile_settings"):
            st.write("### Profil")
            new_name = st.text_input("Nama", value=st.session_state.user_name)
            new_role = st.selectbox("Sebagai", ["Admin", "User"], index=["Admin", "User"].index(st.session_state.user_role))
            new_gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=["Laki-laki", "Perempuan"].index(st.session_state.gender))
            new_address = st.text_area("Alamat", value=st.session_state.address)
            submitted = st.form_submit_button("Simpan")

        if submitted:
            st.session_state.user_name = new_name
            st.session_state.user_role = new_role
            st.session_state.gender = new_gender
            st.session_state.address = new_address
            st.success("Pengaturan profil tersimpan.")

    with account_tab:
        with st.form("form_account_settings"):
            st.write("### Akun")
            new_username = st.text_input("Username", value=st.session_state.settings["account_username"])
            new_password = st.text_input("Password", type="password", value=st.session_state.settings["account_password"])
            new_notifications = st.checkbox("Aktifkan notifikasi", value=st.session_state.settings["notifications"])
            submitted = st.form_submit_button("Simpan")

        if submitted:
            st.session_state.settings["account_username"] = new_username
            st.session_state.settings["account_password"] = new_password
            st.session_state.settings["notifications"] = new_notifications
            st.success("Pengaturan akun tersimpan.")

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
            st.success("Pengaturan model tersimpan.")

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
