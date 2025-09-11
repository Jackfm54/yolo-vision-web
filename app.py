# app.py
import os
from pathlib import Path
import cv2
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration

# -----------------------------
# Configuración de página
# -----------------------------
st.set_page_config(page_title="YOLO Vision Web", page_icon="🧠", layout="wide")
st.title("🧠 YOLO Vision Web – Detect / Segment / Classify / Pose / OBB / Track")

# -----------------------------
# Modelos por tarea (YOLO11 / YOLOv8)
# -----------------------------
DEFAULT_MODELS = {
    "detect":  ["yolo11n.pt",      "yolov8n.pt"],
    "segment": ["yolo11n-seg.pt",  "yolov8n-seg.pt"],
    "classify":["yolo11n-cls.pt",  "yolov8n-cls.pt"],
    "pose":    ["yolo11n-pose.pt", "yolov8n-pose.pt"],
    "obb":     ["yolo11n-obb.pt",  "yolov8n-obb.pt"],
    # Para track usamos un modelo de detección; en vivo no haremos track persistente
    "track":   ["yolo11n.pt",      "yolov8n.pt"],
}

# -----------------------------
# Utilidades
# -----------------------------
def choose_default_weights(task: str) -> str:
    candidates = DEFAULT_MODELS.get(task, [])
    if not candidates:
        st.error(f"Tarea no soportada: {task}")
        st.stop()
    return candidates[0]

def save_upload(upload) -> Path:
    """Guarda un UploadedFile (o cámara) en /uploads y devuelve la ruta."""
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    # nombre seguro
    name = getattr(upload, "name", None) or "camera_photo.jpg"
    out = uploads / name
    # write bytes
    with open(out, "wb") as f:
        f.write(upload.getbuffer())
    return out

@st.cache_resource(show_spinner=False)
def load_model(weights: str) -> YOLO:
    """Cachea el modelo por pesos para acelerar."""
    return YOLO(weights)

def run_predict(task: str, source_path: Path | str, weights: str, conf: float, device: str, tracker: str):
    """Ejecuta inferencia y devuelve results."""
    model = load_model(weights)
    common = dict(source=str(source_path), conf=conf, device=device, save=True, show=False, stream=False, verbose=False)
    if task == "track":
        # En archivos (video), sí permitimos track
        return model.track(**common, tracker=tracker, persist=True)
    else:
        # detect / segment / pose / classify / obb
        return model.predict(**common)

def show_image_result(results, source_path: Path):
    """Muestra imagen anotada en la app."""
    frame = results[0].plot()  # BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, caption=f"Resultado: {Path(source_path).name}", use_container_width=True)

def guess_saved_output(results, source_path: Path) -> Path | None:
    """Intenta adivinar la ruta del archivo generado por Ultralytics en runs/..."""
    try:
        out_dir = Path(results[0].save_dir)
    except Exception:
        return None
    # Primero, el mismo nombre
    candidate = out_dir / source_path.name
    if candidate.exists():
        return candidate
    # Si no coincide, busca el primero de vídeo o imagen en el directorio
    vids = list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.avi")) + list(out_dir.glob("*.mkv")) + list(out_dir.glob("*.webm"))
    if vids:
        return vids[0]
    imgs = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")) + list(out_dir.glob("*.webp"))
    if imgs:
        return imgs[0]
    return None

# -----------------------------
# Sidebar: controles
# -----------------------------
with st.sidebar:
    st.subheader("🎛️ Parámetros")
    task = st.selectbox("Tarea", ["detect","segment","classify","pose","obb","track"])
    default_w = choose_default_weights(task)
    weights = st.text_input("Pesos (.pt)", value=default_w, help="Deja el valor por defecto o pega la ruta/URL de tu .pt", key="weights")
    conf = st.slider("Confianza", 0.05, 0.90, 0.25, 0.05, help="Umbral mínimo para mostrar predicciones")
    device = st.selectbox("Dispositivo", ["cpu", "0"], index=0, help="Usa 0 si tienes GPU CUDA")
    tracker = st.selectbox("Tracker (solo 'track')", ["bytetrack.yaml", "botsort.yaml"], index=0)
    st.markdown("---")
    source_mode = st.radio("Fuente", ["Subir archivo", "Cámara (foto)", "Webcam en vivo"], index=0)

# -----------------------------
# Entradas según modo de fuente
# -----------------------------
col1, col2 = st.columns(2)

image_files = []
video_file = None
cam_photo = None

if source_mode == "Subir archivo":
    with col1:
        image_files = st.file_uploader("Imagen(es)", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True)
    with col2:
        video_file = st.file_uploader("Video", type=["mp4","mov","avi","mkv","webm"])
elif source_mode == "Cámara (foto)":
    cam_photo = st.camera_input("Toma una foto con tu cámara")
else:
    # Webcam en vivo se maneja más abajo con WebRTC
    pass

# -----------------------------
# Botón de procesar (para archivos / foto)
# -----------------------------
if source_mode != "Webcam en vivo":
    run_btn = st.button("▶ Procesar")
else:
    run_btn = False

# -----------------------------
# Lógica de procesamiento (archivos / foto)
# -----------------------------
if run_btn:
    # 1) Clasificación con varias imágenes
    if task == "classify" and image_files:
        st.subheader("Resultados de Clasificación")
        for up in image_files:
            p = save_upload(up)
            with st.spinner(f"Clasificando {up.name}..."):
                results = run_predict(task, p, weights, conf, device, tracker)
            probs = results[0].probs
            if probs is not None:
                top1_id = int(probs.top1)
                top1_name = results[0].names.get(top1_id, str(top1_id))
                top1_score = float(probs.top1conf)
                st.markdown(f"**{up.name} → {top1_name} ({top1_score:.2%})**")
            show_image_result(results, p)

    # 2) Imágenes para detect/segment/pose/obb
    elif image_files and task in {"detect","segment","pose","obb"}:
        st.subheader(f"Resultados de {task.capitalize()}")
        for up in image_files:
            p = save_upload(up)
            with st.spinner(f"Procesando {up.name}..."):
                results = run_predict(task, p, weights, conf, device, tracker)
            show_image_result(results, p)

    # 3) Foto desde cámara
    elif cam_photo is not None:
        p = save_upload(cam_photo)
        st.subheader(f"Resultado de {task.capitalize()} (Foto de cámara)")
        with st.spinner("Procesando foto..."):
            # Si el usuario eligió 'track', no tiene sentido en una sola imagen → lo forzamos a detect
            _task = "detect" if task == "track" else task
            results = run_predict(_task, p, weights, conf, device, tracker)
        show_image_result(results, p)

    # 4) Video subido (detect/segment/pose/obb/track)
    elif video_file is not None:
        p = save_upload(video_file)
        st.subheader(f"Resultado en video – {task.upper()}")
        with st.spinner(f"Procesando {video_file.name}..."):
            results = run_predict(task, p, weights, conf, device, tracker)
        out_path = guess_saved_output(results, p)
        if out_path and out_path.exists():
            st.success(f"Archivo generado: {out_path}")
            st.video(str(out_path))
        else:
            st.warning("No se encontró el video anotado. Revisa la carpeta 'runs/'.")
    else:
        st.info("Sube al menos una **imagen** o un **video**, o usa la **cámara**.")

# -----------------------------
# Webcam en vivo (WebRTC)
# -----------------------------
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class YOLOProcessor(VideoTransformerBase):
    """
    Procesa frames en vivo desde la webcam.
    Nota: 'track' en vivo no se soporta aquí (Ultralytics.track espera stream con estado persistente).
    """
    def __init__(self):
        self.task = "detect"
        self.conf = 0.25
        self.device = "cpu"
        self.weights = choose_default_weights("detect")
        self.model = YOLO(self.weights)

    def set_params(self, task: str, weights: str, conf: float, device: str):
        # Si cambian los pesos, recarga el modelo
        if weights != self.weights:
            self.model = YOLO(weights)
            self.weights = weights
        self.task = "detect" if task == "track" else task  # forzar detect si eligen track
        self.conf = conf
        self.device = device

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Inferencia por frame (predict soporta detect/segment/pose/obb/classify)
        # En classify, la anotación será el label sobre la imagen
        results = self.model.predict(
            source=img, conf=self.conf, device=self.device, verbose=False
        )
        im = results[0].plot()  # BGR
        return av.VideoFrame.from_ndarray(im, format="bgr24")

if source_mode == "Webcam en vivo":
    st.subheader("🎥 Webcam en vivo")
    st.caption("Acepta los permisos del navegador. En vivo soporta detect/segment/pose/obb/clasify. 'track' se fuerza a 'detect'.")
    ctx = webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=YOLOProcessor,
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.set_params(task, weights, conf, device)
    # No continuar con el resto del script en este modo
    st.stop()
