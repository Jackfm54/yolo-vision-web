import argparse
from pathlib import Path
from ultralytics import YOLO

# Mapas de modelos por tarea (YOLO11; si no, YOLOv8 también funciona)
DEFAULT_MODELS = {
    "detect":  ["yolo11n.pt",      "yolov8n.pt"],
    "segment": ["yolo11n-seg.pt",  "yolov8n-seg.pt"],
    "classify":["yolo11n-cls.pt",  "yolov8n-cls.pt"],
    "pose":    ["yolo11n-pose.pt", "yolov8n-pose.pt"],
    "obb":     ["yolo11n-obb.pt",  "yolov8n-obb.pt"],   # Oriented Bounding Boxes
    # Para track usaremos un modelo de detección
    "track":   ["yolo11n.pt",      "yolov8n.pt"],
}

def choose_weights(task: str, custom: str | None):
    """Devuelve la ruta del modelo a usar."""
    if custom:
        return custom
    candidates = DEFAULT_MODELS.get(task, [])
    if not candidates:
        raise ValueError(f"Tarea no soportada: {task}")
    return candidates[0]  # el primero disponible; Ultralytics lo descargará si no existe

def run(task: str, source: str, weights: str | None, conf: float, device: str, tracker_cfg: str):
    w = choose_weights(task, weights)
    print(f"[INFO] Tarea: {task} | Modelo: {w} | Fuente: {source} | conf={conf} | device={device}")

    model = YOLO(w)  # carga y descarga el peso si hace falta

    common_kwargs = dict(source=source, conf=conf, device=device, save=True, show=False, stream=False, verbose=True)

    if task == "track":
        # Seguimiento multi-objeto (ByteTrack por defecto)
        # tracker_cfg puede ser "bytetrack.yaml" u "botsort.yaml"
        results = model.track(**common_kwargs, tracker=tracker_cfg, persist=True)
    elif task in {"detect", "segment", "pose", "classify", "obb"}:
        # Predicción estándar (detección, segmentación, pose, clasificación, obb)
        results = model.predict(**common_kwargs)
    else:
        raise ValueError(f"Tarea no soportada: {task}")

    # Carpeta de salida (Ultralytics crea runs/<task>/predictX)
    if results and hasattr(results[0], "save_dir"):
        out_dir = Path(results[0].save_dir)
        print(f"[OK] Resultados guardados en: {out_dir.resolve()}")
    else:
        print("[OK] Proceso finalizado. Revisa la carpeta 'runs'.")

def main():
    parser = argparse.ArgumentParser(description="App unificada de Visión (YOLO11/YOLOv8) - Detect/Segment/Classify/Pose/OBB/Track")
    parser.add_argument("--task", choices=["detect", "segment", "classify", "pose", "obb", "track"], required=True,
                        help="Tipo de tarea de la imagen")
    parser.add_argument("--source", required=True,
                        help="Ruta a imagen/video/carpeta o webcam (0). Ej: bus.jpg | video.mp4 | 0")
    parser.add_argument("--weights", default=None,
                        help="Ruta a pesos personalizados (.pt). Si no se pasa, usa los por defecto.")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    parser.add_argument("--device", default="cpu", help="Dispositivo: cpu, 0, 0,1, etc.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Config de tracker (para --task track)")
    args = parser.parse_args()

    run(task=args.task, source=args.source, weights=args.weights, conf=args.conf, device=args.device, tracker_cfg=args.tracker)

if __name__ == "__main__":
    main()
