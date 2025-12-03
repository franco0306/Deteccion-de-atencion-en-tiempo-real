# app_gui_multi.py - Sistema Multi-Persona para Detección de Atención en Zoom/Clases
import cv2
import numpy as np
import tensorflow as tf
import time
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime, timedelta
from PIL import Image, ImageTk
import threading
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from collections import defaultdict

# ========= Configuración ==========
MODEL_PATH = Path("modelos/atencion_mnv2_final_mejorado.keras")
CONFIG_PATH = Path("modelos/model_config.json")
IMG_SIZE = 224
UMBRAL = 0.65

CAM_INDEX = 0
CAM_BACKEND = 700
FRAME_W, FRAME_H = 640, 360
FOURCC = 'MJPG'

YUNET_ONNX = Path("modelos/face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DETECT_W, DETECT_H = 480, 270  # Mayor resolución para mejor detección
SCORE_TH = 0.45  # Umbral balanceado para detectar más rostros
NMS_TH = 0.3
TOP_K = 50  # Hasta 50 personas

YOLO_MODEL = Path("modelos/yolov8n.pt")
DISTRACTOR_CLASSES = [67, 63, 73, 64, 75, 66]
OBJECT_EVERY = 10  # Detectar objetos cada 10 frames (más rápido)

HEAD_POSE_THRESHOLD = 35
HEAD_DOWN_THRESHOLD = 40

DETECT_EVERY = 4  # Detectar caras cada 4 frames (balance velocidad/precisión)
CLASSIFY_EVERY = 4  # Clasificar cada 4 frames
SMOOTH_ALPHA_PROB = 0.6
ANGLE_ALPHA = 0.25
POSE_CONFIRMATION_FRAMES = 3
OBJECT_MEMORY_DURATION = 15

# Multi-persona
MIN_FACE_SIZE = 25  # Mínimo tamaño de rostro en píxeles (detecta rostros más pequeños)
MAX_FACES = 40  # Hasta 40 personas
PERSON_TIMEOUT = 30  # Frames sin ver persona antes de eliminarla
IOU_THRESHOLD = 0.5  # Para matching de personas entre frames

PERSON_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128),
    (128, 0, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255), (192, 192, 0)
]

# ========= Funciones auxiliares ==========
def ensure_yunet():
    if not YUNET_ONNX.exists():
        YUNET_ONNX.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(YUNET_URL, YUNET_ONNX.as_posix())

def ensure_yolo():
    if not YOLO_MODEL.exists():
        try:
            import subprocess
            subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            YOLO_MODEL.parent.mkdir(parents=True, exist_ok=True)
            return model
        except:
            return None
    else:
        from ultralytics import YOLO
        return YOLO(YOLO_MODEL)

def calculate_iou(box1, box2):
    """Calcula IoU entre dos bboxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def estimate_head_pose(landmarks):
    if landmarks is None or len(landmarks) < 5:
        return 0, 0
    
    right_eye = landmarks[0]
    left_eye = landmarks[1]
    nose = landmarks[2]
    right_mouth = landmarks[3]
    left_mouth = landmarks[4]
    
    eye_distance = np.linalg.norm(right_eye - left_eye)
    if eye_distance < 10:
        return 0, 0
    
    eye_center = (right_eye + left_eye) / 2
    mouth_center = (right_mouth + left_mouth) / 2
    
    nose_offset_x = nose[0] - eye_center[0]
    yaw = (nose_offset_x / eye_distance) * 50
    
    eye_to_nose = nose[1] - eye_center[1]
    nose_to_mouth = mouth_center[1] - nose[1]
    
    if nose_to_mouth > 5 and abs(eye_to_nose) > 0:
        ratio = eye_to_nose / nose_to_mouth
        if ratio > 1.3:
            pitch = (ratio - 1.3) * 50
        elif ratio < 0.6:
            pitch = (ratio - 0.6) * 35
        else:
            pitch = 0
    else:
        pitch = 0
    
    yaw = np.clip(yaw, -70, 70)
    pitch = np.clip(pitch, -60, 60)
    
    return yaw, pitch

def enhance_frame(frame):
    """Mejora la calidad del frame para mejor detección"""
    # Convertir a LAB para mejorar contraste
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Recombinar canales
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Reducir ruido ligeramente
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    
    # Sharpening suave para mejorar bordes
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def classify_face(frame_bgr, bbox_xywh, model):
    x, y, w, h = bbox_xywh
    if w < 10 or h < 10:
        return 0.5
    
    face_roi = frame_bgr[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(face_roi, axis=0).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    prob = float(model.predict(arr, verbose=0)[0][0])
    return prob

def detect_distractors(frame, yolo_model, face_bbox):
    if yolo_model is None:
        return False, []
    
    try:
        results = yolo_model(frame, verbose=False, conf=0.30, iou=0.45, imgsz=480, max_det=12)  # Balance velocidad/precisión
        
        if len(results) == 0 or results[0].boxes is None:
            return False, []
        
        boxes = results[0].boxes
        detected_objects = []
        
        fx, fy, fw, fh = face_bbox
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2
        search_radius = fw * 4.5
        
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls not in DISTRACTOR_CLASSES:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2
            
            dist = np.sqrt((obj_center_x - face_center_x)**2 + (obj_center_y - face_center_y)**2)
            
            if dist < search_radius:
                obj_names = {67: "celular", 63: "laptop", 73: "libro", 64: "mouse", 75: "control", 66: "teclado"}
                obj_name = obj_names.get(cls, f"objeto-{cls}")
                detected_objects.append({
                    'name': obj_name,
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    'conf': float(box.conf[0]),
                    'class': cls
                })
        
        return len(detected_objects) > 0, detected_objects
        
    except Exception as e:
        return False, []

# ========= Clase Principal Multi-Persona ==========
class AttentionMonitorMultiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Multi-Persona de Monitoreo de Atención")
        self.root.geometry("1500x900")
        self.root.configure(bg='#1e1e1e')
        
        # Variables de estado
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.detector = None
        self.model = None
        self.yolo_model = None
        self.video_mode = "camera"
        self.video_path = None
        self.total_video_frames = 0
        self.current_video_frame = 0
        
        # Estadísticas multi-persona
        self.session_stats = {
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'people': {},  # {person_id: stats}
            'global_stats': {
                'total_people_detected': 0,
                'max_simultaneous': 0,
                'frames_with_faces': 0
            }
        }
        
        # Tracking de personas
        self.person_trackers = {}  # {person_id: tracker_data}
        self.next_person_id = 1
        self.frame_id = 0
        
        # Crear UI
        self.create_ui()
        
        # Cargar modelos
        self.load_models()
    
    def create_ui(self):
        """Crea la interfaz gráfica"""
        
        # Panel Superior: Controles
        control_frame = tk.Frame(self.root, bg='#2d2d2d', height=100)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        title_label = tk.Label(control_frame, text="🎯 Monitor de Atención Multi-Persona (Zoom/Clases)", 
                               font=('Segoe UI', 18, 'bold'), 
                               bg='#2d2d2d', fg='#ffffff')
        title_label.pack(pady=10)
        
        btn_frame = tk.Frame(control_frame, bg='#2d2d2d')
        btn_frame.pack(pady=5)
        
        self.btn_start = tk.Button(btn_frame, text="▶ Iniciar Cámara", 
                                   command=self.start_session,
                                   font=('Segoe UI', 11, 'bold'),
                                   bg='#4CAF50', fg='white',
                                   width=14, height=2,
                                   relief=tk.RAISED, bd=3,
                                   cursor='hand2')
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_pause = tk.Button(btn_frame, text="⏸ Pausar", 
                                   command=self.pause_session,
                                   font=('Segoe UI', 11, 'bold'),
                                   bg='#FF9800', fg='white',
                                   width=14, height=2,
                                   relief=tk.RAISED, bd=3,
                                   cursor='hand2', state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(btn_frame, text="⏹ Finalizar y Reporte", 
                                 command=self.stop_session,
                                 font=('Segoe UI', 11, 'bold'),
                                 bg='#f44336', fg='white',
                                 width=18, height=2,
                                 relief=tk.RAISED, bd=3,
                                 cursor='hand2', state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.btn_video = tk.Button(btn_frame, text="📁 Procesar Video Zoom", 
                                   command=self.process_video_file,
                                   font=('Segoe UI', 11, 'bold'),
                                   bg='#9C27B0', fg='white',
                                   width=18, height=2,
                                   relief=tk.RAISED, bd=3,
                                   cursor='hand2')
        self.btn_video.pack(side=tk.LEFT, padx=5)
        
        # Panel Central
        center_frame = tk.Frame(self.root, bg='#1e1e1e')
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Video
        video_frame = tk.LabelFrame(center_frame, text="📹 Video en Vivo (Multi-Persona)", 
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='#2d2d2d', fg='#ffffff',
                                    bd=2, relief=tk.GROOVE)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Panel de estadísticas
        stats_frame = tk.LabelFrame(center_frame, text="📊 Estadísticas Grupales", 
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='#2d2d2d', fg='#ffffff',
                                    bd=2, relief=tk.GROOVE, width=450)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        stats_frame.pack_propagate(False)
        
        # Tiempo de sesión
        time_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        time_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(time_frame, text="⏱ Duración:", font=('Segoe UI', 11, 'bold'),
                bg='#2d2d2d', fg='#90CAF9').pack(anchor=tk.W)
        self.time_label = tk.Label(time_frame, text="00:00:00", 
                                   font=('Segoe UI', 16, 'bold'),
                                   bg='#2d2d2d', fg='#ffffff')
        self.time_label.pack(anchor=tk.W)
        
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Contador de personas
        people_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        people_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(people_frame, text="👥 Personas Detectadas:", font=('Segoe UI', 11, 'bold'),
                bg='#2d2d2d', fg='#90CAF9').pack(anchor=tk.W)
        
        self.people_count_label = tk.Label(people_frame, text="Ahora: 0 | Total: 0", 
                                          font=('Segoe UI', 14, 'bold'),
                                          bg='#2d2d2d', fg='#FFF9C4')
        self.people_count_label.pack(anchor=tk.W, pady=5)
        
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Porcentajes globales
        perc_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        perc_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(perc_frame, text="📈 Promedio Grupal:", font=('Segoe UI', 11, 'bold'),
                bg='#2d2d2d', fg='#90CAF9').pack(anchor=tk.W)
        
        atento_frame = tk.Frame(perc_frame, bg='#2d2d2d')
        atento_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(atento_frame, text="✅ Atento:", font=('Segoe UI', 10, 'bold'),
                bg='#2d2d2d', fg='#A5D6A7', width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.atento_label = tk.Label(atento_frame, text="0%", 
                                     font=('Segoe UI', 14, 'bold'),
                                     bg='#2d2d2d', fg='#4CAF50')
        self.atento_label.pack(side=tk.LEFT)
        
        desatento_frame = tk.Frame(perc_frame, bg='#2d2d2d')
        desatento_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(desatento_frame, text="❌ Desatento:", font=('Segoe UI', 10, 'bold'),
                bg='#2d2d2d', fg='#EF9A9A', width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.desatento_label = tk.Label(desatento_frame, text="0%", 
                                       font=('Segoe UI', 14, 'bold'),
                                       bg='#2d2d2d', fg='#f44336')
        self.desatento_label.pack(side=tk.LEFT)
        
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Lista de personas con scroll
        list_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        list_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        tk.Label(list_frame, text="📋 Personas Activas:", font=('Segoe UI', 11, 'bold'),
                bg='#2d2d2d', fg='#90CAF9').pack(anchor=tk.W)
        
        # Scrollbar
        scroll_frame = tk.Frame(list_frame, bg='#2d2d2d')
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.people_list = tk.Text(scroll_frame, height=15, width=40,
                                  font=('Consolas', 9),
                                  bg='#1e1e1e', fg='#ffffff',
                                  relief=tk.SUNKEN, bd=2,
                                  yscrollcommand=scrollbar.set,
                                  state=tk.DISABLED)
        self.people_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.people_list.yview)
        
        # Barra de estado
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="🔴 Sistema detenido", 
                                     font=('Segoe UI', 9),
                                     bg='#2d2d2d', fg='#B0BEC5',
                                     anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", 
                                  font=('Segoe UI', 9),
                                  bg='#2d2d2d', fg='#B0BEC5',
                                  anchor=tk.E)
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Progreso de video
        self.progress_frame = tk.Frame(status_frame, bg='#2d2d2d')
        self.progress_label = tk.Label(self.progress_frame, text="", 
                                       font=('Segoe UI', 9),
                                       bg='#2d2d2d', fg='#B0BEC5')
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)
    
    def load_models(self):
        """Carga los modelos"""
        try:
            self.status_label.config(text="⏳ Cargando modelos...")
            self.root.update()
            
            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            ensure_yunet()
            self.detector = cv2.FaceDetectorYN.create(
                model=YUNET_ONNX.as_posix(),
                config="",
                input_size=(DETECT_W, DETECT_H),
                score_threshold=SCORE_TH,
                nms_threshold=NMS_TH,
                top_k=TOP_K
            )
            
            self.yolo_model = ensure_yolo()
            
            self.status_label.config(text="✅ Modelos cargados (Multi-Persona)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos:\n{str(e)}")
            self.status_label.config(text="❌ Error al cargar modelos")
    
    def process_video_file(self):
        """Seleccionar y procesar video"""
        if self.is_running:
            messagebox.showwarning("Advertencia", "Ya hay una sesión en curso")
            return
        
        video_path = filedialog.askopenfilename(
            title="Seleccionar video de Zoom/clase",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not video_path:
            return
        
        self.video_path = video_path
        self.video_mode = "file"
        
        temp_cap = cv2.VideoCapture(video_path)
        if not temp_cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video")
            return
        
        self.total_video_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
        duration_sec = self.total_video_frames / video_fps if video_fps > 0 else 0
        temp_cap.release()
        
        duration_str = str(timedelta(seconds=int(duration_sec)))
        msg = f"Video seleccionado:\n\n"
        msg += f"📁 {Path(video_path).name}\n"
        msg += f"⏱ Duración: {duration_str}\n"
        msg += f"🎬 Frames: {self.total_video_frames}\n"
        msg += f"📹 FPS: {video_fps:.1f}\n\n"
        msg += "Se detectarán TODAS las personas en el video.\n"
        msg += "¿Desea procesar este video?\n(Puede tardar varios minutos)"
        
        if not messagebox.askyesno("Procesar Video Multi-Persona", msg):
            self.video_mode = "camera"
            self.video_path = None
            return
        
        self.start_session()
    
    def start_session(self):
        """Inicia sesión"""
        if self.is_running:
            return
        
        try:
            if self.video_mode == "file" and self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    raise RuntimeError("No se pudo abrir el video")
                self.progress_frame.pack(side=tk.LEFT, padx=10)
                self.current_video_frame = 0
            else:
                self.video_mode = "camera"
                self.cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(CAM_INDEX, 0)
                if not self.cap.isOpened():
                    raise RuntimeError("No se pudo abrir la cámara")
            
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Reset
            self.session_stats = {
                'start_time': datetime.now(),
                'end_time': None,
                'total_frames': 0,
                'people': {},
                'global_stats': {
                    'total_people_detected': 0,
                    'max_simultaneous': 0,
                    'frames_with_faces': 0
                }
            }
            
            self.frame_id = 0
            self.person_trackers = {}
            self.next_person_id = 1
            
            self.is_running = True
            self.is_paused = False
            
            # UI
            self.btn_start.config(state=tk.DISABLED)
            self.btn_video.config(state=tk.DISABLED)
            
            if self.video_mode == "camera":
                self.btn_pause.config(state=tk.NORMAL)
            else:
                self.btn_pause.config(state=tk.DISABLED)
            
            self.btn_stop.config(state=tk.NORMAL)
            
            mode_text = "video de Zoom/clase" if self.video_mode == "file" else "cámara multi-persona"
            self.status_label.config(text=f"🟢 Procesando {mode_text}")
            
            # Hilo
            self.process_thread = threading.Thread(target=self.process_video, daemon=True)
            self.process_thread.start()
            
            self.update_time()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar:\n{str(e)}")
            self.is_running = False
    
    def pause_session(self):
        """Pausa/reanuda"""
        if not self.is_running or self.video_mode == "file":
            return
        
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.btn_pause.config(text="▶ Reanudar")
            self.status_label.config(text="⏸ Sesión pausada")
        else:
            self.btn_pause.config(text="⏸ Pausar")
            self.status_label.config(text="🟢 Sesión en curso")
    
    def stop_session(self):
        """Detiene sesión"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.session_stats['end_time'] = datetime.now()
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_video.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ Pausar")
        self.btn_stop.config(state=tk.DISABLED)
        self.progress_frame.pack_forget()
        self.status_label.config(text="🔴 Generando reportes...")
        
        if self.cap:
            self.cap.release()
        
        report_path = self.generate_report()
        
        self.video_mode = "camera"
        self.video_path = None
        self.total_video_frames = 0
        self.current_video_frame = 0
        
        self.status_label.config(text="✅ Reportes generados")
        
        total_people = self.session_stats['global_stats']['total_people_detected']
        msg = f"Sesión terminada.\n\n"
        msg += f"👥 Total de personas detectadas: {total_people}\n\n"
        msg += f"El reporte PDF ha sido generado:\n{report_path.name}\n\n"
        msg += "📂 Ubicación: reportes/"
        messagebox.showinfo("Sesión Finalizada", msg)
    
    def process_video(self):
        """Procesa video frame por frame"""
        t0, n = time.time(), 0
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ok, frame = self.cap.read()
            if not ok:
                if self.video_mode == "file":
                    self.root.after(0, self.stop_session)
                break
            
            if self.video_mode == "file":
                self.current_video_frame += 1
                progress = (self.current_video_frame / self.total_video_frames) * 100
                self.progress_bar['value'] = progress
                self.progress_label.config(
                    text=f"Progreso: {self.current_video_frame}/{self.total_video_frames} ({progress:.1f}%)"
                )
            
            H, W = frame.shape[:2]
            
            # Mejorar calidad del frame antes de procesar (solo para detección)
            enhanced_frame = enhance_frame(frame) if self.video_mode == "file" else frame
            
            # Detectar TODAS las caras (usar frame mejorado para detección)
            faces_data = self.detect_all_faces(enhanced_frame, W, H)
            
            # Match con personas existentes o crear nuevas
            self.update_person_tracking(faces_data)
            
            # Procesar cada persona
            current_people = []
            for person_id, tracker in list(self.person_trackers.items()):
                if tracker['last_seen'] == self.frame_id:
                    # Clasificar estado (usar frame mejorado para mejor clasificación)
                    state, reason = self.classify_person_state(enhanced_frame, tracker)
                    
                    # Actualizar stats
                    if person_id not in self.session_stats['people']:
                        self.session_stats['people'][person_id] = {
                            'total_frames': 0,
                            'atento_frames': 0,
                            'desatento_frames': 0,
                            'objetos_detectados': {},
                            'pose_issues': defaultdict(int),
                            'color': PERSON_COLORS[person_id % len(PERSON_COLORS)]
                        }
                    
                    person_stats = self.session_stats['people'][person_id]
                    person_stats['total_frames'] += 1
                    
                    if state == "ATENTO":
                        person_stats['atento_frames'] += 1
                    else:
                        person_stats['desatento_frames'] += 1
                    
                    # Dibujar en frame
                    frame = self.draw_person_ui(frame, person_id, tracker, state, reason)
                    current_people.append((person_id, state, reason))
                elif self.frame_id - tracker['last_seen'] > PERSON_TIMEOUT:
                    # Eliminar persona que no se ve hace mucho
                    del self.person_trackers[person_id]
            
            # Actualizar stats globales
            self.session_stats['total_frames'] += 1
            if len(current_people) > 0:
                self.session_stats['global_stats']['frames_with_faces'] += 1
                self.session_stats['global_stats']['max_simultaneous'] = max(
                    self.session_stats['global_stats']['max_simultaneous'],
                    len(current_people)
                )
            
            # FPS
            n += 1
            if time.time() - t0 >= 0.5:
                fps = n / (time.time() - t0)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                t0, n = time.time(), 0
            
            # Mostrar (solo cada 2 frames para reducir overhead)
            # Usar frame original para visualización (más natural)
            if self.frame_id % 2 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((800, 450), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Actualizar UI
                self.update_stats_ui(current_people)
            
            self.frame_id += 1
    
    def detect_all_faces(self, frame, W, H):
        """Detecta todas las caras en el frame"""
        if self.frame_id % DETECT_EVERY != 0:
            return []
        
        small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
        self.detector.setInputSize((DETECT_W, DETECT_H))
        res = self.detector.detect(small)
        faces = res[1] if isinstance(res, tuple) else res
        
        if not isinstance(faces, np.ndarray) or faces.size == 0:
            return []
        
        faces_data = []
        for face_data in faces:
            x, y, w, h = face_data[:4].astype(int)
            
            # Re-escalar
            scale_x = W / DETECT_W
            scale_y = H / DETECT_H
            x = int(x * scale_x); y = int(y * scale_y)
            w = int(w * scale_x); h = int(h * scale_y)
            
            # Filtrar rostros muy pequeños
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            
            # Landmarks
            landmarks = None
            if len(face_data) >= 15:
                lm = face_data[4:14].reshape(5, 2)
                lm[:, 0] *= scale_x
                lm[:, 1] *= scale_y
                landmarks = lm
            
            faces_data.append({
                'bbox': [x, y, w, h],
                'landmarks': landmarks,
                'confidence': float(face_data[14]) if len(face_data) > 14 else 1.0
            })
        
        return faces_data
    
    def update_person_tracking(self, faces_data):
        """Match caras con personas trackeadas o crear nuevas"""
        # Marcar todas como no vistas este frame
        for tracker in self.person_trackers.values():
            tracker['frame_seen'] = False
        
        # Match cada cara detectada
        for face in faces_data:
            best_match_id = None
            best_iou = 0
            
            # Buscar mejor match con personas existentes
            for person_id, tracker in self.person_trackers.items():
                if self.frame_id - tracker['last_seen'] > PERSON_TIMEOUT:
                    continue
                
                iou = calculate_iou(face['bbox'], tracker['bbox'])
                if iou > IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_match_id = person_id
            
            if best_match_id:
                # Actualizar persona existente
                self.person_trackers[best_match_id].update({
                    'bbox': face['bbox'],
                    'landmarks': face['landmarks'],
                    'last_seen': self.frame_id,
                    'frame_seen': True
                })
            else:
                # Nueva persona
                person_id = self.next_person_id
                self.next_person_id += 1
                
                self.person_trackers[person_id] = {
                    'bbox': face['bbox'],
                    'landmarks': face['landmarks'],
                    'last_seen': self.frame_id,
                    'frame_seen': True,
                    'prob_ema': None,
                    'yaw_ema': None,
                    'pitch_ema': None,
                    'pose_counter': 0,
                    'last_objects': [],
                    'object_memory': 0
                }
                
                self.session_stats['global_stats']['total_people_detected'] += 1
    
    def classify_person_state(self, frame, tracker):
        """Clasifica estado de una persona"""
        bbox = tracker['bbox']
        landmarks = tracker['landmarks']
        
        # Clasificación facial
        if self.frame_id % CLASSIFY_EVERY == 0:
            prob = classify_face(frame, bbox, self.model)
            tracker['prob_ema'] = prob if tracker['prob_ema'] is None else \
                (SMOOTH_ALPHA_PROB * tracker['prob_ema'] + (1 - SMOOTH_ALPHA_PROB) * prob)
        
        is_distracted_face = tracker['prob_ema'] >= UMBRAL if tracker['prob_ema'] else False
        
        # Pose
        reasons = []
        if landmarks is not None:
            yaw, pitch = estimate_head_pose(landmarks)
            
            if tracker['yaw_ema'] is None:
                tracker['yaw_ema'] = yaw
                tracker['pitch_ema'] = pitch
            else:
                tracker['yaw_ema'] = ANGLE_ALPHA * yaw + (1 - ANGLE_ALPHA) * tracker['yaw_ema']
                tracker['pitch_ema'] = ANGLE_ALPHA * pitch + (1 - ANGLE_ALPHA) * tracker['pitch_ema']
            
            pose_bad = False
            if abs(tracker['yaw_ema']) > HEAD_POSE_THRESHOLD:
                pose_bad = True
                direction = "derecha" if tracker['yaw_ema'] > 0 else "izquierda"
                if tracker['pose_counter'] >= POSE_CONFIRMATION_FRAMES:
                    reasons.append(f"Mirando {direction}")
            
            if abs(tracker['pitch_ema']) > HEAD_DOWN_THRESHOLD:
                pose_bad = True
                direction = "abajo" if tracker['pitch_ema'] > 0 else "arriba"
                if tracker['pose_counter'] >= POSE_CONFIRMATION_FRAMES:
                    reasons.append(f"Mirando {direction}")
            
            tracker['pose_counter'] = min(tracker['pose_counter'] + 1, POSE_CONFIRMATION_FRAMES + 1) if pose_bad else max(0, tracker['pose_counter'] - 1)
        
        # Objetos
        has_distractor = False
        if self.yolo_model and self.frame_id % OBJECT_EVERY == 0:
            has_distractor, detected_objs = detect_distractors(frame, self.yolo_model, bbox)
            if has_distractor:
                tracker['last_objects'] = detected_objs
                tracker['object_memory'] = OBJECT_MEMORY_DURATION
                obj_names = [obj['name'] for obj in detected_objs]
                reasons.append(f"{', '.join(obj_names).upper()}")
        
        if not has_distractor and tracker['object_memory'] > 0:
            tracker['object_memory'] -= 1
            if tracker['last_objects']:
                has_distractor = True
                obj_names = [obj['name'] for obj in tracker['last_objects']]
                reasons.append(f"{', '.join(obj_names).upper()}")
        
        # Decisión
        has_object = any(r.isupper() for r in reasons)
        has_pose = any("Mirando" in r for r in reasons)
        
        if has_object:
            return "DESATENTO", [r for r in reasons if r.isupper()][0]
        elif has_pose:
            return "DESATENTO", [r for r in reasons if "Mirando" in r][0].upper()
        elif is_distracted_face:
            return "DESATENTO", "NO CONCENTRADO"
        else:
            return "ATENTO", "Concentrado"
    
    def draw_person_ui(self, frame, person_id, tracker, state, reason):
        """Dibuja UI de una persona en el frame"""
        x, y, w, h = tracker['bbox']
        
        # Color según persona
        color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
        
        # Bbox
        thickness = 3 if state == "DESATENTO" else 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # ID y estado
        label = f"P{person_id}: {state[:8]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Fondo para texto
        cv2.rectangle(frame, (x, y-th-8), (x+tw+8, y), color, -1)
        cv2.putText(frame, label, (x+4, y-4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def update_stats_ui(self, current_people):
        """Actualiza estadísticas en UI"""
        # Contador
        total_detected = self.session_stats['global_stats']['total_people_detected']
        self.people_count_label.config(text=f"Ahora: {len(current_people)} | Total: {total_detected}")
        
        # Promedio grupal
        if self.session_stats['people']:
            total_atento = sum(p['atento_frames'] for p in self.session_stats['people'].values())
            total_frames = sum(p['total_frames'] for p in self.session_stats['people'].values())
            
            if total_frames > 0:
                atento_pct = (total_atento / total_frames) * 100
                desatento_pct = 100 - atento_pct
                
                self.atento_label.config(text=f"{atento_pct:.1f}%")
                self.desatento_label.config(text=f"{desatento_pct:.1f}%")
        
        # Lista de personas
        self.people_list.config(state=tk.NORMAL)
        self.people_list.delete(1.0, tk.END)
        
        for person_id, state, reason in current_people:
            if person_id in self.session_stats['people']:
                person_stats = self.session_stats['people'][person_id]
                total = person_stats['total_frames']
                atento = person_stats['atento_frames']
                
                if total > 0:
                    atento_pct = (atento / total) * 100
                    icon = "✅" if state == "ATENTO" else "❌"
                    self.people_list.insert(tk.END, 
                        f"{icon} P{person_id}: {atento_pct:.0f}% atento | {reason[:15]}\n")
        
        if not current_people:
            self.people_list.insert(tk.END, "Ninguna persona activa ahora")
        
        self.people_list.config(state=tk.DISABLED)
    
    def update_time(self):
        """Actualiza tiempo"""
        if self.is_running and self.session_stats['start_time']:
            if self.video_mode == "file" and self.total_video_frames > 0:
                video_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
                elapsed_seconds = int(self.current_video_frame / video_fps)
                hours, remainder = divmod(elapsed_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                if not self.is_paused:
                    elapsed = datetime.now() - self.session_stats['start_time']
                    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    return
            
            self.time_label.config(text=time_str)
            self.root.after(1000, self.update_time)
    
    def generate_report(self):
        """Genera reporte PDF multi-persona"""
        reports_dir = Path("reportes")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = self.session_stats['start_time'].strftime("%Y%m%d_%H%M%S")
        mode_suffix = "multi_video" if self.video_mode == "file" else "multi_camara"
        pdf_path = reports_dir / f"reporte_atencion_{mode_suffix}_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1976D2'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#424242'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        elements = []
        
        # Título
        elements.append(Paragraph("📊 Reporte Multi-Persona de Atención", title_style))
        elements.append(Paragraph("(Zoom / Clase Grupal)", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Info general
        start = self.session_stats['start_time'].strftime("%d/%m/%Y %H:%M:%S")
        end = self.session_stats['end_time'].strftime("%d/%m/%Y %H:%M:%S")
        duration = self.session_stats['end_time'] - self.session_stats['start_time']
        
        info_data = [
            ['Información General', ''],
            ['Modo:', 'Video Zoom/Clase' if self.video_mode == "file" else 'Cámara Multi-Persona'],
            ['Inicio:', start],
            ['Fin:', end],
            ['Duración:', str(duration).split('.')[0]],
            ['Total de frames:', str(self.session_stats['total_frames'])],
            ['Personas detectadas:', str(self.session_stats['global_stats']['total_people_detected'])],
            ['Máx. simultáneas:', str(self.session_stats['global_stats']['max_simultaneous'])]
        ]
        
        if self.video_mode == "file" and self.video_path:
            info_data.insert(2, ['Archivo:', Path(self.video_path).name])
        
        info_table = Table(info_data, colWidths=[200, 300])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        # Estadísticas por persona
        elements.append(Paragraph("👥 Estadísticas Individuales", heading_style))
        
        person_data = [['Persona', 'Frames', '% Atento', '% Desatento', 'Estado']]
        
        for person_id in sorted(self.session_stats['people'].keys()):
            person_stats = self.session_stats['people'][person_id]
            total = person_stats['total_frames']
            atento = person_stats['atento_frames']
            desatento = person_stats['desatento_frames']
            
            if total > 0:
                atento_pct = (atento / total) * 100
                desatento_pct = (desatento / total) * 100
                
                estado = "✅ Excelente" if atento_pct >= 80 else "⚠️ Aceptable" if atento_pct >= 60 else "❌ Bajo"
                
                person_data.append([
                    f"Persona {person_id}",
                    str(total),
                    f"{atento_pct:.1f}%",
                    f"{desatento_pct:.1f}%",
                    estado
                ])
        
        person_table = Table(person_data, colWidths=[100, 80, 80, 100, 120])
        person_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#424242')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(person_table)
        elements.append(Spacer(1, 20))
        
        # Promedio grupal
        elements.append(Paragraph("📈 Promedio Grupal", heading_style))
        
        total_atento = sum(p['atento_frames'] for p in self.session_stats['people'].values())
        total_frames_all = sum(p['total_frames'] for p in self.session_stats['people'].values())
        
        if total_frames_all > 0:
            avg_atento = (total_atento / total_frames_all) * 100
            avg_desatento = 100 - avg_atento
            
            avg_data = [
                ['Métrica', 'Valor'],
                ['Promedio Atento', f'{avg_atento:.1f}%'],
                ['Promedio Desatento', f'{avg_desatento:.1f}%'],
                ['Total Personas', str(self.session_stats['global_stats']['total_people_detected'])],
                ['Máx. Simultáneas', str(self.session_stats['global_stats']['max_simultaneous'])]
            ]
            
            avg_table = Table(avg_data, colWidths=[250, 200])
            avg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F5E9')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(avg_table)
            elements.append(Spacer(1, 20))
        
        # Conclusiones
        elements.append(Paragraph("📝 Conclusiones", heading_style))
        
        conclusion = f"""
        Durante esta sesión multi-persona de {str(duration).split('.')[0]}, el sistema detectó:
        <br/><br/>
        • <b>{self.session_stats['global_stats']['total_people_detected']} personas</b> en total.
        <br/>
        • Máximo de <b>{self.session_stats['global_stats']['max_simultaneous']} personas simultáneas</b>.
        <br/>
        • Promedio grupal de atención: <b>{avg_atento:.1f}%</b>
        """
        
        if avg_atento >= 75:
            conclusion += "<br/><br/>✅ <b>Excelente nivel de atención grupal!</b>"
        elif avg_atento >= 60:
            conclusion += "<br/><br/>⚠️ <b>Nivel de atención grupal aceptable.</b>"
        else:
            conclusion += "<br/><br/>❌ <b>Nivel de atención grupal bajo. Revisar distractores.</b>"
        
        elements.append(Paragraph(conclusion, styles['BodyText']))
        elements.append(Spacer(1, 20))
        
        # Footer
        footer_text = f"<i>Reporte multi-persona generado - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
        elements.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(elements)
        
        print(f"[INFO] Reporte multi-persona generado: {pdf_path}")
        return pdf_path

# ========= Main =====
if __name__ == "__main__":
    root = tk.Tk()
    app = AttentionMonitorMultiApp(root)
    root.mainloop()
