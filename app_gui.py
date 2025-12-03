# app_gui.py - Sistema de Detección de Atención con Interfaz Gráfica y Reportes
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ========= Configuración (mismo que app_enhanced.py) =========
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
DETECT_W, DETECT_H = 320, 180
SCORE_TH = 0.6
NMS_TH = 0.3
TOP_K = 1

YOLO_MODEL = Path("modelos/yolov8n.pt")
DISTRACTOR_CLASSES = [67, 63, 73, 64, 75, 66]
OBJECT_EVERY = 5

HEAD_POSE_THRESHOLD = 35
HEAD_DOWN_THRESHOLD = 40

DETECT_EVERY = 3
CLASSIFY_EVERY = 3
MISS_TOLERANCE = 6
SMOOTH_ALPHA_BBOX = 0.7
SMOOTH_ALPHA_PROB = 0.6
ANGLE_ALPHA = 0.25
POSE_CONFIRMATION_FRAMES = 3
OBJECT_MEMORY_DURATION = 15

# ========= Funciones auxiliares (reutilizadas) =========
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

def smooth_bbox(prev, curr, alpha=SMOOTH_ALPHA_BBOX):
    if prev is None: return curr
    return [
        int(alpha*prev[0] + (1-alpha)*curr[0]),
        int(alpha*prev[1] + (1-alpha)*curr[1]),
        int(alpha*prev[2] + (1-alpha)*curr[2]),
        int(alpha*prev[3] + (1-alpha)*curr[3]),
    ]

def clip_bbox(b, W, H):
    x,y,w,h = b
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
    return [x,y,w,h]

def classify_face(frame_bgr, bbox_xywh, model):
    x,y,w,h = bbox_xywh
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
        results = yolo_model(frame, verbose=False, conf=0.30, iou=0.45, imgsz=640, max_det=15)
        
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

# ========= Clase Principal de la Aplicación =========
class AttentionMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Monitoreo de Atención - Deep Learning")
        self.root.geometry("1400x850")
        self.root.configure(bg='#1e1e1e')
        
        # Variables de estado
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.detector = None
        self.model = None
        self.yolo_model = None
        self.video_mode = "camera"  # "camera" o "file"
        self.video_path = None
        self.total_video_frames = 0
        self.current_video_frame = 0
        
        # Estadísticas de sesión
        self.session_stats = {
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'atento_frames': 0,
            'desatento_frames': 0,
            'sin_rostro_frames': 0,
            'objetos_detectados': {},  # {objeto: count}
            'pose_issues': {'izquierda': 0, 'derecha': 0, 'arriba': 0, 'abajo': 0},
            'timeline': []  # [(timestamp, estado, razon)]
        }
        
        # Variables de procesamiento
        self.prev_bbox = None
        self.last_ok_bbox = None
        self.miss_counter = 0
        self.prob_ema = None
        self.frame_id = 0
        self.last_landmarks = None
        self.yaw_ema = None
        self.pitch_ema = None
        self.pose_distracted_count = 0
        self.last_detected_objects = []
        self.object_memory_frames = 0
        
        # Crear UI
        self.create_ui()
        
        # Cargar modelos
        self.load_models()
    
    def create_ui(self):
        """Crea la interfaz gráfica moderna"""
        
        # ===== Panel Superior: Controles =====
        control_frame = tk.Frame(self.root, bg='#2d2d2d', height=100)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Título
        title_label = tk.Label(control_frame, text="🎯 Monitor de Atención", 
                               font=('Segoe UI', 20, 'bold'), 
                               bg='#2d2d2d', fg='#ffffff')
        title_label.pack(pady=10)
        
        # Botones de control
        btn_frame = tk.Frame(control_frame, bg='#2d2d2d')
        btn_frame.pack(pady=5)
        
        self.btn_start = tk.Button(btn_frame, text="▶ Iniciar Sesión", 
                                   command=self.start_session,
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#4CAF50', fg='white',
                                   width=15, height=2,
                                   relief=tk.RAISED, bd=3,
                                   cursor='hand2')
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_pause = tk.Button(btn_frame, text="⏸ Pausar", 
                                   command=self.pause_session,
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#FF9800', fg='white',
                                   width=15, height=2,
                                   relief=tk.RAISED, bd=3,
                                   cursor='hand2', state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(btn_frame, text="⏹ Finalizar y Reporte", 
                                 command=self.stop_session,
                                 font=('Segoe UI', 12, 'bold'),
                                 bg='#f44336', fg='white',
                                 width=20, height=2,
                                 relief=tk.RAISED, bd=3,
                                 cursor='hand2', state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        # Botón para procesar video
        self.btn_video = tk.Button(btn_frame, text="📁 Procesar Video", 
                                   command=self.process_video_file,
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#9C27B0', fg='white',
                                   width=18, height=2,
                                   relief=tk.RAISED, bd=3,
                                   cursor='hand2')
        self.btn_video.pack(side=tk.LEFT, padx=5)
        
        # ===== Panel Central: Video + Estadísticas =====
        center_frame = tk.Frame(self.root, bg='#1e1e1e')
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Video a la izquierda
        video_frame = tk.LabelFrame(center_frame, text="📹 Video en Vivo", 
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='#2d2d2d', fg='#ffffff',
                                    bd=2, relief=tk.GROOVE)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(padx=10, pady=10)
        
        # Panel de estadísticas a la derecha
        stats_frame = tk.LabelFrame(center_frame, text="📊 Estadísticas en Tiempo Real", 
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='#2d2d2d', fg='#ffffff',
                                    bd=2, relief=tk.GROOVE, width=400)
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
        
        # Separador
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Estado actual
        state_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        state_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(state_frame, text="📍 Estado Actual:", font=('Segoe UI', 11, 'bold'),
                bg='#2d2d2d', fg='#90CAF9').pack(anchor=tk.W)
        self.state_label = tk.Label(state_frame, text="Sin iniciar", 
                                    font=('Segoe UI', 14, 'bold'),
                                    bg='#2d2d2d', fg='#FFF9C4',
                                    wraplength=350, justify=tk.LEFT)
        self.state_label.pack(anchor=tk.W, pady=5)
        
        # Separador
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Porcentajes
        perc_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        perc_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # Atento
        atento_frame = tk.Frame(perc_frame, bg='#2d2d2d')
        atento_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(atento_frame, text="✅ Atento:", font=('Segoe UI', 10, 'bold'),
                bg='#2d2d2d', fg='#A5D6A7', width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.atento_label = tk.Label(atento_frame, text="0%", 
                                     font=('Segoe UI', 14, 'bold'),
                                     bg='#2d2d2d', fg='#4CAF50')
        self.atento_label.pack(side=tk.LEFT)
        
        # Desatento
        desatento_frame = tk.Frame(perc_frame, bg='#2d2d2d')
        desatento_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(desatento_frame, text="❌ Desatento:", font=('Segoe UI', 10, 'bold'),
                bg='#2d2d2d', fg='#EF9A9A', width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.desatento_label = tk.Label(desatento_frame, text="0%", 
                                       font=('Segoe UI', 14, 'bold'),
                                       bg='#2d2d2d', fg='#f44336')
        self.desatento_label.pack(side=tk.LEFT)
        
        # Sin rostro
        noface_frame = tk.Frame(perc_frame, bg='#2d2d2d')
        noface_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(noface_frame, text="⚠️ Sin rostro:", font=('Segoe UI', 10, 'bold'),
                bg='#2d2d2d', fg='#FFE082', width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.noface_label = tk.Label(noface_frame, text="0%", 
                                     font=('Segoe UI', 14, 'bold'),
                                     bg='#2d2d2d', fg='#FFC107')
        self.noface_label.pack(side=tk.LEFT)
        
        # Separador
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Objetos detectados
        obj_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        obj_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        tk.Label(obj_frame, text="📱 Objetos Detectados:", font=('Segoe UI', 11, 'bold'),
                bg='#2d2d2d', fg='#90CAF9').pack(anchor=tk.W)
        
        self.obj_text = tk.Text(obj_frame, height=6, width=30, 
                               font=('Consolas', 9),
                               bg='#1e1e1e', fg='#ffffff',
                               relief=tk.SUNKEN, bd=2,
                               state=tk.DISABLED)
        self.obj_text.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # ===== Barra de Estado =====
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
        
        # Progreso de video (oculto inicialmente)
        self.progress_frame = tk.Frame(status_frame, bg='#2d2d2d')
        self.progress_label = tk.Label(self.progress_frame, text="", 
                                       font=('Segoe UI', 9),
                                       bg='#2d2d2d', fg='#B0BEC5')
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)
    
    def load_models(self):
        """Carga los modelos de ML"""
        try:
            self.status_label.config(text="⏳ Cargando modelos...")
            self.root.update()
            
            # Cargar modelo de clasificación
            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            # Cargar YuNet
            ensure_yunet()
            self.detector = cv2.FaceDetectorYN.create(
                model=YUNET_ONNX.as_posix(),
                config="",
                input_size=(DETECT_W, DETECT_H),
                score_threshold=SCORE_TH,
                nms_threshold=NMS_TH,
                top_k=TOP_K
            )
            
            # Cargar YOLO
            self.yolo_model = ensure_yolo()
            
            self.status_label.config(text="✅ Modelos cargados correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos:\n{str(e)}")
            self.status_label.config(text="❌ Error al cargar modelos")
    
    def process_video_file(self):
        """Permite seleccionar y procesar un video grabado"""
        if self.is_running:
            messagebox.showwarning("Advertencia", "Ya hay una sesión en curso")
            return
        
        # Seleccionar archivo de video
        video_path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not video_path:
            return
        
        self.video_path = video_path
        self.video_mode = "file"
        
        # Obtener información del video
        temp_cap = cv2.VideoCapture(video_path)
        if not temp_cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video")
            return
        
        self.total_video_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
        duration_sec = self.total_video_frames / video_fps if video_fps > 0 else 0
        temp_cap.release()
        
        # Confirmar procesamiento
        duration_str = str(timedelta(seconds=int(duration_sec)))
        msg = f"Video seleccionado:\n\n"
        msg += f"📁 {Path(video_path).name}\n"
        msg += f"⏱ Duración: {duration_str}\n"
        msg += f"🎬 Frames: {self.total_video_frames}\n"
        msg += f"📹 FPS: {video_fps:.1f}\n\n"
        msg += "¿Desea procesar este video?\n(Puede tardar varios minutos)"
        
        if not messagebox.askyesno("Procesar Video", msg):
            self.video_mode = "camera"
            self.video_path = None
            return
        
        # Iniciar procesamiento
        self.start_session()
    
    def start_session(self):
        """Inicia una nueva sesión de monitoreo (cámara o video)"""
        if self.is_running:
            return
        
        try:
            # Abrir fuente de video (cámara o archivo)
            if self.video_mode == "file" and self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    raise RuntimeError("No se pudo abrir el video")
                
                # Mostrar barra de progreso
                self.progress_frame.pack(side=tk.LEFT, padx=10)
                self.current_video_frame = 0
                
            else:
                # Modo cámara
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
            
            # Resetear estadísticas
            self.session_stats = {
                'start_time': datetime.now(),
                'end_time': None,
                'total_frames': 0,
                'atento_frames': 0,
                'desatento_frames': 0,
                'sin_rostro_frames': 0,
                'objetos_detectados': {},
                'pose_issues': {'izquierda': 0, 'derecha': 0, 'arriba': 0, 'abajo': 0},
                'timeline': []
            }
            
            # Resetear variables de procesamiento
            self.prev_bbox = None
            self.last_ok_bbox = None
            self.miss_counter = 0
            self.prob_ema = None
            self.frame_id = 0
            self.last_landmarks = None
            self.yaw_ema = None
            self.pitch_ema = None
            self.pose_distracted_count = 0
            self.last_detected_objects = []
            self.object_memory_frames = 0
            
            self.is_running = True
            self.is_paused = False
            
            # Actualizar UI
            self.btn_start.config(state=tk.DISABLED)
            self.btn_video.config(state=tk.DISABLED)
            
            if self.video_mode == "camera":
                self.btn_pause.config(state=tk.NORMAL)
            else:
                self.btn_pause.config(state=tk.DISABLED)  # No pausar videos
            
            self.btn_stop.config(state=tk.NORMAL)
            
            mode_text = "archivo de video" if self.video_mode == "file" else "cámara en vivo"
            self.status_label.config(text=f"🟢 Procesando {mode_text}")
            
            # Iniciar hilo de procesamiento
            self.process_thread = threading.Thread(target=self.process_video, daemon=True)
            self.process_thread.start()
            
            # Iniciar actualización de tiempo
            self.update_time()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar sesión:\n{str(e)}")
            self.is_running = False
    
    def pause_session(self):
        """Pausa/reanuda la sesión (solo para cámara)"""
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
        """Detiene la sesión y genera reporte"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.session_stats['end_time'] = datetime.now()
        
        # Actualizar UI
        self.btn_start.config(state=tk.NORMAL)
        self.btn_video.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ Pausar")
        self.btn_stop.config(state=tk.DISABLED)
        self.progress_frame.pack_forget()  # Ocultar barra de progreso
        self.status_label.config(text="🔴 Generando reporte...")
        
        # Liberar cámara
        if self.cap:
            self.cap.release()
        
        # Generar reporte
        report_path = self.generate_report()
        
        # Resetear modo
        self.video_mode = "camera"
        self.video_path = None
        self.total_video_frames = 0
        self.current_video_frame = 0
        
        self.status_label.config(text="✅ Reporte generado")
        
        msg = "Sesión terminada.\n\n"
        msg += f"El reporte PDF ha sido generado:\n{report_path.name}\n\n"
        msg += "📂 Ubicación: reportes/"
        messagebox.showinfo("Sesión Finalizada", msg)
    
    def process_video(self):
        """Procesa el video frame por frame (hilo separado)"""
        t0, n = time.time(), 0
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ok, frame = self.cap.read()
            if not ok:
                # Si es video, terminó de procesar
                if self.video_mode == "file":
                    self.root.after(0, self.stop_session)
                break
            
            # Actualizar progreso de video
            if self.video_mode == "file":
                self.current_video_frame += 1
                progress = (self.current_video_frame / self.total_video_frames) * 100
                self.progress_bar['value'] = progress
                self.progress_label.config(
                    text=f"Progreso: {self.current_video_frame}/{self.total_video_frames} ({progress:.1f}%)"
                )
            
            H, W = frame.shape[:2]
            
            # Procesar frame (misma lógica que app_enhanced.py)
            bbox, landmarks = self.detect_face(frame, W, H)
            
            if bbox is not None:
                # Clasificar estado
                state, reason = self.classify_state(frame, bbox, landmarks, W, H)
                
                # Actualizar estadísticas
                self.session_stats['total_frames'] += 1
                
                if state == "ATENTO":
                    self.session_stats['atento_frames'] += 1
                elif state == "DESATENTO":
                    self.session_stats['desatento_frames'] += 1
                    
                    # Contar objetos y poses
                    if "celular" in reason.lower():
                        self.session_stats['objetos_detectados']['celular'] = \
                            self.session_stats['objetos_detectados'].get('celular', 0) + 1
                    elif "laptop" in reason.lower():
                        self.session_stats['objetos_detectados']['laptop'] = \
                            self.session_stats['objetos_detectados'].get('laptop', 0) + 1
                    elif "libro" in reason.lower():
                        self.session_stats['objetos_detectados']['libro'] = \
                            self.session_stats['objetos_detectados'].get('libro', 0) + 1
                    
                    if "izquierda" in reason.lower():
                        self.session_stats['pose_issues']['izquierda'] += 1
                    elif "derecha" in reason.lower():
                        self.session_stats['pose_issues']['derecha'] += 1
                    elif "arriba" in reason.lower():
                        self.session_stats['pose_issues']['arriba'] += 1
                    elif "abajo" in reason.lower():
                        self.session_stats['pose_issues']['abajo'] += 1
                
                # Guardar en timeline (cada 5 segundos)
                if self.frame_id % 150 == 0:  # ~5 seg a 30fps
                    self.session_stats['timeline'].append(
                        (datetime.now(), state, reason)
                    )
                
                # Dibujar en frame
                frame = self.draw_ui(frame, bbox, state, reason)
            else:
                self.session_stats['sin_rostro_frames'] += 1
                cv2.putText(frame, "Sin rostro detectado", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # FPS
            n += 1
            if time.time() - t0 >= 0.5:
                fps = n / (time.time() - t0)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                t0, n = time.time(), 0
            
            # Mostrar frame en UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 360), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Actualizar estadísticas en UI
            self.update_stats_ui()
            
            self.frame_id += 1
    
    def detect_face(self, frame, W, H):
        """Detecta rostro y landmarks"""
        bbox = None
        landmarks = None
        
        if self.frame_id % DETECT_EVERY == 0:
            small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
            self.detector.setInputSize((DETECT_W, DETECT_H))
            res = self.detector.detect(small)
            faces = res[1] if isinstance(res, tuple) else res

            if isinstance(faces, np.ndarray) and faces.size > 0:
                face_data = faces[0]
                x, y, w, h = face_data[:4].astype(int)
                
                scale_x = W / DETECT_W
                scale_y = H / DETECT_H
                x = int(x * scale_x); y = int(y * scale_y)
                w = int(w * scale_x); h = int(h * scale_y)
                bbox = [x, y, w, h]
                
                if len(face_data) >= 15:
                    lm = face_data[4:14].reshape(5, 2)
                    lm[:, 0] *= scale_x
                    lm[:, 1] *= scale_y
                    landmarks = lm
                    self.last_landmarks = landmarks
                
                self.last_ok_bbox = bbox
                self.miss_counter = 0
        
        if bbox is None and self.last_ok_bbox is not None and self.miss_counter < MISS_TOLERANCE:
            self.miss_counter += 1
            bbox = self.last_ok_bbox
            landmarks = self.last_landmarks
        
        if bbox is not None:
            bbox = clip_bbox(bbox, W, H)
            bbox = smooth_bbox(self.prev_bbox, bbox)
            self.prev_bbox = bbox
        
        return bbox, landmarks
    
    def classify_state(self, frame, bbox, landmarks, W, H):
        """Clasifica el estado actual"""
        x, y, w, h = bbox
        reasons = []
        
        # Clasificación facial
        do_classify = (self.frame_id % CLASSIFY_EVERY == 0)
        if do_classify:
            prob = classify_face(frame, bbox, self.model)
            self.prob_ema = prob if self.prob_ema is None else \
                (SMOOTH_ALPHA_PROB*self.prob_ema + (1-SMOOTH_ALPHA_PROB)*prob)
        elif self.prob_ema is None:
            prob = classify_face(frame, bbox, self.model)
            self.prob_ema = prob
        
        is_distracted_face = self.prob_ema >= UMBRAL if self.prob_ema is not None else False
        
        # Análisis de pose
        if landmarks is not None:
            yaw, pitch = estimate_head_pose(landmarks)
            
            if self.yaw_ema is None or self.pitch_ema is None:
                self.yaw_ema = yaw
                self.pitch_ema = pitch
            else:
                self.yaw_ema = ANGLE_ALPHA * yaw + (1 - ANGLE_ALPHA) * self.yaw_ema
                self.pitch_ema = ANGLE_ALPHA * pitch + (1 - ANGLE_ALPHA) * self.pitch_ema
            
            pose_is_bad = False
            
            if abs(self.yaw_ema) > HEAD_POSE_THRESHOLD:
                pose_is_bad = True
                direction_h = "derecha" if self.yaw_ema > 0 else "izquierda"
                if self.pose_distracted_count >= POSE_CONFIRMATION_FRAMES:
                    reasons.append(f"Mirando {direction_h}")
            
            if abs(self.pitch_ema) > HEAD_DOWN_THRESHOLD:
                pose_is_bad = True
                direction_v = "abajo" if self.pitch_ema > 0 else "arriba"
                if self.pose_distracted_count >= POSE_CONFIRMATION_FRAMES:
                    reasons.append(f"Mirando {direction_v}")
            
            if pose_is_bad:
                self.pose_distracted_count = min(self.pose_distracted_count + 1, POSE_CONFIRMATION_FRAMES + 1)
            else:
                self.pose_distracted_count = max(0, self.pose_distracted_count - 1)
        
        # Detección de objetos
        has_distractor = False
        do_object_detect = (self.frame_id % OBJECT_EVERY == 0)
        
        if self.yolo_model and do_object_detect:
            has_distractor, detected_objs = detect_distractors(frame, self.yolo_model, bbox)
            
            if has_distractor:
                self.last_detected_objects = detected_objs.copy()
                self.object_memory_frames = OBJECT_MEMORY_DURATION
                obj_names = [obj['name'] for obj in detected_objs]
                reasons.append(f"{', '.join(obj_names).upper()}")
        
        if not has_distractor and self.object_memory_frames > 0:
            self.object_memory_frames -= 1
            if len(self.last_detected_objects) > 0:
                has_distractor = True
                obj_names = [obj['name'] for obj in self.last_detected_objects]
                reasons.append(f"{', '.join(obj_names).upper()}")
        
        # Decisión final
        has_object = len([r for r in reasons if r.isupper()]) > 0
        has_pose_issue = any("Mirando" in r for r in reasons)
        
        if has_object:
            obj_reason = [r for r in reasons if r.isupper()][0]
            return "DESATENTO", obj_reason
        elif has_pose_issue:
            pose_reason = [r for r in reasons if "Mirando" in r][0]
            return "DESATENTO", pose_reason.upper()
        elif is_distracted_face:
            return "DESATENTO", "NO CONCENTRADO"
        else:
            return "ATENTO", "Concentrado"
    
    def draw_ui(self, frame, bbox, state, reason):
        """Dibuja UI en el frame"""
        H, W = frame.shape[:2]
        x, y, w, h = bbox
        
        # Bbox del rostro
        color = (0, 255, 0) if state == "ATENTO" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Overlay semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Texto principal
        label = f"{state}: {reason}"
        cv2.putText(frame, label[:45], (10, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
        
        return frame
    
    def update_stats_ui(self):
        """Actualiza las estadísticas en la UI"""
        total = self.session_stats['total_frames']
        if total == 0:
            return
        
        atento = self.session_stats['atento_frames']
        desatento = self.session_stats['desatento_frames']
        sin_rostro = self.session_stats['sin_rostro_frames']
        
        # Porcentajes
        atento_pct = (atento / total) * 100
        desatento_pct = (desatento / total) * 100
        sin_rostro_pct = (sin_rostro / total) * 100
        
        self.atento_label.config(text=f"{atento_pct:.1f}%")
        self.desatento_label.config(text=f"{desatento_pct:.1f}%")
        self.noface_label.config(text=f"{sin_rostro_pct:.1f}%")
        
        # Objetos detectados
        self.obj_text.config(state=tk.NORMAL)
        self.obj_text.delete(1.0, tk.END)
        
        if self.session_stats['objetos_detectados']:
            for obj, count in self.session_stats['objetos_detectados'].items():
                self.obj_text.insert(tk.END, f"• {obj.capitalize()}: {count}x\n")
        else:
            self.obj_text.insert(tk.END, "Ninguno detectado")
        
        self.obj_text.config(state=tk.DISABLED)
    
    def update_time(self):
        """Actualiza el tiempo de sesión"""
        if self.is_running and self.session_stats['start_time']:
            # Para videos, usar tiempo basado en frames procesados
            if self.video_mode == "file" and self.total_video_frames > 0:
                video_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
                elapsed_seconds = int(self.current_video_frame / video_fps)
                hours, remainder = divmod(elapsed_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                # Para cámara, usar tiempo real
                if not self.is_paused:
                    elapsed = datetime.now() - self.session_stats['start_time']
                    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    return  # No actualizar si está pausado
            
            self.time_label.config(text=time_str)
            self.root.after(1000, self.update_time)
    
    def generate_report(self):
        """Genera reporte PDF de la sesión"""
        # Crear carpeta de reportes
        reports_dir = Path("reportes")
        reports_dir.mkdir(exist_ok=True)
        
        # Nombre del archivo
        timestamp = self.session_stats['start_time'].strftime("%Y%m%d_%H%M%S")
        mode_suffix = "video" if self.video_mode == "file" else "camara"
        pdf_path = reports_dir / f"reporte_atencion_{mode_suffix}_{timestamp}.pdf"
        
        # Crear PDF
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Estilos
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
        
        # Elementos del PDF
        elements = []
        
        # Título
        elements.append(Paragraph("📊 Reporte de Sesión de Atención", title_style))
        elements.append(Spacer(1, 12))
        
        # Información general
        start = self.session_stats['start_time'].strftime("%d/%m/%Y %H:%M:%S")
        end = self.session_stats['end_time'].strftime("%d/%m/%Y %H:%M:%S")
        duration = self.session_stats['end_time'] - self.session_stats['start_time']
        
        info_data = [
            ['Información General', ''],
            ['Modo:', 'Video grabado' if self.video_mode == "file" else 'Cámara en vivo'],
            ['Inicio:', start],
            ['Fin:', end],
            ['Duración:', str(duration).split('.')[0]],
            ['Total de frames:', str(self.session_stats['total_frames'])]
        ]
        
        # Agregar nombre de archivo si es video
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
        
        # Estadísticas de atención
        total = self.session_stats['total_frames']
        atento = self.session_stats['atento_frames']
        desatento = self.session_stats['desatento_frames']
        sin_rostro = self.session_stats['sin_rostro_frames']
        
        atento_pct = (atento / total * 100) if total > 0 else 0
        desatento_pct = (desatento / total * 100) if total > 0 else 0
        sin_rostro_pct = (sin_rostro / total * 100) if total > 0 else 0
        
        elements.append(Paragraph("📈 Estadísticas de Atención", heading_style))
        
        stats_data = [
            ['Estado', 'Frames', 'Porcentaje'],
            ['✅ Atento', str(atento), f'{atento_pct:.1f}%'],
            ['❌ Desatento', str(desatento), f'{desatento_pct:.1f}%'],
            ['⚠️ Sin rostro', str(sin_rostro), f'{sin_rostro_pct:.1f}%']
        ]
        
        stats_table = Table(stats_data, colWidths=[150, 150, 150])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#424242')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (0, 1), colors.lightgreen),
            ('BACKGROUND', (0, 2), (0, 2), colors.lightcoral),
            ('BACKGROUND', (0, 3), (0, 3), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 20))
        
        # Objetos detectados
        if self.session_stats['objetos_detectados']:
            elements.append(Paragraph("📱 Objetos Distractores Detectados", heading_style))
            
            obj_data = [['Objeto', 'Veces Detectado']]
            for obj, count in self.session_stats['objetos_detectados'].items():
                obj_data.append([obj.capitalize(), str(count)])
            
            obj_table = Table(obj_data, colWidths=[250, 200])
            obj_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF5722')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightpink),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(obj_table)
            elements.append(Spacer(1, 20))
        
        # Problemas de pose
        pose_total = sum(self.session_stats['pose_issues'].values())
        if pose_total > 0:
            elements.append(Paragraph("👤 Análisis de Pose de Cabeza", heading_style))
            
            pose_data = [['Dirección', 'Veces Detectado']]
            for direction, count in self.session_stats['pose_issues'].items():
                if count > 0:
                    pose_data.append([f"Mirando {direction}", str(count)])
            
            pose_table = Table(pose_data, colWidths=[250, 200])
            pose_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF9800')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFE0B2')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(pose_table)
            elements.append(Spacer(1, 20))
        
        # Conclusiones
        elements.append(Paragraph("📝 Conclusiones", heading_style))
        
        conclusion = f"""
        Durante esta sesión de {str(duration).split('.')[0]}, el sistema detectó que:
        <br/><br/>
        • Estuviste <b>atento el {atento_pct:.1f}%</b> del tiempo ({atento} frames).
        <br/>
        • Estuviste <b>desatento el {desatento_pct:.1f}%</b> del tiempo ({desatento} frames).
        <br/>
        • El rostro no fue detectado el {sin_rostro_pct:.1f}% del tiempo.
        """
        
        if self.session_stats['objetos_detectados']:
            most_detected = max(self.session_stats['objetos_detectados'].items(), key=lambda x: x[1])
            conclusion += f"<br/><br/>• El objeto más detectado fue: <b>{most_detected[0]}</b> ({most_detected[1]} veces)."
        
        if atento_pct >= 80:
            conclusion += "<br/><br/>✅ <b>¡Excelente nivel de atención!</b>"
        elif atento_pct >= 60:
            conclusion += "<br/><br/>⚠️ <b>Nivel de atención aceptable, pero puede mejorar.</b>"
        else:
            conclusion += "<br/><br/>❌ <b>Nivel de atención bajo. Intenta reducir distracciones.</b>"
        
        elements.append(Paragraph(conclusion, styles['BodyText']))
        elements.append(Spacer(1, 20))
        
        # Footer
        elements.append(Spacer(1, 30))
        footer_text = f"<i>Reporte generado por Sistema de Monitoreo de Atención - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
        elements.append(Paragraph(footer_text, styles['Italic']))
        
        # Construir PDF
        doc.build(elements)
        
        print(f"[INFO] Reporte generado: {pdf_path}")
        return pdf_path

# ========= Main =====
if __name__ == "__main__":
    root = tk.Tk()
    app = AttentionMonitorApp(root)
    root.mainloop()
