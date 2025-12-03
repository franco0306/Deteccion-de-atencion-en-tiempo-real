# app_enhanced.py - Detección mejorada con objetos distractores y pose de cabeza
import cv2, numpy as np, tensorflow as tf, time, json
from pathlib import Path
from urllib.request import urlretrieve

# ========= Config =========
MODEL_PATH = Path("modelos/atencion_mnv2_final_mejorado.keras")
CONFIG_PATH = Path("modelos/model_config.json")
IMG_SIZE = 224
UMBRAL = 0.65  # Umbral más estricto (0.5177 era óptimo, pero muy sensible)

# Cámara
CAM_INDEX   = 0
CAM_BACKEND = 700  # CAP_DSHOW en Windows (valor numérico directo para compatibilidad)
FRAME_W, FRAME_H = 640, 360  # Mantener resolución (buena para visualización)
FOURCC = 'MJPG'

# Detector YuNet
YUNET_ONNX = Path("modelos/face_detection_yunet_2023mar.onnx")
YUNET_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DETECT_W, DETECT_H = 320, 180
SCORE_TH = 0.6
NMS_TH   = 0.3
TOP_K    = 1

# YOLOv8n para detección de objetos distractores
YOLO_MODEL = Path("modelos/yolov8n.pt")
# Clases COCO: cell phone (67), laptop (63), book (73), mouse (64), remote (75), keyboard (66)
DISTRACTOR_CLASSES = [67, 63, 73, 64, 75, 66]  # Más clases para mejor detección
OBJECT_EVERY = 5  # Detectar objetos cada 5 frames (FPS optimizado, memoria compensa)
SHOW_ALL_DETECTIONS = False  # Debug: mostrar todas las detecciones de YOLO en consola

# Análisis de pose de cabeza (usando landmarks de YuNet)
HEAD_POSE_THRESHOLD = 35  # grados - umbral para "mirando a los lados" (balanceado)
HEAD_DOWN_THRESHOLD = 40  # grados - umbral para "mirando hacia abajo" (balanceado)

# Frecuencias (sin tracking para máxima compatibilidad)
DETECT_EVERY   = 3   # Detectar rostro cada 3 frames (mejor FPS)
CLASSIFY_EVERY = 3   # Clasificar cada 3 frames (mejor FPS)
MISS_TOLERANCE = 6   # Aumentado para compensar
SMOOTH_ALPHA_BBOX = 0.7  # Mayor suavizado
SMOOTH_ALPHA_PROB = 0.6

def ensure_yunet():
    if not YUNET_ONNX.exists():
        YUNET_ONNX.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] Descargando YuNet ONNX...")
        urlretrieve(YUNET_URL, YUNET_ONNX.as_posix())
        print("[OK] YuNet descargado:", YUNET_ONNX)

def ensure_yolo():
    """Descarga YOLOv8n si no existe"""
    if not YOLO_MODEL.exists():
        try:
            print("[INFO] Instalando ultralytics y descargando YOLOv8n...")
            import subprocess
            subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # Auto-descarga
            YOLO_MODEL.parent.mkdir(parents=True, exist_ok=True)
            model.export(format='onnx')  # Opcional: exportar a ONNX para rapidez
            print("[OK] YOLOv8n instalado")
            return model
        except Exception as e:
            print(f"[WARNING] No se pudo instalar YOLO: {e}")
            print("[INFO] Continuando sin detección de objetos...")
            return None
    else:
        from ultralytics import YOLO
        return YOLO(YOLO_MODEL)

def estimate_head_pose(landmarks):
    """
    Estima orientación de cabeza usando landmarks faciales de YuNet
    landmarks: array de shape (5, 2) -> [ojo_der, ojo_izq, nariz, boca_der, boca_izq]
    Retorna: (yaw, pitch) en grados aproximados CALIBRADOS
    """
    if landmarks is None or len(landmarks) < 5:
        return 0, 0
    
    # Landmarks: [right_eye, left_eye, nose, right_mouth, left_mouth]
    right_eye = landmarks[0]
    left_eye = landmarks[1]
    nose = landmarks[2]
    right_mouth = landmarks[3]
    left_mouth = landmarks[4]
    
    # Calcular referencias
    eye_distance = np.linalg.norm(right_eye - left_eye)
    if eye_distance < 10:  # Evitar divisiones por cero
        return 0, 0
    
    eye_center = (right_eye + left_eye) / 2
    mouth_center = (right_mouth + left_mouth) / 2
    
    # YAW (rotación horizontal - cabeza a los lados)
    # Usar nariz respecto al centro de los ojos
    nose_offset_x = nose[0] - eye_center[0]
    # Normalizar y convertir a grados (calibrado para YuNet)
    yaw = (nose_offset_x / eye_distance) * 50  # Factor reducido (era 90)
    
    # PITCH (rotación vertical - cabeza arriba/abajo)
    # Usar distancia vertical nariz-ojos vs nariz-boca
    eye_to_nose = nose[1] - eye_center[1]
    nose_to_mouth = mouth_center[1] - nose[1]
    
    # Validar que las distancias sean razonables
    if nose_to_mouth > 5 and abs(eye_to_nose) > 0:  # Mínimo 5px para evitar ruido
        ratio = eye_to_nose / nose_to_mouth
        # Calibrado MÁS CONSERVADOR: valor normal ~0.8-1.2
        # Solo alerta si ratio es MUY diferente
        if ratio > 1.3:
            pitch = (ratio - 1.3) * 50  # Mirando abajo (positivo)
        elif ratio < 0.6:
            pitch = (ratio - 0.6) * 35  # Mirando arriba (negativo)
        else:
            pitch = 0  # Rango normal, no alerta
    else:
        pitch = 0
    
    # Limitar valores extremos (evitar falsos positivos)
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
    """
    Detecta objetos distractores (celular, laptop, libro) cerca del rostro
    Retorna: (tiene_distractor, lista_objetos_detectados)
    """
    if yolo_model is None:
        return False, []
    
    try:
        # Inferencia YOLO con confidence balanceado (reduce falsos positivos)
        results = yolo_model(frame, verbose=False, conf=0.30, iou=0.45, imgsz=640, max_det=15)
        
        if len(results) == 0 or results[0].boxes is None:
            return False, []
        
        boxes = results[0].boxes
        detected_objects = []
        all_objects = []  # Para debug
        
        fx, fy, fw, fh = face_bbox
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2
        
        # Área de búsqueda MUY AMPLIADA: 4.5x el ancho de la cara (casi toda la pantalla)
        search_radius = fw * 4.5
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Guardar TODAS las detecciones para debug
            if SHOW_ALL_DETECTIONS:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                all_objects.append(f"cls{cls}:{conf:.0%}")
            
            # Solo objetos distractores
            if cls not in DISTRACTOR_CLASSES:
                continue
            
            # Obtener bbox del objeto
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2
            
            # Calcular distancia al rostro
            dist = np.sqrt((obj_center_x - face_center_x)**2 + (obj_center_y - face_center_y)**2)
            
            # Si está cerca del rostro (área ampliada)
            if dist < search_radius:
                obj_names = {
                    67: "celular", 63: "laptop", 73: "libro", 
                    64: "mouse", 75: "control", 66: "teclado"
                }
                obj_name = obj_names.get(cls, f"objeto-{cls}")
                detected_objects.append({
                    'name': obj_name,
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    'conf': float(box.conf[0]),
                    'class': cls
                })
        
        # Debug info
        if SHOW_ALL_DETECTIONS and len(all_objects) > 0:
            print(f"[DEBUG] YOLO detectó: {', '.join(all_objects)}")
        
        return len(detected_objects) > 0, detected_objects
        
    except Exception as e:
        print(f"[WARNING] Error en YOLO: {e}")
        return False, []

# ========= Carga modelo y configuración =========
print(f"[INFO] Cargando modelo de atención: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        UMBRAL = config.get('optimal_threshold', UMBRAL)
        print(f"[INFO] Umbral óptimo: {UMBRAL:.4f}")
        print(f"[INFO] Test Accuracy: {config['metrics']['test_accuracy']:.2%}")
        print(f"[INFO] Test AUC: {config['metrics']['test_auc']:.2%}")
else:
    print(f"[INFO] Usando umbral por defecto: {UMBRAL:.4f}")

# ========= YOLO para objetos =========
print("[INFO] Cargando YOLOv8n para detección de objetos...")
yolo_model = ensure_yolo()
if yolo_model:
    print("[OK] YOLO cargado - detectará: celular, laptop, libro, mouse")
else:
    print("[WARNING] YOLO no disponible - solo usará modelo de atención")

# ========= YuNet =========
ensure_yunet()
detector = cv2.FaceDetectorYN.create(
    model=YUNET_ONNX.as_posix(),
    config="",
    input_size=(DETECT_W, DETECT_H),
    score_threshold=SCORE_TH,
    nms_threshold=NMS_TH,
    top_k=TOP_K
)

# ========= Cámara =========
cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAM_INDEX, 0)  # 0 = CAP_ANY (genérico)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)

# ========= Variables de estado (SIN TRACKING) =========
prev_bbox = None
last_ok_bbox = None
miss_counter = 0
prob_ema = None
frame_id = 0
last_landmarks = None

# Suavizado de ángulos (EMA para estabilidad)
yaw_ema = None
pitch_ema = None
ANGLE_ALPHA = 0.25  # Suavizado MUY agresivo (más estable, menos reactivo)

# Contador de frames consecutivos para evitar falsos positivos
pose_distracted_count = 0
POSE_CONFIRMATION_FRAMES = 3  # Requiere 3 frames consecutivos (balanceado)

# Sistema de memoria para objetos detectados (persistencia entre frames)
last_detected_objects = []
object_memory_frames = 0
OBJECT_MEMORY_DURATION = 15  # Mantener detección por 15 frames (~2 segundos)

print("\n" + "="*60)
print("🎯 MODO MEJORADO ACTIVADO:")
print("  ✅ Detección de expresión facial (MobileNetV2)")
print("  ✅ Detección de objetos distractores (YOLOv8n)")
print("  ✅ Análisis de pose de cabeza (YuNet landmarks)")
print("  ⚡ Sin tracking - Máxima compatibilidad")
print("="*60)
print("[INFO] Presiona 'q' para salir.\n")

t0, n, fps_val = time.time(), 0, 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    H, W = frame.shape[:2]

    # --- Detección de rostro (cada N frames) ---
    bbox = None
    landmarks = None
    
    if frame_id % DETECT_EVERY == 0:
        small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
        detector.setInputSize((DETECT_W, DETECT_H))
        res = detector.detect(small)
        faces = res[1] if isinstance(res, tuple) else res

        if isinstance(faces, np.ndarray) and faces.size > 0:
            face_data = faces[0]
            x, y, w, h = face_data[:4].astype(int)
            
            # Re-escalar bbox
            scale_x = W / DETECT_W
            scale_y = H / DETECT_H
            x = int(x * scale_x); y = int(y * scale_y)
            w = int(w * scale_x); h = int(h * scale_y)
            bbox = [x, y, w, h]
            
            # Extraer landmarks (5 puntos: ojos, nariz, boca)
            if len(face_data) >= 15:
                lm = face_data[4:14].reshape(5, 2)
                lm[:, 0] *= scale_x
                lm[:, 1] *= scale_y
                landmarks = lm
                last_landmarks = landmarks
            
            # Guardar para histéresis
            last_ok_bbox = bbox
            miss_counter = 0
    
    # --- Histéresis: si no detectamos, usar último bbox válido brevemente ---
    if bbox is None and last_ok_bbox is not None and miss_counter < MISS_TOLERANCE:
        miss_counter += 1
        bbox = last_ok_bbox
        landmarks = last_landmarks

    # --- Clasificación multi-criterio ---
    label_text = "Sin rostro"
    color = (200, 200, 200)
    reasons = []  # Razones de "desatento"

    do_classify = (frame_id % CLASSIFY_EVERY == 0)
    do_object_detect = (frame_id % OBJECT_EVERY == 0)
    
    if bbox is not None:
        bbox = clip_bbox(bbox, W, H)
        bbox = smooth_bbox(prev_bbox, bbox)
        prev_bbox = bbox
        x, y, w, h = bbox
        
        # Dibujar bbox del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 255, 180), 2)
        
        # 1. CLASIFICACIÓN FACIAL
        if do_classify:
            prob = classify_face(frame, bbox, model)
            prob_ema = prob if prob_ema is None else (SMOOTH_ALPHA_PROB*prob_ema + (1-SMOOTH_ALPHA_PROB)*prob)
        elif prob_ema is None:
            prob = classify_face(frame, bbox, model)
            prob_ema = prob
        
        is_distracted_face = prob_ema >= UMBRAL if prob_ema is not None else False
        if is_distracted_face:
            reasons.append(f"Expresión ({prob_ema:.2f})")
        
        # 2. ANÁLISIS DE POSE DE CABEZA (con suavizado EMA)
        if landmarks is not None:
            yaw, pitch = estimate_head_pose(landmarks)
            
            # Aplicar suavizado exponencial (EMA) para estabilidad
            if yaw_ema is None or pitch_ema is None:
                yaw_ema = yaw
                pitch_ema = pitch
            else:
                yaw_ema = ANGLE_ALPHA * yaw + (1 - ANGLE_ALPHA) * yaw_ema
                pitch_ema = ANGLE_ALPHA * pitch + (1 - ANGLE_ALPHA) * pitch_ema
            
            # Dibujar landmarks (opcional - comentar para limpiar UI)
            # for lm in landmarks:
            #     cv2.circle(frame, tuple(lm.astype(int)), 2, (0, 255, 255), -1)
            
            # Verificar pose con confirmación (evitar falsos positivos)
            pose_is_bad = False
            
            if abs(yaw_ema) > HEAD_POSE_THRESHOLD:
                pose_is_bad = True
                direction_h = "derecha" if yaw_ema > 0 else "izquierda"
                if pose_distracted_count >= POSE_CONFIRMATION_FRAMES:
                    reasons.append(f"Mirando a {direction_h}")
            
            if abs(pitch_ema) > HEAD_DOWN_THRESHOLD:
                pose_is_bad = True
                direction_v = "abajo" if pitch_ema > 0 else "arriba"
                if pose_distracted_count >= POSE_CONFIRMATION_FRAMES:
                    reasons.append(f"Mirando {direction_v}")
            
            # Actualizar contador de confirmación
            if pose_is_bad:
                pose_distracted_count = min(pose_distracted_count + 1, POSE_CONFIRMATION_FRAMES + 1)
            else:
                pose_distracted_count = max(0, pose_distracted_count - 1)
            
            # Mostrar ángulos suavizados
            cv2.putText(frame, f"Yaw: {yaw_ema:.1f}deg", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(frame, f"Pitch: {pitch_ema:.1f}deg", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # 3. DETECCIÓN DE OBJETOS DISTRACTORES CON MEMORIA
        has_distractor = False
        detected_objs = []
        
        if yolo_model and do_object_detect:
            has_distractor, detected_objs = detect_distractors(frame, yolo_model, bbox)
            
            if has_distractor:
                # Actualizar memoria de objetos
                last_detected_objects = detected_objs.copy()
                object_memory_frames = OBJECT_MEMORY_DURATION
                
                obj_names = [obj['name'] for obj in detected_objs]
                reasons.append(f"Objeto: {', '.join(obj_names)}")
                
                # Dibujar bboxes de objetos MUY VISIBLES
                for obj in detected_objs:
                    ox, oy, ow, oh = obj['bbox']
                    # Bbox naranja grueso con efecto glow
                    cv2.rectangle(frame, (ox-2, oy-2), (ox+ow+2, oy+oh+2), (0, 140, 255), 5)
                    cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (0, 200, 255), 2)
                    
                    # Etiqueta con fondo
                    label = f"{obj['name'].upper()}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                    cv2.rectangle(frame, (ox, oy-th-8), (ox+tw+8, oy), (0, 165, 255), -1)
                    cv2.putText(frame, label, (ox+4, oy-4),
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        # Usar memoria de objetos si no se detectó nada nuevo pero hay memoria reciente
        if not has_distractor and object_memory_frames > 0:
            object_memory_frames -= 1
            if len(last_detected_objects) > 0:
                has_distractor = True
                obj_names = [obj['name'] for obj in last_detected_objects]
                reasons.append(f"Objeto: {', '.join(obj_names)}")
                
                # Dibujar objetos de memoria con borde amarillo (memoria)
                for obj in last_detected_objects:
                    ox, oy, ow, oh = obj['bbox']
                    cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (0, 255, 255), 2)
                    label = f"{obj['name'].upper()} (MEM)"
                    cv2.putText(frame, label, (ox, oy-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # DECISIÓN FINAL - Mensajes claros y priorizados
        has_object = any("Objeto:" in r for r in reasons)
        has_pose_issue = any("Mirando" in r for r in reasons)
        
        if has_object:
            # PRIORIDAD 1: Objeto detectado → Mensaje MÁS CLARO
            obj_reason = [r for r in reasons if "Objeto:" in r][0]
            main_reason = obj_reason.replace("Objeto: ", "").upper()
            label_text = f"DESATENTO: {main_reason}"
            color = (0, 0, 255)
            
        elif has_pose_issue:
            # PRIORIDAD 2: Pose mala → Mostrar dirección (sin "expresión")
            if "izquierda" in str(reasons):
                label_text = "DESATENTO: MIRANDO IZQUIERDA"
            elif "derecha" in str(reasons):
                label_text = "DESATENTO: MIRANDO DERECHA"
            elif "abajo" in str(reasons):
                label_text = "DESATENTO: MIRANDO ABAJO"
            elif "arriba" in str(reasons):
                label_text = "DESATENTO: MIRANDO ARRIBA"
            else:
                label_text = "DESATENTO: POSE INCORRECTA"
            color = (0, 0, 255)
            
        elif is_distracted_face:
            # PRIORIDAD 3: Solo expresión (sin pose ni objeto)
            label_text = "DESATENTO: NO CONCENTRADO"
            color = (0, 0, 255)
            
        else:
            # Todo normal
            if prob_ema is not None:
                label_text = f"ATENTO ({prob_ema:.2f})"
            else:
                label_text = "ATENTO"
            color = (0, 255, 0)

    # --- UI MEJORADA ---
    # Fondo semitransparente para mejor legibilidad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Texto principal más grande y legible
    cv2.putText(frame, label_text[:45], (10, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
    
    # Indicador de escaneo de objetos
    if do_object_detect and yolo_model:
        cv2.circle(frame, (W-25, 25), 8, (0, 255, 255), -1)  # Punto amarillo
        cv2.putText(frame, "SCAN", (W-70, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Memoria de objetos activa
    if object_memory_frames > 0:
        cv2.putText(frame, f"Memoria: {object_memory_frames}f", (W-150, H-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    n += 1
    if time.time() - t0 >= 0.5:
        fps_val = n / (time.time() - t0)
        t0 = time.time(); n = 0
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Atencion MEJORADA - Multi-criterio", frame)
    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
