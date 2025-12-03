# 🎯 Sistema de Detección de Atención - MODO MEJORADO

## 📋 Descripción

**`app_enhanced.py`** es una versión mejorada que detecta desatención usando **3 criterios simultáneos**:

### ✅ 1. **Expresión Facial** (MobileNetV2)
- Modelo entrenado con 74.22% accuracy
- Umbral óptimo: 0.5177 (Youden)

### ✅ 2. **Detección de Objetos Distractores** (YOLOv8n)
Detecta si el estudiante tiene cerca:
- 📱 **Celular**
- 💻 **Laptop** 
- 📖 **Libro** (si no está mirando la cámara)
- 🖱️ **Mouse** (uso excesivo)

### ✅ 3. **Análisis de Pose de Cabeza** (YuNet Landmarks)
Detecta orientación de la cabeza:
- **Yaw** (rotación horizontal): Si gira >25° a los lados → **Desatento**
- **Pitch** (rotación vertical): Si mira >30° arriba/abajo → **Desatento**

---

## 🚀 Uso

### **Ejecutar Modo Mejorado:**
```powershell
python app_enhanced.py
```

### **Ejecutar Modo Básico (solo expresión facial):**
```powershell
python app.py
```

---

## 📊 Comparación de Modos

| Característica | `app.py` (Básico) | `app_enhanced.py` (Mejorado) |
|---------------|-------------------|------------------------------|
| **Expresión facial** | ✅ | ✅ |
| **Detección de objetos** | ❌ | ✅ (celular, laptop, etc.) |
| **Pose de cabeza** | ❌ | ✅ (ángulos yaw/pitch) |
| **FPS** | ~30-40 | ~20-30 (por YOLO) |
| **Precisión** | 74% | **85-90%** (estimado) |
| **Casos de uso** | Expresión básica | Distracción completa |

---

## 🎯 Escenarios Detectados

### ❌ **DESATENTO** cuando:
1. **Expresión facial desatenta** (modelo MNV2 >= 0.5177)
2. **Celular cerca del rostro** (YOLO detecta phone)
3. **Cabeza girada a los lados** (yaw > 25°)
4. **Cabeza mirando hacia abajo** (pitch > 30°, leyendo/escribiendo)
5. **Cabeza mirando hacia arriba** (pitch < -30°, distraído)

### ✅ **ATENTO** cuando:
- Ninguno de los criterios anteriores se cumple
- Mirando directamente a la cámara
- Sin objetos distractores
- Cabeza centrada (-25° < yaw < 25°, -30° < pitch < 30°)

---

## ⚙️ Configuración Avanzada

### **Ajustar Sensibilidad de Pose de Cabeza**

Edita en `app_enhanced.py`:

```python
HEAD_POSE_THRESHOLD = 25  # grados - umbral para "mirando a los lados"
HEAD_DOWN_THRESHOLD = 30  # grados - umbral para "mirando hacia abajo"
```

**Valores recomendados:**
- **Estricto**: 15-20° (detecta movimientos pequeños)
- **Balanceado**: 25-30° (valor actual)
- **Tolerante**: 35-45° (solo movimientos grandes)

### **Ajustar Detección de Objetos**

```python
DISTRACTOR_CLASSES = [67, 63, 73, 64]  # cell phone, laptop, book, mouse
OBJECT_EVERY = 6  # Detectar objetos cada N frames
```

**Para más objetos** (ver clases COCO):
```python
DISTRACTOR_CLASSES = [
    67,  # cell phone
    63,  # laptop
    73,  # book
    64,  # mouse
    66,  # keyboard
    76,  # scissors
    # ... agregar más según necesites
]
```

### **Frecuencias de Procesamiento**

```python
DETECT_EVERY   = 4   # Detectar rostro cada 4 frames
CLASSIFY_EVERY = 2   # Clasificar expresión cada 2 frames
OBJECT_EVERY   = 6   # Detectar objetos cada 6 frames
```

**Para MÁS FPS** (menos precisión):
- Aumentar valores: `DETECT_EVERY = 6`, `OBJECT_EVERY = 10`

**Para MÁS PRECISIÓN** (menos FPS):
- Reducir valores: `DETECT_EVERY = 2`, `OBJECT_EVERY = 4`

---

## 🐛 Solución de Problemas

### **YOLO no se carga / Error al descargar**

Si falla la instalación automática:

```powershell
pip install ultralytics
```

Luego ejecuta en Python:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Auto-descarga
```

### **FPS muy bajo (<15)**

Opciones:
1. Aumentar `OBJECT_EVERY = 10` (detectar objetos menos frecuente)
2. Reducir resolución: `FRAME_W, FRAME_H = 480, 270`
3. Usar solo `app.py` (sin YOLO)

### **Falsos positivos con objetos**

Si detecta objetos que no son distractores:
- Aumentar `search_radius` en `detect_distractors()`:
```python
search_radius = fw * 1.5  # Más estricto (solo muy cerca)
```

---

## 📈 Métricas Esperadas

### **Modo Básico (`app.py`)**
- Test Accuracy: 74.22%
- Test Recall: 80%
- FPS: 30-40

### **Modo Mejorado (`app_enhanced.py`)**
- Accuracy Estimada: **85-90%** (multi-criterio)
- Recall Estimado: **90-95%** (detecta más casos)
- FPS: 20-30 (por procesamiento YOLO)

---

## 🎓 Próximos Pasos

### **Mejoras Futuras:**

1. **Convertir a TensorFlow Lite**
   - Acelerar modelo MNV2: 3-4x más rápido
   - Mantener precisión

2. **Agregar Rastreo de Ojos** (Eye Gaze)
   - Detectar si mira fuera de pantalla
   - Requiere: MediaPipe Face Mesh

3. **Detección de Emociones**
   - Aburrimiento, frustración, confusión
   - Requiere: modelo adicional

4. **Sistema de Alertas**
   - Sonido cuando desatento >5 segundos
   - Log de eventos
   - Reporte semanal

---

## 📝 Notas Técnicas

### **Arquitectura del Sistema:**

```
Frame de cámara
    ↓
┌──────────────────────────────────────┐
│ 1. YuNet Face Detector               │
│    - Detecta rostro + 5 landmarks    │
│    - Cada 4 frames (resto tracking)  │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 2. Análisis de Pose de Cabeza        │
│    - Calcula yaw/pitch con landmarks │
│    - Detecta rotación >25°           │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 3. MobileNetV2 Classifier            │
│    - Expresión facial atento/desatento│
│    - Cada 2 frames + EMA smoothing   │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 4. YOLOv8n Object Detector           │
│    - Detecta celular/laptop/etc.     │
│    - Cada 6 frames (bajo costo)      │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 5. Lógica de Decisión Multi-Criterio │
│    - OR lógico: cualquier criterio   │
│    - → DESATENTO si cumple 1+        │
└──────────────────────────────────────┘
```

### **Optimizaciones Implementadas:**
- ✅ Tracking KCF entre detecciones
- ✅ EMA (Exponential Moving Average) para suavizado
- ✅ Detección en frame reducido (320x180)
- ✅ Procesamiento asíncrono de diferentes módulos
- ✅ Clipping y validación de bboxes

---

¡Listo para usar! 🚀
