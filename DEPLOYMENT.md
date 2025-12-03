# 🚀 Deployment en Streamlit

## 📋 Requisitos previos

1. Cuenta en [Streamlit Cloud](https://streamlit.io/cloud) (gratuita)
2. Repositorio GitHub con tu proyecto

## 🛠️ Instalación local

```powershell
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app_streamlit.py
```

La app se abrirá en `http://localhost:8501`

## ☁️ Deployment en Streamlit Cloud

### Paso 1: Preparar repositorio

Asegúrate de tener estos archivos en tu repo:

```
atencion/
├── app_streamlit.py          # Aplicación principal
├── requirements.txt           # Dependencias
├── modelos/
│   ├── atencion_mnv2_final_mejorado.keras
│   ├── face_detection_yunet_2023mar.onnx
│   ├── yolov8n.pt
│   └── model_config.json
└── reportes/                  # Se creará automáticamente
```

### Paso 2: Subir a GitHub

```powershell
cd c:\Users\franc\Downloads\atencion
git add app_streamlit.py requirements.txt
git commit -m "Add Streamlit deployment"
git push
```

### Paso 3: Deployar en Streamlit Cloud

1. Ve a https://streamlit.io/cloud
2. Click en **"New app"**
3. Conecta tu repo de GitHub: `franco0306/Deteccion-de-atencion-en-tiempo-real`
4. Configura:
   - **Main file path:** `app_streamlit.py`
   - **Python version:** 3.10
5. Click **"Deploy"**

### Paso 4: Configuración opcional

Crea un archivo `.streamlit/config.toml` para personalizar:

```toml
[theme]
primaryColor = "#1976D2"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
```

## ⚠️ Consideraciones importantes

### Limitaciones de Streamlit Cloud (Free Tier)

- **RAM:** 1 GB (puede ser insuficiente para YOLO + TensorFlow)
- **CPU:** Compartido
- **Storage:** 1 GB
- **Tiempo de ejecución:** Apps inactivas se duermen después de 7 días

### Optimizaciones recomendadas

Si encuentras problemas de memoria:

1. **Usar modelo TFLite** (más ligero):
   ```python
   # Convertir a TFLite
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   ```

2. **Desactivar YOLO** temporalmente:
   ```python
   # En app_streamlit.py
   yolo_model = None  # Comentar carga de YOLO
   ```

3. **Reducir resolución de video**:
   ```python
   FRAME_W, FRAME_H = 320, 240  # En lugar de 640x360
   ```

## 🔧 Troubleshooting

### Error: "Memory limit exceeded"

Solución: Usa `opencv-python-headless` en lugar de `opencv-python`:

```txt
opencv-python-headless==4.8.1.78
```

### Error: "Module not found"

Verifica que `requirements.txt` tenga todas las dependencias:

```powershell
pip freeze > requirements.txt
```

### Error: "Model file not found"

Asegúrate de que la carpeta `modelos/` esté en el repo y no en `.gitignore`.

Si los modelos son muy grandes (>100MB), usa **Git LFS**:

```powershell
git lfs install
git lfs track "modelos/*.keras"
git lfs track "modelos/*.pt"
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

## 🌐 Acceso a la app

Una vez deployada, obtendrás una URL como:

```
https://tu-app-nombre.streamlit.app
```

## 📊 Monitoreo

Streamlit Cloud proporciona:
- Logs en tiempo real
- Métricas de uso
- Reinicio automático en caso de errores

## 🔄 Actualizar deployment

Cualquier push a `main` redespliega automáticamente:

```powershell
git add .
git commit -m "Update model"
git push
```

## 💡 Alternativas si Streamlit Cloud no funciona

1. **Hugging Face Spaces** (2 CPU cores, 16GB RAM gratis)
2. **Railway.app** (5$ crédito gratis)
3. **Render.com** (750 horas gratis/mes)

---

## 📞 Soporte

- Documentación: https://docs.streamlit.io/
- Community: https://discuss.streamlit.io/
