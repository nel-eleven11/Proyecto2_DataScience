# dashboard.py
# Aplicación principal de Streamlit para comparar modelos de clasificación
# y detección de mosquitos usando pesos pre-entrenados.

import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import io
import os

from models_classification import (
    CLASES,
    cargar_modelo_clasificacion,
    predecir_imagenes_clasificacion,
    evaluar_csv_clasificacion,
)
from models_detection import (
    cargar_modelo_deteccion,
    predecir_imagenes_deteccion,
    evaluar_csv_deteccion_simple,
)
from utils_metrics import (
    mostrar_resumen_clasificacion,
    graficar_confusion_clasificacion,
    graficar_metricas_deteccion_simple,
)

# Ruta base donde están guardados los modelos
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models"


def configurar_pagina():
    """Configura título y parámetros generales de la app."""
    st.set_page_config(
        page_title="Dashboard modelos mosquitos",
        layout="wide",
    )
    st.title("Dashboard de modelos de mosquitos")
    st.markdown(
        """
        Esta aplicación permite comparar el desempeño de varios modelos de **clasificación**
        y **detección** de especies de mosquitos usando pesos previamente entrenados.
        """
    )


def sidebar_seleccion():
    """Despliega controles de la barra lateral y devuelve selección de vista y modelo."""
    st.sidebar.header("Configuración")

    vista = st.sidebar.radio(
        "Vista",
        ["Modelos de clasificación", "Modelos de detección"],
    )

    if vista == "Modelos de clasificación":
        modelos_disp = [
            "YOLOv8s Clasificación",
            "ConvNeXt Clasificación",
            "ViT Clasificación",
        ]
    else:
        modelos_disp = [
            "YOLOv8s Detección",
            "RetinaNet Detección",
            "Faster R-CNN Detección",
        ]

    modelo_nombre = st.sidebar.selectbox("Modelo", modelos_disp)

    modo_entrada = st.sidebar.radio(
        "Tipo de entrada",
        ["Imágenes sueltas", "CSV (batch)"],
    )

    umbral_score = st.sidebar.slider(
        "Umbral de score / confianza", 0.0, 1.0, 0.5, 0.01
    )

    return vista, modelo_nombre, modo_entrada, umbral_score


# --------------------------- VISTA CLASIFICACIÓN ---------------------------


def vista_clasificacion(modelo_nombre, modo_entrada, umbral_score):
    st.subheader("Modelos de clasificación")

    st.markdown(
        """
        Modelos disponibles:

        - **YOLOv8s Clasificación**: modelo ligero basado en Ultralytics YOLOv8.
        - **ConvNeXt Clasificación**: red convolucional moderna entrenada con timm.
        - **ViT Clasificación**: Vision Transformer (ViT-B/16) entrenado con timm.

        Entrada esperada:
        - Imágenes individuales (JPG/PNG).
        - O un CSV con la estructura:  
          `img_fName,img_w,img_h,bbx_xtl,bbx_ytl,bbx_xbr,bbx_ybr,class_label`
        """
    )

    # Cargar modelo seleccionado
    with st.spinner("Cargando modelo de clasificación..."):
        modelo, dispositivo, tipo_modelo, transform = cargar_modelo_clasificacion(
            modelo_nombre, MODELS_DIR
        )

    if modo_entrada == "Imágenes sueltas":
        subir_imagenes_clasificacion(
            modelo, dispositivo, tipo_modelo, transform, umbral_score
        )
    else:
        subir_csv_clasificacion(
            modelo, dispositivo, tipo_modelo, transform, umbral_score
        )


def subir_imagenes_clasificacion(
    modelo, dispositivo, tipo_modelo, transform, umbral_score
):
    """Permite subir imágenes y ver predicciones de clasificación."""
    uploaded_files = st.file_uploader(
        "Sube una o varias imágenes de mosquitos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Aún no has subido imágenes.")
        return

    imagenes_pil = []
    nombres = []

    for file in uploaded_files:
        imagen = Image.open(file).convert("RGB")
        imagenes_pil.append(imagen)
        nombres.append(file.name)

    with st.spinner("Calculando predicciones..."):
        resultados = predecir_imagenes_clasificacion(
            modelo,
            dispositivo,
            tipo_modelo,
            transform,
            imagenes_pil,
        )

    # Mostrar resultados uno por uno
    for img, nombre, res in zip(imagenes_pil, nombres, resultados):
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(img, caption=nombre, use_column_width=True)
        with col2:
            st.markdown(f"**Archivo:** {nombre}")
            st.markdown(f"**Predicción principal:** {res['top1_clase']}")
            st.markdown(f"**Probabilidad:** {res['top1_prob']:.3f}")
            st.markdown("Top-5:")
            for c, p in zip(res["topk_clases"], res["topk_probs"]):
                st.write(f"- {c}: {p:.3f}")


def subir_csv_clasificacion(
    modelo, dispositivo, tipo_modelo, transform, umbral_score
):
    """Permite subir un CSV y evaluar el modelo de clasificación."""
    csv_file = st.file_uploader(
        "Sube un CSV con las columnas: img_fName,img_w,img_h,bbx_xtl,bbx_ytl,bbx_xbr,bbx_ybr,class_label",
        type=["csv"],
    )
    img_root = st.text_input(
        "Ruta base donde están las imágenes (en el servidor donde corre Streamlit)",
        value="./",
    )

    if csv_file is None:
        st.info("Sube un CSV para evaluar el modelo.")
        return

    df = pd.read_csv(csv_file)

    # Construir rutas completas
    df["img_path"] = df["img_fName"].apply(
        lambda x: str(Path(img_root) / str(x))
    )

    # Validar existencia de archivos
    df_exist = df[df["img_path"].apply(lambda p: Path(p).exists())]
    if df_exist.empty:
        st.error("Ninguna ruta de imagen existe. Revisa la columna img_fName y la ruta base.")
        return

    st.write(f"Total filas en CSV: {len(df)}")
    st.write(f"Imágenes encontradas en disco: {len(df_exist)}")

    with st.spinner("Calculando predicciones y métricas..."):
        df_eval = evaluar_csv_clasificacion(
            modelo,
            dispositivo,
            tipo_modelo,
            transform,
            df_exist,
        )

    st.subheader("Resumen de métricas")

    resumen = mostrar_resumen_clasificacion(df_eval)
    st.dataframe(resumen)

    st.subheader("Matriz de confusión")
    fig_cm = graficar_confusion_clasificacion(df_eval)
    st.pyplot(fig_cm)


# --------------------------- VISTA DETECCIÓN ---------------------------


def vista_deteccion(modelo_nombre, modo_entrada, umbral_score):
    st.subheader("Modelos de detección")

    st.markdown(
        """
        Modelos disponibles:

        - **YOLOv8s Detección**: detector rápido basado en Ultralytics YOLOv8.
        - **RetinaNet Detección**: detector con backbone ResNet50 FPN.
        - **Faster R-CNN Detección**: detector de dos etapas con ResNet50 FPN.

        Entrada esperada:
        - Imágenes individuales (JPG/PNG).
        - O un CSV de anotaciones (mismo formato que para clasificación) para hacer
          una evaluación simple de detección (IOU y coincidencia de clase).
        """
    )

    with st.spinner("Cargando modelo de detección..."):
        modelo, dispositivo, tipo_modelo, transform = cargar_modelo_deteccion(
            modelo_nombre, MODELS_DIR, score_thresh=umbral_score
        )

    if modo_entrada == "Imágenes sueltas":
        subir_imagenes_deteccion(
            modelo, dispositivo, tipo_modelo, transform, umbral_score
        )
    else:
        subir_csv_deteccion(
            modelo, dispositivo, tipo_modelo, transform, umbral_score
        )


def subir_imagenes_deteccion(
    modelo, dispositivo, tipo_modelo, transform, umbral_score
):
    """Permite subir imágenes y ver predicciones de detección."""
    uploaded_files = st.file_uploader(
        "Sube una o varias imágenes de mosquitos para detección",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Aún no has subido imágenes.")
        return

    imagenes_pil = []
    nombres = []

    for file in uploaded_files:
        imagen = Image.open(file).convert("RGB")
        imagenes_pil.append(imagen)
        nombres.append(file.name)

    with st.spinner("Calculando predicciones..."):
        resultados = predecir_imagenes_deteccion(
            modelo,
            dispositivo,
            tipo_modelo,
            transform,
            imagenes_pil,
        )

    for nombre, resultado in zip(nombres, resultados):
        st.markdown(f"### {nombre}")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(resultado["imagen_visualizada"], use_column_width=True)
        with col2:
            st.markdown("Detecciones:")
            if len(resultado["detecciones"]) == 0:
                st.write("Sin detecciones por encima del umbral.")
            else:
                for det in resultado["detecciones"]:
                    st.write(
                        f"- clase: {det['clase']} | score: {det['score']:.3f} | "
                        f"bbox: {det['bbox']}"
                    )


def subir_csv_deteccion(
    modelo, dispositivo, tipo_modelo, transform, umbral_score
):
    """Evalúa el modelo de detección contra un CSV de anotaciones."""
    csv_file = st.file_uploader(
        "Sube un CSV con anotaciones de bounding boxes",
        type=["csv"],
    )
    img_root = st.text_input(
        "Ruta base donde están las imágenes (en el servidor donde corre Streamlit)",
        value="./",
    )

    if csv_file is None:
        st.info("Sube un CSV para evaluar el modelo.")
        return

    df = pd.read_csv(csv_file)
    df["img_path"] = df["img_fName"].apply(
        lambda x: str(Path(img_root) / str(x))
    )
    df_exist = df[df["img_path"].apply(lambda p: Path(p).exists())]

    if df_exist.empty:
        st.error("Ninguna ruta de imagen existe. Revisa la columna img_fName y la ruta base.")
        return

    st.write(f"Total filas en CSV: {len(df)}")
    st.write(f"Imágenes encontradas en disco: {len(df_exist['img_path'].unique())}")

    with st.spinner("Ejecutando evaluación simple de detección..."):
        df_eval = evaluar_csv_deteccion_simple(
            modelo,
            dispositivo,
            tipo_modelo,
            transform,
            df_exist,
            umbral_score=umbral_score,
        )

    st.subheader("Métricas agregadas (evaluación simple)")

    fig = graficar_metricas_det_evaluadas(df_eval=df_eval)
    st.pyplot(fig)


def graficar_metricas_det_evaluadas(df_eval):
    """Pequeño wrapper para reutilizar función y evitar imports duplicados."""
    return graficar_metricas_deteccion_simple(df_eval)


def main():
    configurar_pagina()
    vista, modelo_nombre, modo_entrada, umbral_score = sidebar_seleccion()

    if vista == "Modelos de clasificación":
        vista_clasificacion(modelo_nombre, modo_entrada, umbral_score)
    else:
        vista_deteccion(modelo_nombre, modo_entrada, umbral_score)


if __name__ == "__main__":
    main()
