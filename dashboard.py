# dashboard.py
# Streamlit dashboard para comparar modelos de clasificación y detección
# usando métricas precomputadas + inferencia en tiempo real.

import io
import os
import shutil
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)

# ===========================
# MAPA DE MODELOS → CSV FILTRO

# ===========================
# MODELOS → EVALUACIÓN
# ===========================


# ===========================
MODEL_EVAL_MAP = {
    # CLASSIFICATION
    "YOLOv8s Clasificación": ("classification", "YOLO"),
    "ConvNeXt Clasificación": ("classification", "ConvNeXt"),
    "ViT Clasificación": ("classification", "ViT"),
    # DETECTION
    "YOLOv8s Detección": ("detection", "YOLO"),
    "RetinaNet Detección": ("detection", "RetinaNet"),
    "Faster R-CNN Detección": ("detection", "FRCNN"),
}

# ===========================
# PIPELINES COMPLETOS
# ===========================
PIPELINE_EVAL_MAP = {
    "YOLO → YOLO-CLS": ("full", ("YOLO", "YOLO-CLS")),
    "RetinaNet → ConvNeXt": ("full", ("RetinaNet", "ConvNeXt")),
    "RetinaNet → ViT": ("full", ("RetinaNet", "ViT")),
    "FRCNN → ConvNeXt": ("full", ("FRCNN", "ConvNeXt")),
    "FRCNN → ViT": ("full", ("FRCNN", "ViT")),
}

MODEL_EVAL_MAP.update(PIPELINE_EVAL_MAP)

# ===========================
# TRAINING TIMES (in minutes)
# ===========================
# Update these values with actual training times from your experiments
TRAINING_TIMES = {
    # Classification models
    "YOLOv8s Clasificación": 120,  # Example: 2 hours
    "ConvNeXt Clasificación": 180,  # Example: 3 hours
    "ViT Clasificación": 200,  # Example: 3.33 hours
    # Detection models
    "YOLOv8s Detección": 150,  # Example: 2.5 hours
    "RetinaNet Detección": 453,  # ~7.55 hours (from notebook duration)
    "Faster R-CNN Detección": 400,  # Example: ~6.67 hours
    # Full pipelines (sum of detection + classification training times)
    "YOLO → YOLO-CLS": 270,  # YOLO detection + YOLO classification
    "RetinaNet → ConvNeXt": 633,  # RetinaNet + ConvNeXt
    "RetinaNet → ViT": 653,  # RetinaNet + ViT
    "FRCNN → ConvNeXt": 580,  # FRCNN + ConvNeXt
    "FRCNN → ViT": 600,  # FRCNN + ViT
}

# ===========================
# IMPORTS DE MODELOS
# ===========================
from models_classification import (CLASES, cargar_modelo_clasificacion,
                                   evaluar_csv_clasificacion,
                                   predecir_imagenes_clasificacion)
from models_detection import (cargar_modelo_deteccion,
                              evaluar_csv_deteccion_simple,
                              predecir_imagenes_deteccion)
from utils_metrics import (graficar_confusion_clasificacion,
                           graficar_metricas_deteccion_simple,
                           mostrar_resumen_clasificacion)

# ===========================
# PATHS BASE
# ===========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models"
IMAGES_DIR = BASE_DIR / "images"
EVALS_DIR = BASE_DIR / "evals"


# ===========================
# LOAD EVAL CSVs
# ===========================
@st.cache_data
def load_eval_results():
    files = {
        "classification": EVALS_DIR / "classification_results.csv",
        "detection": EVALS_DIR / "detection_results.csv",
        "full": EVALS_DIR / "full_eval_results.csv",
    }
    dfs = {}
    for key, path in files.items():
        dfs[key] = pd.read_csv(path) if path.exists() else None
    return dfs


# ===========================
# BOOTSTRAP DATASET
# ===========================
@st.cache_resource
def bootstrap_dataset() -> str:
    IMAGES_DIR.mkdir(exist_ok=True)

    if any(IMAGES_DIR.iterdir()):
        return str(IMAGES_DIR)

    st.info("Descargando dataset de Kaggle (primer uso)...")

    #dataset_path = kagglehub.dataset_download("vishakkbhat/mosquito-data")
    dataset_path = Path("dataset")
    root = Path(dataset_path)

    image_files = (
        list(root.rglob("*.jpg"))
        + list(root.rglob("*.jpeg"))
        + list(root.rglob("*.png"))
    )

    st.info(f"Copiando {len(image_files)} imágenes...")

    for img in image_files:
        dst = IMAGES_DIR / img.name
        if not dst.exists():
            shutil.copy(img, dst)

    return str(IMAGES_DIR)


def pick_random_image_by_class(class_name: str) -> str:
    """
    Returns a random image path belonging to the given class.

    Uses:
        - CSV: data_csv/test.csv
        - Images: images/ (IMAGES_DIR)

    Returns:
        str: absolute path to image OR None
    """
    csv_path = BASE_DIR / "data_csv" / "test.csv"
    imgs_dir = IMAGES_DIR  # already defined in your dashboard

    if not csv_path.exists():
        st.error("❌ No existe data_csv/test.csv")
        return None

    df = pd.read_csv(csv_path)

    if "class_label" not in df.columns or "img_fName" not in df.columns:
        st.error("❌ test.csv debe tener columnas 'class_label' y 'img_fName'")
        return None

    df_class = df[df["class_label"] == class_name]

    if df_class.empty:
        st.warning(f"No hay imágenes para la clase '{class_name}'.")
        return None

    # Pick random row
    row = df_class.sample(1).iloc[0]

    img_path = imgs_dir / row["img_fName"]

    # If missing, warn and skip
    if not img_path.exists():
        st.warning(f"La imagen '{img_path}' no existe en images/.")
        return None

    return str(img_path)


def subir_imagenes_clasificacion(modelo, device, tipo, transform):
    files = st.file_uploader(
        "Sube imágenes", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    # 🔥 RANDOM PICKER GOES HERE
    with st.expander("Elegir imagen aleatoria por clase"):
        clase = st.selectbox("Clase", CLASES, key="rand_classif")
        if st.button("Seleccionar aleatoria", key="btn_rand_classif"):
            rand_path = pick_random_image_by_class(clase)
            if rand_path:
                st.success(f"Imagen elegida: {rand_path}")
                files = [open(rand_path, "rb")]

    if not files:
        return

    imgs = [Image.open(f).convert("RGB") for f in files]

    with st.spinner("Prediciendo..."):
        res = predecir_imagenes_clasificacion(modelo, device, tipo, transform, imgs)

    for img, r in zip(imgs, res):
        st.image(img, width=300)
        st.write(f"Predicción: **{r['top1_clase']}** — {r['top1_prob']:.3f}")
        st.write("Top-5:")
        for c, p in zip(r["topk_clases"], r["topk_probs"]):
            st.write(f"- {c}: {p:.3f}")


def subir_imagenes_deteccion(modelo, dev, tipo, transform, umbral):
    files = st.file_uploader(
        "Sube imágenes", type=["jpg", "png"], accept_multiple_files=True
    )

    # 🔥 RANDOM PICKER GOES HERE
    with st.expander("Elegir imagen aleatoria por clase"):
        clase = st.selectbox("Clase", CLASES, key="rand_det")
        if st.button("Seleccionar aleatoria", key="btn_rand_det"):
            rand_path = pick_random_image_by_class(clase)
            if rand_path:
                st.success(f"Imagen elegida: {rand_path}")
                files = [open(rand_path, "rb")]

    if not files:
        return

    imgs = [Image.open(f).convert("RGB") for f in files]

    with st.spinner("Detectando..."):
        res = predecir_imagenes_deteccion(modelo, dev, tipo, transform, imgs, umbral)

    for img_name, out in zip([f.name for f in files], res):
        st.markdown(f"### {img_name}")
        st.image(out["imagen_visualizada"], use_column_width=True)


def subir_imagenes_pipeline(
    det_model, det_dev, det_tipo, det_tf, cls_model, cls_dev, cls_tipo, cls_tf
):
    files = st.file_uploader(
        "Sube imágenes", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    # 🔥 RANDOM PICKER GOES HERE
    with st.expander("Elegir imagen aleatoria por clase"):
        clase = st.selectbox("Clase", CLASES, key="rand_pipe")
        if st.button("Seleccionar aleatoria", key="btn_rand_pipe"):
            rand_path = pick_random_image_by_class(clase)
            if rand_path:
                st.success(f"Imagen elegida: {rand_path}")
                files = [open(rand_path, "rb")]

    if not files:
        return

    imgs = [Image.open(f).convert("RGB") for f in files]

    for img, name in zip(imgs, [f.name for f in files]):
        det = predecir_imagenes_deteccion(
            det_model, det_dev, det_tipo, det_tf, [img], 0.5
        )[0]

        if len(det["detecciones"]) == 0:
            st.write(f"❌ {name}: No se detectó mosquito.")
            continue

        # best detection
        det_box = det["detecciones"][0]["bbox"]
        x1, y1, x2, y2 = map(int, det_box)
        crop = img.crop((x1, y1, x2, y2))

        cls_res = predecir_imagenes_clasificacion(
            cls_model, cls_dev, cls_tipo, cls_tf, [crop]
        )[0]

        st.markdown(f"### {name}")
        st.image(det["imagen_visualizada"])
        st.image(
            crop,
            caption=f"{cls_res['top1_clase']} ({cls_res['top1_prob']:.3f})",
            width=300,
        )


# ============================================================
# MÉTRICAS (classification / detection / full pipelines)
# ============================================================
def mostrar_metricas_modelo(modelo_nombre, dfs):
    eval_type, model_key = MODEL_EVAL_MAP.get(modelo_nombre, (None, None))
    if eval_type is None:
        return

    df_class = dfs.get("classification")
    df_det = dfs.get("detection")
    df_full = dfs.get("full")

    # ============================================================
    # CLASSIFICATION
    # ============================================================
    if eval_type == "classification":
        if df_class is None:
            st.warning("⚠️ No existe classification_results.csv")
            return

        pred_col_map = {
            "YOLO": "yolo_pred",
            "ConvNeXt": "convnext_pred",
            "ViT": "vit_pred",
        }
        col = pred_col_map[model_key]

        if col not in df_class.columns:
            st.error(f"El CSV no contiene '{col}'")
            return

        st.subheader(f"Métricas — {modelo_nombre}")

        y_true = df_class["true_label"].astype(str).values
        y_pred = df_class[col].astype(str).values

        acc = (y_true == y_pred).mean()

        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASES, average="macro", zero_division=0
        )
        prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASES, average="weighted", zero_division=0
        )

        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{acc*100:.2f}%")
        cols[1].metric("Macro F1", f"{f1_macro:.3f}")
        cols[2].metric("Weighted F1", f"{f1_weight:.3f}")
        cols[3].metric("Macro Precision", f"{prec_macro:.3f}")
        cols[4].metric("Macro Recall", f"{rec_macro:.3f}")

        st.markdown("---")

        df_temp = pd.DataFrame({"true_label": y_true, "pred_label": y_pred})
        fig_cm = graficar_confusion_clasificacion(df_temp)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        st.subheader("Métricas por Clase")

        report = classification_report(
            y_true, y_pred, labels=CLASES, output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(report).T.loc[CLASES]
        df_report = df_report.rename(
            columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1",
                "support": "Muestras",
            }
        )

        st.dataframe(
            df_report.style.format(
                {
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                    "Muestras": "{:.0f}",
                }
            )
        )
        return

    # ============================================================
    # DETECTION
    # ============================================================
    if eval_type == "detection":
        if df_det is None:
            st.warning("⚠ No existe detection_results.csv")
            return

        df_model = df_det[df_det["det_model"] == model_key].copy()

        if df_model.empty:
            st.warning("No hay datos para este modelo.")
            return

        st.subheader(f"Métricas — {modelo_nombre}")

        mean_iou = df_model["iou"].mean()
        det_acc = df_model["match"].mean()

        cols = st.columns(3)
        cols[0].metric("Accuracy (IoU≥0.5)", f"{det_acc*100:.2f}%")
        cols[1].metric("IoU Promedio", f"{mean_iou:.3f}")
        cols[2].metric("Muestras", f"{len(df_model)}")

        st.markdown("---")

        fig = px.histogram(
            df_model,
            x="iou",
            nbins=25,
            title="Distribución IoU",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Por Clase")

        per_class_acc = (
            df_model.groupby("true_label")["match"].mean().reindex(CLASES).fillna(0)
        )
        per_class_iou = (
            df_model.groupby("true_label")["iou"].mean().reindex(CLASES).fillna(0)
        )

        df_table = pd.DataFrame(
            {
                "Clase": CLASES,
                "Accuracy": per_class_acc.values,
                "IoU": per_class_iou.values,
                "Muestras": df_model["true_label"]
                .value_counts()
                .reindex(CLASES)
                .fillna(0)
                .astype(int)
                .values,
            }
        )

        st.dataframe(df_table.style.format({"Accuracy": "{:.3f}", "IoU": "{:.3f}"}))

        return

    # ============================================================
    # FULL PIPELINES (DET → CLS)
    # ============================================================
    if eval_type == "full":
        if df_full is None:
            st.warning("⚠ No existe full_eval_results.csv")
            return

        det_key, cls_key = model_key
        df_model = df_full[
            (df_full["det_model"] == det_key) & (df_full["cls_model"] == cls_key)
        ].copy()

        if df_model.empty:
            st.warning("No hay datos para este pipeline.")
            return

        st.subheader(f"Métricas Pipeline — {modelo_nombre}")

        mean_iou = df_model["iou"].mean()
        acc = df_model["match"].mean()

        cols = st.columns(3)
        cols[0].metric("Accuracy Pipeline", f"{acc*100:.2f}%")
        cols[1].metric("IoU Promedio", f"{mean_iou:.3f}")
        cols[2].metric("Muestras", f"{len(df_model)}")

        st.markdown("---")

        df_temp = df_model[["true_label", "pred_label"]]
        fig_cm = graficar_confusion_clasificacion(df_temp)
        st.plotly_chart(fig_cm, use_container_width=True)

        per_class_acc = (
            df_model.groupby("true_label")["match"].mean().reindex(CLASES).fillna(0)
        )
        per_class_iou = (
            df_model.groupby("true_label")["iou"].mean().reindex(CLASES).fillna(0)
        )

        df_table = pd.DataFrame(
            {
                "Clase": CLASES,
                "Accuracy": per_class_acc.values,
                "IoU": per_class_iou.values,
                "Muestras": df_model["true_label"]
                .value_counts()
                .reindex(CLASES)
                .fillna(0)
                .astype(int),
            }
        )

        st.markdown("---")
        st.subheader("Por Clase")
        st.dataframe(df_table.style.format({"Accuracy": "{:.3f}", "IoU": "{:.3f}"}))
        return


# ============================================================
# CONFIGURACIÓN
# ============================================================
def configurar_pagina():
    st.set_page_config(page_title="Dashboard Mosquitos", layout="wide")
    st.title("Dashboard de modelos de Mosquitos")
    st.markdown(
        """
    Compara modelos de **clasificación**, **detección** y **pipelines completos**.
    """
    )


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================
def comparar_modelos_clasificacion(dfs):
    """Compara modelos de clasificación usando F1-macro, accuracy y recall."""
    df_class = dfs.get("classification")
    
    if df_class is None:
        st.warning("⚠️ No existe classification_results.csv")
        return
    
    st.subheader("Comparación de Modelos de Clasificación")
    
    modelos = {
        "YOLOv8s Clasificación": "yolo_pred",
        "ConvNeXt Clasificación": "convnext_pred",
        "ViT Clasificación": "vit_pred",
    }
    
    resultados = []
    y_true = df_class["true_label"].astype(str).values
    
    for nombre_modelo, col_pred in modelos.items():
        if col_pred not in df_class.columns:
            continue
        
        y_pred = df_class[col_pred].astype(str).values
        
        # Calcular métricas
        acc = (y_true == y_pred).mean()
        _, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASES, average="macro", zero_division=0
        )
        
        training_time = TRAINING_TIMES.get(nombre_modelo, None)
        resultados.append({
            "Modelo": nombre_modelo,
            "F1-Macro": f1_macro,
            "Accuracy": acc,
            "Recall": rec_macro,
            "Tiempo Entrenamiento (min)": training_time,
        })
    
    if not resultados:
        st.error("No hay datos disponibles para comparar.")
        return
    
    df_comparacion = pd.DataFrame(resultados)
    
    # Mostrar tabla comparativa
    format_dict = {
        "F1-Macro": "{:.4f}",
        "Accuracy": "{:.4f}",
        "Recall": "{:.4f}",
    }
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        format_dict["Tiempo Entrenamiento (min)"] = "{:.1f}"
    
    highlight_subset = ["F1-Macro", "Accuracy", "Recall"]
    style_obj = df_comparacion.style.format(format_dict).highlight_max(axis=0, subset=highlight_subset)
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        style_obj = style_obj.highlight_min(axis=0, subset=["Tiempo Entrenamiento (min)"])
    
    st.dataframe(style_obj, use_container_width=True)
    
    # Gráfico de barras comparativo
    st.markdown("---")
    st.subheader("Visualización Comparativa")
    
    df_melted = df_comparacion.melt(
        id_vars=["Modelo"],
        value_vars=["F1-Macro", "Accuracy", "Recall"],
        var_name="Métrica",
        value_name="Valor"
    )
    
    fig = px.bar(
        df_melted,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        title="Comparación de Métricas de Clasificación",
        labels={"Valor": "Valor de la Métrica", "Modelo": "Modelo"},
        color_discrete_map={
            "F1-Macro": "#1f77b4",
            "Accuracy": "#ff7f0e",
            "Recall": "#2ca02c"
        },
        text="Valor"
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, 1.05]),
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap de comparación
    st.markdown("---")
    st.subheader("Mapa de Calor Comparativo")
    
    df_heatmap = df_comparacion.set_index("Modelo")[["F1-Macro", "Accuracy", "Recall"]]
    fig_heatmap = px.imshow(
        df_heatmap.T,
        labels=dict(x="Modelo", y="Métrica", color="Valor"),
        color_continuous_scale="Blues",
        text_auto=".3f",
        aspect="auto"
    )
    fig_heatmap.update_layout(
        title="Mapa de Calor: Comparación de Métricas",
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Gráfico de tiempo de entrenamiento vs rendimiento
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        st.markdown("---")
        st.subheader("Tiempo de Entrenamiento vs Rendimiento")
        
        # Crear gráfico de dispersión con tamaño basado en F1-Macro
        fig_scatter = px.scatter(
            df_comparacion,
            x="Tiempo Entrenamiento (min)",
            y="F1-Macro",
            size="Accuracy",
            color="Modelo",
            hover_data=["Recall"],
            title="Relación entre Tiempo de Entrenamiento y Rendimiento",
            labels={
                "Tiempo Entrenamiento (min)": "Tiempo de Entrenamiento (minutos)",
                "F1-Macro": "F1-Macro",
                "Accuracy": "Accuracy",
            },
            template="plotly_white"
        )
        fig_scatter.update_layout(
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Gráfico de barras para tiempo de entrenamiento
        st.markdown("---")
        st.subheader("Tiempo de Entrenamiento por Modelo")
        
        fig_time = px.bar(
            df_comparacion,
            x="Modelo",
            y="Tiempo Entrenamiento (min)",
            title="Tiempo de Entrenamiento por Modelo",
            labels={"Tiempo Entrenamiento (min)": "Tiempo (minutos)", "Modelo": "Modelo"},
            color="Tiempo Entrenamiento (min)",
            color_continuous_scale="Reds",
            text="Tiempo Entrenamiento (min)"
        )
        fig_time.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig_time.update_layout(
            xaxis=dict(tickangle=45),
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig_time, use_container_width=True)


def comparar_modelos_deteccion(dfs):
    """Compara modelos de detección usando IoU e IoU por clase."""
    df_det = dfs.get("detection")
    
    if df_det is None:
        st.warning("⚠️ No existe detection_results.csv")
        return
    
    st.subheader("Comparación de Modelos de Detección")
    
    modelos_map = {
        "YOLOv8s Detección": "YOLO",
        "RetinaNet Detección": "RetinaNet",
        "Faster R-CNN Detección": "FRCNN",
    }
    
    resultados = []
    resultados_por_clase = []
    
    for nombre_modelo, model_key in modelos_map.items():
        df_model = df_det[df_det["det_model"] == model_key].copy()
        
        if df_model.empty:
            continue
        
        # Métricas globales
        mean_iou = df_model["iou"].mean()
        
        training_time = TRAINING_TIMES.get(nombre_modelo, None)
        resultados.append({
            "Modelo": nombre_modelo,
            "IoU Promedio": mean_iou,
            "Tiempo Entrenamiento (min)": training_time,
        })
        
        # Métricas por clase
        per_class_iou = (
            df_model.groupby("true_label")["iou"]
            .mean()
            .reindex(CLASES)
            .fillna(0)
        )
        
        for clase in CLASES:
            resultados_por_clase.append({
                "Modelo": nombre_modelo,
                "Clase": clase,
                "IoU": per_class_iou.loc[clase],
            })
    
    if not resultados:
        st.error("No hay datos disponibles para comparar.")
        return
    
    # Tabla comparativa global
    df_comparacion = pd.DataFrame(resultados)
    format_dict = {"IoU Promedio": "{:.4f}"}
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        format_dict["Tiempo Entrenamiento (min)"] = "{:.1f}"
    
    st.dataframe(
        df_comparacion.style.format(format_dict).highlight_max(axis=0, subset=["IoU Promedio"]).highlight_min(axis=0, subset=["Tiempo Entrenamiento (min)"]),
        use_container_width=True
    )
    
    # Gráfico de barras para IoU promedio
    st.markdown("---")
    st.subheader("IoU Promedio por Modelo")
    
    fig_global = px.bar(
        df_comparacion,
        x="Modelo",
        y="IoU Promedio",
        title="Comparación de IoU Promedio",
        labels={"IoU Promedio": "IoU Promedio", "Modelo": "Modelo"},
        color="IoU Promedio",
        color_continuous_scale="Blues",
        text="IoU Promedio"
    )
    fig_global.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_global.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, 1.05]),
        height=400,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig_global, use_container_width=True)
    
    # Tabla y gráfico por clase
    st.markdown("---")
    st.subheader("IoU por Clase")
    
    df_por_clase = pd.DataFrame(resultados_por_clase)
    
    fig_clases = px.bar(
        df_por_clase,
        x="Clase",
        y="IoU",
        color="Modelo",
        barmode="group",
        title="Comparación de IoU por Clase",
        labels={"IoU": "IoU", "Clase": "Clase"},
        text="IoU"
    )
    fig_clases.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_clases.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, 1.05]),
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_clases, use_container_width=True)
    
    # Heatmap de IoU por clase
    st.markdown("---")
    st.subheader("Mapa de Calor: IoU por Clase")
    
    df_heatmap_iou = df_por_clase.pivot(index="Clase", columns="Modelo", values="IoU")
    fig_heatmap_iou = px.imshow(
        df_heatmap_iou,
        labels=dict(x="Modelo", y="Clase", color="IoU"),
        color_continuous_scale="Blues",
        text_auto=".3f",
        aspect="auto"
    )
    fig_heatmap_iou.update_layout(
        title="Mapa de Calor: IoU por Clase y Modelo",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_heatmap_iou, use_container_width=True)
    
    # Tabla detallada por clase
    st.markdown("---")
    st.subheader("Tabla Detallada: IoU por Clase")
    
    df_pivot = df_por_clase.pivot(index="Clase", columns="Modelo", values="IoU")
    st.dataframe(
        df_pivot.style.format("{:.4f}").highlight_max(axis=1),
        use_container_width=True
    )
    
    # Gráfico de tiempo de entrenamiento vs IoU
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        st.markdown("---")
        st.subheader("Tiempo de Entrenamiento vs IoU Promedio")
        
        fig_scatter = px.scatter(
            df_comparacion,
            x="Tiempo Entrenamiento (min)",
            y="IoU Promedio",
            color="Modelo",
            size=[10] * len(df_comparacion),
            hover_data=["Modelo"],
            title="Relación entre Tiempo de Entrenamiento y IoU Promedio",
            labels={
                "Tiempo Entrenamiento (min)": "Tiempo de Entrenamiento (minutos)",
                "IoU Promedio": "IoU Promedio",
            },
            template="plotly_white"
        )
        fig_scatter.update_layout(
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Gráfico de barras para tiempo de entrenamiento
        st.markdown("---")
        st.subheader("Tiempo de Entrenamiento por Modelo")
        
        fig_time = px.bar(
            df_comparacion,
            x="Modelo",
            y="Tiempo Entrenamiento (min)",
            title="Tiempo de Entrenamiento por Modelo",
            labels={"Tiempo Entrenamiento (min)": "Tiempo (minutos)", "Modelo": "Modelo"},
            color="Tiempo Entrenamiento (min)",
            color_continuous_scale="Reds",
            text="Tiempo Entrenamiento (min)"
        )
        fig_time.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig_time.update_layout(
            xaxis=dict(tickangle=45),
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig_time, use_container_width=True)


def comparar_pipelines_completos(dfs):
    """Compara pipelines completos usando F1-macro, accuracy y recall."""
    df_full = dfs.get("full")
    
    if df_full is None:
        st.warning("⚠️ No existe full_eval_results.csv")
        return
    
    st.subheader("Comparación de Pipelines Completos")
    
    pipelines = {
        "YOLO → YOLO-CLS": ("YOLO", "YOLO-CLS"),
        "RetinaNet → ConvNeXt": ("RetinaNet", "ConvNeXt"),
        "RetinaNet → ViT": ("RetinaNet", "ViT"),
        "FRCNN → ConvNeXt": ("FRCNN", "ConvNeXt"),
        "FRCNN → ViT": ("FRCNN", "ViT"),
    }
    
    resultados = []
    
    for nombre_pipeline, (det_key, cls_key) in pipelines.items():
        df_pipeline = df_full[
            (df_full["det_model"] == det_key) & (df_full["cls_model"] == cls_key)
        ].copy()
        
        if df_pipeline.empty:
            continue
        
        y_true = df_pipeline["true_label"].astype(str).values
        y_pred = df_pipeline["pred_label"].astype(str).values
        
        # Calcular métricas
        acc = (y_true == y_pred).mean()
        _, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASES, average="macro", zero_division=0
        )
        
        training_time = TRAINING_TIMES.get(nombre_pipeline, None)
        resultados.append({
            "Pipeline": nombre_pipeline,
            "F1-Macro": f1_macro,
            "Accuracy": acc,
            "Recall": rec_macro,
            "Tiempo Entrenamiento (min)": training_time,
        })
    
    if not resultados:
        st.error("No hay datos disponibles para comparar.")
        return
    
    df_comparacion = pd.DataFrame(resultados)
    
    # Mostrar tabla comparativa
    format_dict = {
        "F1-Macro": "{:.4f}",
        "Accuracy": "{:.4f}",
        "Recall": "{:.4f}",
    }
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        format_dict["Tiempo Entrenamiento (min)"] = "{:.1f}"
    
    highlight_subset = ["F1-Macro", "Accuracy", "Recall"]
    style_obj = df_comparacion.style.format(format_dict).highlight_max(axis=0, subset=highlight_subset)
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        style_obj = style_obj.highlight_min(axis=0, subset=["Tiempo Entrenamiento (min)"])
    
    st.dataframe(style_obj, use_container_width=True)
    
    # Gráfico de barras comparativo
    st.markdown("---")
    st.subheader("Visualización Comparativa")
    
    df_melted = df_comparacion.melt(
        id_vars=["Pipeline"],
        value_vars=["F1-Macro", "Accuracy", "Recall"],
        var_name="Métrica",
        value_name="Valor"
    )
    
    fig = px.bar(
        df_melted,
        x="Pipeline",
        y="Valor",
        color="Métrica",
        barmode="group",
        title="Comparación de Métricas de Pipelines Completos",
        labels={"Valor": "Valor de la Métrica", "Pipeline": "Pipeline"},
        color_discrete_map={
            "F1-Macro": "#1f77b4",
            "Accuracy": "#ff7f0e",
            "Recall": "#2ca02c"
        },
        text="Valor"
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, 1.05]),
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap de comparación
    st.markdown("---")
    st.subheader("Mapa de Calor Comparativo")
    
    df_heatmap = df_comparacion.set_index("Pipeline")[["F1-Macro", "Accuracy", "Recall"]]
    fig_heatmap = px.imshow(
        df_heatmap.T,
        labels=dict(x="Pipeline", y="Métrica", color="Valor"),
        color_continuous_scale="Blues",
        text_auto=".3f",
        aspect="auto"
    )
    fig_heatmap.update_layout(
        title="Mapa de Calor: Comparación de Métricas de Pipelines",
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Gráfico de tiempo de entrenamiento vs rendimiento
    if "Tiempo Entrenamiento (min)" in df_comparacion.columns:
        st.markdown("---")
        st.subheader("Tiempo de Entrenamiento vs Rendimiento")
        
        # Crear gráfico de dispersión con tamaño basado en F1-Macro
        fig_scatter = px.scatter(
            df_comparacion,
            x="Tiempo Entrenamiento (min)",
            y="F1-Macro",
            size="Accuracy",
            color="Pipeline",
            hover_data=["Recall"],
            title="Relación entre Tiempo de Entrenamiento y Rendimiento",
            labels={
                "Tiempo Entrenamiento (min)": "Tiempo de Entrenamiento (minutos)",
                "F1-Macro": "F1-Macro",
                "Accuracy": "Accuracy",
            },
            template="plotly_white"
        )
        fig_scatter.update_layout(
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Gráfico de barras para tiempo de entrenamiento
        st.markdown("---")
        st.subheader("Tiempo de Entrenamiento por Pipeline")
        
        fig_time = px.bar(
            df_comparacion,
            x="Pipeline",
            y="Tiempo Entrenamiento (min)",
            title="Tiempo de Entrenamiento por Pipeline",
            labels={"Tiempo Entrenamiento (min)": "Tiempo (minutos)", "Pipeline": "Pipeline"},
            color="Tiempo Entrenamiento (min)",
            color_continuous_scale="Reds",
            text="Tiempo Entrenamiento (min)"
        )
        fig_time.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig_time.update_layout(
            xaxis=dict(tickangle=45),
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig_time, use_container_width=True)


# ============================================================
# SIDEBAR
# ============================================================


def sidebar_seleccion():
    st.sidebar.header("Configuración")

    vista = st.sidebar.radio(
        "Vista",
        ["Modelos de clasificación", "Modelos de detección", "Pipelines completos", "Comparar Modelos"],
    )

    if vista == "Modelos de clasificación":
        modelos_disp = [
            "YOLOv8s Clasificación",
            "ConvNeXt Clasificación",
            "ViT Clasificación",
        ]
        modelo = st.sidebar.selectbox("Modelo", modelos_disp)
        modo = "Imágenes sueltas"
        umbral = 0.5  # HARD CODED

    elif vista == "Modelos de detección":
        modelos_disp = [
            "YOLOv8s Detección",
            "RetinaNet Detección",
            "Faster R-CNN Detección",
        ]
        modelo = st.sidebar.selectbox("Modelo", modelos_disp)
        modo = "Imágenes sueltas"
        umbral = 0.5  # HARD CODED

    elif vista == "Pipelines completos":
        modelos_disp = [
            "YOLO → YOLO-CLS",
            "RetinaNet → ConvNeXt",
            "RetinaNet → ViT",
            "FRCNN → ConvNeXt",
            "FRCNN → ViT",
        ]
        modelo = st.sidebar.selectbox("Modelo", modelos_disp)
        modo = "Imágenes sueltas"
        umbral = 0.5  # HARD CODED

    else:  # Comparar Modelos
        modelo = None
        modo = None
        umbral = None

    return vista, modelo, modo, umbral


# ============================================================
# CLASIFICACIÓN
# ============================================================
def vista_clasificacion(modelo_nombre, modo_entrada, umbral_score, dfs):
    mostrar_metricas_modelo(modelo_nombre, dfs)
    st.markdown("---")
    st.subheader("Probar imágenes")

    with st.spinner("Cargando modelo..."):
        modelo, device, tipo, transform = cargar_modelo_clasificacion(
            modelo_nombre, MODELS_DIR
        )

    if modo_entrada == "Imágenes sueltas":
        subir_imagenes_clasificacion(modelo, device, tipo, transform)
    else:
        st.info("CSV classification batch — not implemented fully yet.")


# ============================================================
# DETECCIÓN
# ============================================================
def vista_deteccion(modelo_nombre, modo_entrada, umbral_score, dfs):
    mostrar_metricas_modelo(modelo_nombre, dfs)
    st.markdown("---")
    st.subheader("Probar detección")

    with st.spinner("Cargando modelo..."):
        modelo, device, tipo, transform = cargar_modelo_deteccion(
            modelo_nombre, MODELS_DIR, score_thresh=umbral_score
        )

    if modo_entrada == "Imágenes sueltas":
        subir_imagenes_deteccion(modelo, device, tipo, transform, umbral_score)
    else:
        st.info("CSV detection batch — not implemented fully yet.")


# ============================================================
# FULL PIPELINES — INFERENCE
# ============================================================
def vista_pipeline(modelo_nombre, modo_entrada, umbral_score, dfs):
    mostrar_metricas_modelo(modelo_nombre, dfs)
    st.markdown("---")
    st.subheader("Probar Pipeline Completo")

    det_key, cls_key = MODEL_EVAL_MAP[modelo_nombre][1]

    with st.spinner("Cargando modelos del pipeline..."):
        det_model, det_dev, det_tipo, det_tf = cargar_modelo_deteccion(
            det_key, MODELS_DIR, score_thresh=0.5
        )

        cls_model, cls_dev, cls_tipo, cls_tf = cargar_modelo_clasificacion(
            (
                cls_key + " Clasificación"
                if cls_key != "YOLO-CLS"
                else "YOLOv8s Clasificación"
            ),
            MODELS_DIR,
        )

    if modo_entrada == "Imágenes sueltas":
        subir_imagenes_pipeline(
            det_model, det_dev, det_tipo, det_tf, cls_model, cls_dev, cls_tipo, cls_tf
        )
    else:
        subir_csv_pipeline(
            det_model, det_dev, det_tipo, det_tf, cls_model, cls_dev, cls_tipo, cls_tf
        )


def subir_csv_pipeline(
    det_model, det_dev, det_tipo, det_tf, cls_model, cls_dev, cls_tipo, cls_tf
):
    st.info("CSV pipeline batch — not implemented fully yet.")


# ============================================================
# MAIN
# ============================================================
def main():
    configurar_pagina()
    dfs = load_eval_results()
    vista, modelo, modo, umbral = sidebar_seleccion()

    images_root = bootstrap_dataset()
    st.sidebar.success(f"Imágenes en: {images_root}")

    if vista == "Modelos de clasificación":
        vista_clasificacion(modelo, modo, umbral, dfs)

    elif vista == "Modelos de detección":
        vista_deteccion(modelo, modo, umbral, dfs)

    elif vista == "Pipelines completos":
        vista_pipeline(modelo, modo, umbral, dfs)
    
    else:  # Comparar Modelos
        st.header("Comparación de Modelos")
        st.markdown("Compara el rendimiento de diferentes modelos usando métricas estándar.")
        
        tipo_comparacion = st.radio(
            "Tipo de comparación",
            ["Modelos de Clasificación", "Modelos de Detección", "Pipelines Completos"],
            horizontal=True
        )
        
        if tipo_comparacion == "Modelos de Clasificación":
            comparar_modelos_clasificacion(dfs)
        elif tipo_comparacion == "Modelos de Detección":
            comparar_modelos_deteccion(dfs)
        else:  # Pipelines Completos
            comparar_pipelines_completos(dfs)


if __name__ == "__main__":
    main()
