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

    dataset_path = kagglehub.dataset_download("vishakkbhat/mosquito-data")
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
# SIDEBAR
# ============================================================


def sidebar_seleccion():
    st.sidebar.header("Configuración")

    vista = st.sidebar.radio(
        "Vista",
        ["Modelos de clasificación", "Modelos de detección", "Pipelines completos"],
    )

    if vista == "Modelos de clasificación":
        modelos_disp = [
            "YOLOv8s Clasificación",
            "ConvNeXt Clasificación",
            "ViT Clasificación",
        ]

    elif vista == "Modelos de detección":
        modelos_disp = [
            "YOLOv8s Detección",
            "RetinaNet Detección",
            "Faster R-CNN Detección",
        ]

    else:  # Pipelines completos
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
        subir_csv_clasificacion(modelo, device, tipo, transform)


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
        subir_csv_deteccion(modelo, device, tipo, transform, umbral_score)


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

    else:
        vista_pipeline(modelo, modo, umbral, dfs)


if __name__ == "__main__":
    main()
