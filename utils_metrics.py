# utils_metrics.py
# Funciones auxiliares para métricas y gráficas (versión Plotly)

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

from models_classification import CLASES

# -------------------- CLASIFICACIÓN --------------------


def mostrar_resumen_clasificacion(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un pequeño resumen con accuracy global y por clase."""
    df_eval = df_eval.copy()
    df_eval["correcto"] = df_eval["true_label"] == df_eval["pred_label"]

    acc_global = df_eval["correcto"].mean()

    por_clase = df_eval.groupby("true_label")["correcto"].mean().reindex(CLASES)

    resumen = pd.DataFrame({"accuracy": por_clase})
    resumen.loc["GLOBAL", "accuracy"] = acc_global

    return resumen


def graficar_confusion_clasificacion(df_eval: pd.DataFrame):
    """
    Matriz de confusión interactiva con Plotly.
    Normalizada por filas.
    """
    y_true = df_eval["true_label"].values
    y_pred = df_eval["pred_label"].values

    cm = confusion_matrix(y_true, y_pred, labels=CLASES, normalize="true")

    fig = px.imshow(
        cm,
        x=CLASES,
        y=CLASES,
        color_continuous_scale="Blues",
        labels=dict(x="Predicción", y="Etiqueta Real"),
        text_auto=".2f",
    )

    fig.update_layout(
        title="Matriz de Confusión (Normalizada)",
        xaxis=dict(tickangle=45),
        height=550,
    )

    return fig


# -------------------- DETECCIÓN — EVALUACIÓN SIMPLE --------------------


def graficar_metricas_deteccion_simple(df_eval: pd.DataFrame):
    """
    Produce 2 gráficas interactivas:
    - Histograma de IoU
    - Porcentaje de coincidencias por clase
    """
    # --- IoU distribution ---
    fig_iou = px.histogram(
        df_eval,
        x="iou",
        nbins=20,
        title="Distribución de IoU",
        color_discrete_sequence=["#1f77b4"],
    )
    fig_iou.update_layout(
        bargap=0.05,
        xaxis_title="IoU",
        yaxis_title="Frecuencia",
        height=450,
    )

    # --- Match rate per class ---
    por_clase = (
        df_eval.groupby("true_label")["match"]
        .mean()
        .reindex(CLASES)
        .reset_index()
        .rename(columns={"match": "match_rate"})
    )

    fig_match = px.bar(
        por_clase,
        x="true_label",
        y="match_rate",
        labels={"true_label": "Clase", "match_rate": "Match Rate"},
        title="Aciertos por Clase (detección)",
        range_y=[0, 1],
        text="match_rate",
        color="match_rate",
        color_continuous_scale="Blues",
    )
    fig_match.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_match.update_layout(
        height=450,
        xaxis=dict(tickangle=45),
    )

    return fig_iou, fig_match
