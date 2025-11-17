# utils_metrics.py
# Funciones auxiliares para métricas y gráficas.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from models_classification import CLASES


# -------------------- CLASIFICACIÓN --------------------


def mostrar_resumen_clasificacion(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un pequeño resumen con accuracy global y por clase."""
    df_eval = df_eval.copy()
    df_eval["correcto"] = df_eval["true_label"] == df_eval["pred_label"]

    acc_global = df_eval["correcto"].mean()

    por_clase = (
        df_eval.groupby("true_label")["correcto"].mean().reindex(CLASES)
    )

    resumen = pd.DataFrame(
        {
            "accuracy": por_clase,
        }
    )
    resumen.loc["GLOBAL", "accuracy"] = acc_global

    return resumen


def graficar_confusion_clasificacion(df_eval: pd.DataFrame):
    """Genera figura de matriz de confusión."""
    y_true = df_eval["true_label"].values
    y_pred = df_eval["pred_label"].values

    cm = confusion_matrix(y_true, y_pred, labels=CLASES, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASES)),
        yticks=np.arange(len(CLASES)),
        xticklabels=CLASES,
        yticklabels=CLASES,
        ylabel="Etiqueta verdadera",
        xlabel="Etiqueta predicha",
        title="Matriz de confusión (normalizada)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Escribir valores dentro de la matriz
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


# -------------------- DETECCIÓN (EVAL SIMPLE) --------------------


def graficar_metricas_deteccion_simple(df_eval: pd.DataFrame):
    """
    Toma el DataFrame devuelto por evaluar_csv_deteccion_simple y genera:
    - Histograma de IoU
    - Barra de porcentaje de matches por clase
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histograma de IoU
    axes[0].hist(df_eval["iou"], bins=20, range=(0, 1.0))
    axes[0].set_xlabel("IoU")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de IoU")

    # Porcentaje de matches por clase
    por_clase = (
        df_eval.groupby("true_label")["match"].mean().reindex(CLASES)
    )

    axes[1].bar(range(len(CLASES)), por_clase.values)
    axes[1].set_xticks(range(len(CLASES)))
    axes[1].set_xticklabels(CLASES, rotation=45, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Porcentaje de aciertos")
    axes[1].set_title("Aciertos por clase (detección)")

    fig.tight_layout()
    return fig
