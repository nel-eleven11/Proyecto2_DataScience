# models_classification.py
# Funciones para cargar y usar modelos de CLASIFICACIÓN de mosquitos.

from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image
import numpy as np
import pandas as pd
import timm
from torchvision import transforms as T
from ultralytics import YOLO

# Clases de tu problema
CLASES = [
    "aegypti",
    "albopictus",
    "anopheles",
    "culex",
    "culiseta",
    "japonicus/koreicus",
]


# -------------------------- CARGA DE MODELOS --------------------------


def _transform_timm(img_size: int = 224):
    """Transformación estándar para modelos timm (ConvNeXt, ViT)."""
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def cargar_modelo_clasificacion(
    nombre_modelo: str, models_dir: Path
) -> Tuple[torch.nn.Module, torch.device, str, T.Compose]:
    """
    Carga el modelo de clasificación correspondiente.

    Devuelve:
        modelo, dispositivo, tipo_modelo ("yolo" o "timm"), transform
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "YOLO" in nombre_modelo:
        pesos = models_dir / "yolo_cls_best.pt"
        modelo = YOLO(str(pesos))  # modelo ya está en modo eval por defecto
        tipo = "yolo"
        transform = None  # YOLO maneja internamente la lectura/transformación
    elif "ConvNeXt" in nombre_modelo:
        pesos = models_dir / "convnext_cls_weights.pth"
        modelo = timm.create_model(
            "convnext_base",
            pretrained=False,
            num_classes=len(CLASES),
        )
        state_dict = torch.load(pesos, map_location=device)
        modelo.load_state_dict(state_dict)
        modelo.to(device)
        modelo.eval()
        tipo = "timm"
        transform = _transform_timm(img_size=224)
    elif "ViT" in nombre_modelo:
        pesos = models_dir / "ViT_cls_weights.pth"
        modelo = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=len(CLASES),
        )
        state_dict = torch.load(pesos, map_location=device)
        modelo.load_state_dict(state_dict)
        modelo.to(device)
        modelo.eval()
        tipo = "timm"
        transform = _transform_timm(img_size=224)
    else:
        raise ValueError(f"Modelo de clasificación no soportado: {nombre_modelo}")

    return modelo, device, tipo, transform


# -------------------------- INFERENCIA --------------------------


def _pred_yolo_cls(
    modelo: YOLO, imagenes_pil: List[Image.Image]
) -> List[Dict]:
    """Predicciones con YOLOv8 clasificación."""
    # Ultralytics acepta rutas, arrays o PIL directamente
    resultados = modelo(
        imagenes_pil,
        verbose=False,
    )
    salida = []
    for r in resultados:
        probs = r.probs  # objeto ultralytics
        scores = probs.data.cpu().numpy()
        topk_idx = np.argsort(scores)[::-1][:5]
        salida.append(
            {
                "top1_idx": int(topk_idx[0]),
                "top1_clase": CLASES[int(topk_idx[0])],
                "top1_prob": float(scores[topk_idx[0]]),
                "topk_idx": topk_idx.tolist(),
                "topk_clases": [CLASES[i] for i in topk_idx],
                "topk_probs": [float(scores[i]) for i in topk_idx],
            }
        )
    return salida


def _pred_timm_cls(
    modelo: torch.nn.Module,
    dispositivo: torch.device,
    transform: T.Compose,
    imagenes_pil: List[Image.Image],
) -> List[Dict]:
    """Predicciones con modelos timm (ConvNeXt, ViT)."""
    modelo.eval()
    salida = []
    with torch.no_grad():
        for img in imagenes_pil:
            x = transform(img).unsqueeze(0).to(dispositivo)
            logits = modelo(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            topk_idx = np.argsort(probs)[::-1][:5]
            salida.append(
                {
                    "top1_idx": int(topk_idx[0]),
                    "top1_clase": CLASES[int(topk_idx[0])],
                    "top1_prob": float(probs[topk_idx[0]]),
                    "topk_idx": topk_idx.tolist(),
                    "topk_clases": [CLASES[i] for i in topk_idx],
                    "topk_probs": [float(probs[i]) for i in topk_idx],
                }
            )
    return salida


def predecir_imagenes_clasificacion(
    modelo, dispositivo, tipo_modelo, transform, imagenes_pil: List[Image.Image]
) -> List[Dict]:
    """Wrapper de alto nivel usado por el dashboard."""
    if tipo_modelo == "yolo":
        return _pred_yolo_cls(modelo, imagenes_pil)
    else:
        return _pred_timm_cls(modelo, dispositivo, transform, imagenes_pil)


# -------------------------- EVALUACIÓN CON CSV --------------------------


def evaluar_csv_clasificacion(
    modelo,
    dispositivo,
    tipo_modelo,
    transform,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lee las imágenes indicadas en df["img_path"], predice la clase
    y devuelve un DataFrame con columnas:
    img_path, true_label, true_idx, pred_label, pred_idx
    """
    imagenes = []
    rutas_validas = []
    for p in df["img_path"]:
        img = Image.open(p).convert("RGB")
        imagenes.append(img)
        rutas_validas.append(p)

    resultados = predecir_imagenes_clasificacion(
        modelo, dispositivo, tipo_modelo, transform, imagenes
    )

    true_labels = df.loc[df["img_path"].isin(rutas_validas), "class_label"].tolist()

    registros = []
    for ruta, true_lab, res in zip(rutas_validas, true_labels, resultados):
        registros.append(
            {
                "img_path": ruta,
                "true_label": true_lab,
                "true_idx": CLASES.index(true_lab),
                "pred_label": res["top1_clase"],
                "pred_idx": res["top1_idx"],
            }
        )

    return pd.DataFrame(registros)
