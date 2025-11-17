# models_detection.py
# Funciones para cargar y usar modelos de DETECCIÓN de mosquitos.

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          retinanet_resnet50_fpn)
from ultralytics import YOLO

from models_classification import CLASES


def _transform_det(img_size: int = 640):
    """Transformación simple para modelos de detección torchvision."""
    return T.Compose([T.ToTensor()])


def _safe_load_state_dict(model, state_dict):
    """
    Carga un state_dict ignorando SOLO los pesos incompatibles,
    como cls_logits y cls_bias en detection heads de PyTorch.
    """
    model_dict = model.state_dict()
    filtered = {}

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered[k] = v
        else:
            # Ignorar pesos incompatibles del head
            print(
                f"[IGNORADO] {k} — shape checkpoint {v.shape} "
                f"≠ shape modelo {model_dict.get(k, None)}"
            )

    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)
    return model


def cargar_modelo_deteccion(
    nombre_modelo: str, models_dir: Path, score_thresh: float = 0.5
) -> Tuple[torch.nn.Module, torch.device, str, T.Compose]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------
    # YOLO — no problem
    # ----------------------------------------------------
    if "YOLO" in nombre_modelo:
        pesos = models_dir / "yolov8s_det_best.pt"
        modelo = YOLO(str(pesos))
        modelo.to(device)
        tipo = "yolo"
        transform = None
        return modelo, device, tipo, transform

    # ----------------------------------------------------
    # PYTORCH MODELS — need safe loading
    # trained with NUM_CLASSES = len(CLASES) + 1 (background)
    # ----------------------------------------------------
    num_classes_trained = len(CLASES) + 1  # 7

    if "RetinaNet" in nombre_modelo:
        pesos = models_dir / "ret_det_weights.pth"
        modelo = retinanet_resnet50_fpn(weights=None, num_classes=num_classes_trained)

        state_dict = torch.load(pesos, map_location=device)
        modelo = _safe_load_state_dict(modelo, state_dict)

        modelo.to(device)
        modelo.eval()
        modelo.score_thresh = score_thresh
        tipo = "torchvision"
        transform = _transform_det()
        return modelo, device, tipo, transform

    if "Faster" in nombre_modelo or "RCNN" in nombre_modelo:
        pesos = models_dir / "rcnn_model_weights.pth"
        modelo = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes_trained)

        state_dict = torch.load(pesos, map_location=device)
        modelo = _safe_load_state_dict(modelo, state_dict)

        modelo.to(device)
        modelo.eval()
        tipo = "torchvision"
        transform = _transform_det()
        return modelo, device, tipo, transform

    # ----------------------------------------------------
    raise ValueError(f"Modelo de detección no soportado: {nombre_modelo}")


# -------------------------- INFERENCIA --------------------------


def _dibujar_cajas(img_pil, boxes, scores, score_thresh=0.5):
    """Dibuja cajas sobre una imagen PIL y devuelve un array RGB."""
    img = np.array(img_pil).copy()

    for box, sc in zip(boxes, scores):
        if sc < score_thresh:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"mosquito {sc:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def predecir_imagenes_deteccion(
    modelo, device, tipo_modelo, transform, imagenes_pil, score_thresh=0.5
):

    resultados_salida = []

    if tipo_modelo == "yolo":
        res = modelo(imagenes_pil, conf=score_thresh, verbose=False)
        for r, img_pil in zip(res, imagenes_pil):
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    sc = float(b.conf.item())
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    dets.append({"score": sc, "bbox": [x1, y1, x2, y2]})

            img_bgr = r.plot()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resultados_salida.append(
                {"detecciones": dets, "imagen_visualizada": img_rgb}
            )

        return resultados_salida

    # ------------------- PYTORCH (RetinaNet / RCNN) -------------------

    modelo.eval()
    with torch.no_grad():
        for img_pil in imagenes_pil:
            x = transform(img_pil).to(device)
            out = modelo([x])[0]

            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            dets = []
            for box, sc in zip(boxes, scores):
                if sc < score_thresh:
                    continue
                dets.append({"score": float(sc), "bbox": box.tolist()})

            img_viz = _dibujar_cajas(img_pil, boxes, scores, score_thresh)
            resultados_salida.append(
                {"detecciones": dets, "imagen_visualizada": img_viz}
            )

    return resultados_salida


# ------------------------ EVALUACIÓN CSV ------------------------


def _iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-6)


import pandas as pd


def evaluar_csv_deteccion_simple(
    modelo, device, tipo, transform, df, umbral_score=0.5, umbral_iou=0.5
):
    registros = []

    for img_path, grupo in df.groupby("img_path"):

        img = Image.open(img_path).convert("RGB")
        pred = predecir_imagenes_deteccion(
            modelo, device, tipo, transform, [img], umbral_score
        )[0]["detecciones"]

        if not pred:
            for _, row in grupo.iterrows():
                registros.append(
                    {
                        "img_path": img_path,
                        "true_label": row["class_label"],
                        "iou": 0.0,
                        "match": False,
                    }
                )
            continue

        for _, row in grupo.iterrows():
            gt = [row["bbx_xtl"], row["bbx_ytl"], row["bbx_xbr"], row["bbx_ybr"]]

            best_iou = 0
            for det in pred:
                best_iou = max(best_iou, _iou(gt, det["bbox"]))

            registros.append(
                {
                    "img_path": img_path,
                    "true_label": row["class_label"],
                    "iou": best_iou,
                    "match": best_iou >= umbral_iou,
                }
            )

    return pd.DataFrame(registros)
