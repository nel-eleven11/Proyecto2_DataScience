# models_detection.py
# Funciones para cargar y usar modelos de DETECCIÓN de mosquitos.

from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torchvision import transforms as T
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    fasterrcnn_resnet50_fpn,
)
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

from models_classification import CLASES


def _transform_det(img_size: int = 640):
    """Transformación simple para modelos de detección torchvision."""
    return T.Compose(
        [
            T.ToTensor(),  # convierte a [0,1] y mueve canales
        ]
    )


def cargar_modelo_deteccion(
    nombre_modelo: str, models_dir: Path, score_thresh: float = 0.5
) -> Tuple[torch.nn.Module, torch.device, str, T.Compose]:
    """
    Carga el modelo de detección.

    Devuelve:
        modelo, dispositivo, tipo_modelo ("yolo" o "torchvision"), transform
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "YOLO" in nombre_modelo:
        pesos = models_dir / "yolov8s_det_best.pt"
        modelo = YOLO(str(pesos))
        tipo = "yolo"
        transform = None  # YOLO maneja internamente la lectura
        modelo.to(device)
    elif "RetinaNet" in nombre_modelo:
        pesos = models_dir / "ret_det_weights.pth"
        modelo = retinanet_resnet50_fpn(weights=None, num_classes=len(CLASES))
        state_dict = torch.load(pesos, map_location=device)
        modelo.load_state_dict(state_dict)
        modelo.to(device)
        modelo.eval()
        modelo.score_thresh = score_thresh
        tipo = "torchvision"
        transform = _transform_det()
    elif "Faster" in nombre_modelo:
        pesos = models_dir / "rcnn_model_weights.pth"
        modelo = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(CLASES))
        state_dict = torch.load(pesos, map_location=device)
        modelo.load_state_dict(state_dict)
        modelo.to(device)
        modelo.eval()
        tipo = "torchvision"
        transform = _transform_det()
    else:
        raise ValueError(f"Modelo de detección no soportado: {nombre_modelo}")

    return modelo, device, tipo, transform


# -------------------------- INFERENCIA --------------------------


def _dibujar_cajas(
    img_pil: Image.Image, boxes, labels, scores, score_thresh: float = 0.5
):
    """Dibuja cajas sobre una imagen PIL y devuelve un array RGB."""
    img = np.array(img_pil).copy()
    h, w, _ = img.shape

    for box, lab, sc in zip(boxes, labels, scores):
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        texto = f"{CLASES[int(lab)]} {sc:.2f}"
        cv2.putText(
            img,
            texto,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return img


def predecir_imagenes_deteccion(
    modelo,
    dispositivo,
    tipo_modelo: str,
    transform,
    imagenes_pil: List[Image.Image],
    score_thresh: float = 0.5,
) -> List[Dict]:
    """Predice detecciones y devuelve info + imagen visualizada."""
    resultados_salida = []

    if tipo_modelo == "yolo":
        # Ultralytics se encarga de todo internamente
        res = modelo(
            imagenes_pil,
            conf=score_thresh,
            verbose=False,
        )
        for r, img_pil in zip(res, imagenes_pil):
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    cls_idx = int(b.cls.item())
                    sc = float(b.conf.item())
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    dets.append(
                        {
                            "clase": CLASES[cls_idx],
                            "score": sc,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )
            # r.plot() devuelve BGR; convertir a RGB
            img_bgr = r.plot()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resultados_salida.append(
                {
                    "detecciones": dets,
                    "imagen_visualizada": img_rgb,
                }
            )
    else:
        modelo.eval()
        with torch.no_grad():
            for img_pil in imagenes_pil:
                x = transform(img_pil).to(dispositivo)
                outputs = modelo([x])[0]
                boxes = outputs["boxes"].cpu().numpy()
                labels = outputs["labels"].cpu().numpy()
                scores = outputs["scores"].cpu().numpy()

                dets = []
                for box, lab, sc in zip(boxes, labels, scores):
                    if sc < score_thresh:
                        continue
                    dets.append(
                        {
                            "clase": CLASES[int(lab)],
                            "score": float(sc),
                            "bbox": box.tolist(),
                        }
                    )

                img_viz = _dibujar_cajas(
                    img_pil, boxes, labels, scores, score_thresh=score_thresh
                )
                resultados_salida.append(
                    {
                        "detecciones": dets,
                        "imagen_visualizada": img_viz,
                    }
                )

    return resultados_salida


# ------------------------ EVALUACIÓN SIMPLE CSV ------------------------


def _iou(boxA, boxB):
    """Calcula IOU entre dos cajas [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return iou


import pandas as pd


def evaluar_csv_deteccion_simple(
    modelo,
    dispositivo,
    tipo_modelo,
    transform,
    df: pd.DataFrame,
    umbral_score: float = 0.5,
    umbral_iou: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluación muy simple de detección:
    - Para cada anotación del CSV se busca la mejor predicción de esa imagen
      y se calcula IOU.
    - Marca True/False si la clase coincide y IOU >= umbral_iou.
    Devuelve un DataFrame con columnas:
        img_path, true_label, pred_label, iou, match
    """
    registros = []

    # Agrupar por imagen
    for img_path, grupo in df.groupby("img_path"):
        img = Image.open(img_path).convert("RGB")
        res = predecir_imagenes_deteccion(
            modelo,
            dispositivo,
            tipo_modelo,
            transform,
            [img],
            score_thresh=umbral_score,
        )[0]
        pred_dets = res["detecciones"]

        # Si no hay detecciones, todas las anotaciones son fallo
        if not pred_dets:
            for _, row in grupo.iterrows():
                registros.append(
                    {
                        "img_path": img_path,
                        "true_label": row["class_label"],
                        "pred_label": None,
                        "iou": 0.0,
                        "match": False,
                    }
                )
            continue

        # Para cada anotación GT buscar la mejor predicción por IOU
        for _, row in grupo.iterrows():
            gt_box = [
                row["bbx_xtl"],
                row["bbx_ytl"],
                row["bbx_xbr"],
                row["bbx_ybr"],
            ]
            gt_label = row["class_label"]

            best_iou = 0.0
            best_pred_label = None

            for det in pred_dets:
                iou_val = _iou(gt_box, det["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred_label = det["clase"]

            match = best_iou >= umbral_iou and best_pred_label == gt_label

            registros.append(
                {
                    "img_path": img_path,
                    "true_label": gt_label,
                    "pred_label": best_pred_label,
                    "iou": best_iou,
                    "match": match,
                }
            )

    return pd.DataFrame(registros)
