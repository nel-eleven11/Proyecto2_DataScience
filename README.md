# Proyecto 2. Mosquito Alert

- Análisis Exploratorio e Implementación de Modelos:
  - **Nota:** Realizamos cambios a los modelos entre la Fase 2 y Fase 3 (más que todo entrenamiento más robusto usando el compute de Kaggle), en el subfolder se encuentran todos los Jupyter por temas de documentación. También corrimos los Notebooks dentro de Kaggle, por lo que se pueden ver aquí o dentro de los links de kaggle adjuntados para la Fase 3.
  - models_pipeline/
    - Fase 2
      - Fase 2.ipynb
    - Fase 3 / Completo
      - [00_EDA_and_Cleaning](https://www.kaggle.com/code/josantoniomrida/00-eda-and-cleaning)
      - [01_YoloV8_Detection](https://www.kaggle.com/code/josantoniomrida/01-yolov8-detection)
      - [02_YoloV8_Classification](https://www.kaggle.com/code/josantoniomrida/02-yolov8-classification)
      - [03_RetinaNet_Detection](https://www.kaggle.com/code/josantoniomrida/03-retinanet-detection)
      - [04_RCNN_Detection](https://www.kaggle.com/code/josantoniomrida/04-rcnn-detection)
      - [05_ConvNext_Classification](https://www.kaggle.com/code/josantoniomrida/05-convnext-classification)
      - [06_ViT_Classification](https://www.kaggle.com/code/josantoniomrida/06-vit-classification)
      - [07_Eval](https://www.kaggle.com/code/josantoniomrida/07-eval)
        - El script de Eval fue super corto, solo queríamos tener un .csv para cargar el rendimiento de los modelos a Streamlit.
        - Algunos modelos dicen que corrieron con un "error", sin emabrgo esta únicamente fue la última celda donde se guardaban datos cómo duración de épocas etc. dentro de un .json.
- Informes:
  - informes/
  -   Fase 1.pdf y [Link de Office](https://uvggt-my.sharepoint.com/:w:/g/personal/gar22434_uvg_edu_gt/IQCjrjGDKbD3QqQHbmXPjFVLAcKjOH0JfZWrotX-aXAUloE?e=pHkpDP)
  -   Fase 2.pdf y [Link de office](https://uvggt-my.sharepoint.com/:w:/g/personal/pue22296_uvg_edu_gt/IQC4PtVyOwyCR6vKT2DpHu7tAbRmN9hWwY-7KbBrhZPGSP4?e=4ABzGH)
  -   Fase 3.pdf
  -   **Nota:** Movimos los archivos ahorita para la última entrega, ya estaban agregados al repo en las fechas correspondientes.
 
- Presentaciones:
  - presentaciones/
    - Fase 2.pdf
---

Correr Dashboard con: 

```bash
streamlit run dashboard.py
```
