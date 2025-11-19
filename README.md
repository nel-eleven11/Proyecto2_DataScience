# Proyecto 2. Mosquito Alert
La ubicación de todos los archivos correspondientes se encuentra a continuación, para el dashboard tenemos un [demo en YouTube](https://youtu.be/oQT28CPXuGI) y también están los comandos para correrlo en el repositorio.
## Ubicación de Archivos
- Análisis Exploratorio e Implementación de Modelos:
  - **Nota:** Realizamos cambios a los modelos entre la Fase 2 y Fase 3 (más que todo entrenamiento más robusto usando el compute de Kaggle), en el subfolder se encuentran todos los Jupyter por temas de documentación. También corrimos los Notebooks dentro de Kaggle, por lo que se pueden ver aquí o dentro de los links de kaggle adjuntados para la Fase 3. Adicionalmente, los notebooks descargados de Kaggle **pueden no cargar dentro de GitHub** por lo cual sugerimos utilizar nbviewer.org y pegar el link del repositorio para ver sus contenidos, o simplemente visitar los links de Kaggle.
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
  -   Fase 3.pdf / Informe Final y [Link de Office](https://uvggt-my.sharepoint.com/:w:/g/personal/mer201105_uvg_edu_gt/IQDSZsMa1rMASprM6y7TZxgHAcgrkBnM1WvXOnbn-KkbKKY?e=K1kXu0)
  -   **Nota:** Movimos los archivos ahorita para la última entrega, ya estaban agregados al repo en las fechas correspondientes.
 
- Presentaciones:
  - presentaciones/
    - Fase 2.pdf
    - Fase 3.pdf
---

## Dashboard
El Dashboard necesita realizar algunas descargas, por lo que sugiero basarse en la [demo en YouTube](https://youtu.be/oQT28CPXuGI) principalmente. Tiene que descargar imágenes y los mejores pesos de los  modelos usando Kagglehub.

Clonar el Repositorio
```bash
git clone git@github.com:nel-eleven11/Proyecto2_DataScience
```
Clonar Virtual Environment
```bash
python3 -m venv .venv 
```
```bash
source .venv/bin/activate
```

Instalar Dependencias
```bash
pip install -r requirements.txt
```
Correr Proyecto
```bash
streamlit run dashboard.py
```
