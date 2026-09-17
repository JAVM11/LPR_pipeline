# Créditos y Agradecimientos

Este proyecto es un **pipeline de integración** que combina el trabajo de varios proyectos open source. 
Sin ellos, este repo no existiría.

## Detección de Objetos (YOLO)

### Ultralytics YOLOv8
- **Repositorio**: https://github.com/ultralytics/ultralytics
- **Autores**: Glenn Jocher y equipo de Ultralytics
- **Licencia**: AGPL-3.0 (modelos) / GPL-3.0 (código)
- **Uso en este proyecto**: Detección de vehículos y placas
- **Paper**: [YOLOv8: A New State-of-the-Art Computer Vision Model](https://docs.ultralytics.com/)

## OCR de Placas

### Fast-Plate-OCR
- **Repositorio**: https://github.com/ankandrew/fast-plate-ocr
- **Autor**: [ankandrew](https://github.com/ankandrew)
- **Licencia**: MIT
- **Uso en este proyecto**: Reconocimiento de caracteres en placas
- **Modelo usado**: `cct-s-v2-global-model`

**Características destacadas**:
- Ligero y rápido (optimizado para edge devices)
- Soporta múltiples backends (TensorFlow, PyTorch, JAX)
- Exportable a ONNX, TFLite, CoreML

## Tracking

### SORT (Simple Online and Realtime Tracking)
- **Repositorio**: https://github.com/abewley/sort
- **Autor**: Alex Bewley
- **Licencia**: GPL-3.0
- **Uso en este proyecto**: Seguimiento de vehículos entre frames

## Datasets Referenciados

### CCPD (Chinese City Parking Dataset)
- **Repositorio**: https://github.com/detectRecog/CCPD
- **Paper**: [Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf)
- **Uso**: Referencia para fine-tuning de modelos de placas

### Roboflow Universe
- **URL**: https://universe.roboflow.com/
- **Uso**: Datasets de placas de diversos países para transfer learning

## Agradecimientos Especiales

- Comunidad de Ultralytics por el soporte en issues
- ankandrew por mantener Fast-Plate-OCR actualizado
- A todos los contribuidores de los repos mencionados

---

**Nota**: Si usas este proyecto, considera darle una estrella a los repos originales. 
Ellos hicieron el trabajo pesado. 🌟
