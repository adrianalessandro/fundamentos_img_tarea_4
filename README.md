# Fundamentos de procesamiento de imágenes - Tarea 4

Ejecutar los comandos desde la raíz del repositorio. 

**Importante**, en la raiz del repo debe estar la carpeta de archivos_t4/, no logre incluir esta carpeta al repositorio dado errores de commit que no logre resolver.

## Pregunta 1: procesamiento multiescala

Consolidado en [`pregunta_1/`](./pregunta_1/).

- [`main.py`](./pregunta_1/main.py): carga la imagen, ejecuta los experimentos
  y guarda las figuras.
- [`funciones.py`](./pregunta_1/funciones.py): pirámide Laplaciana, Starlet,
  umbralización y operaciones relacionadas.
- [`experimentaciones.py`](./pregunta_1/experimentaciones.py): reconstrucción,
  métricas y experimentos.
- [`figuras.py`](./pregunta_1/figuras.py): generación y guardado de figuras.
- [`preguntas_guiadas_1.md`](./preguntas_guiadas_1.md): respuestas a las
  preguntas guiadas.

Requiere `archivos_t4/memorial.tif`. Para ejecutar desde la raíz:

```bash
python -m pregunta_1.main
```

Las figuras se guardan en [`pregunta_1/output/`](./pregunta_1/output/).

## Pregunta 2: mapeo de tonos HDR

Consolidado en [`pregunta_2/`](./pregunta_2/).

- [`main.py`](./pregunta_2/main.py): carga las imágenes, ejecuta los
  experimentos y guarda las figuras.
- [`funciones.py`](./pregunta_2/funciones.py): filtros, transformaciones de
  color y métricas.
- [`experimentaciones.py`](./pregunta_2/experimentaciones.py): métodos de
  mapeo de tonos y estrategias de reconstrucción de color.
- [`figuras.py`](./pregunta_2/figuras.py): generación y guardado de figuras.
- [`preguntas_guiadas_2.md`](./preguntas_guiadas_2.md): respuestas a las
  preguntas guiadas.

Requiere `archivos_t4/memorial.tif` y `archivos_t4/belgium.tif`. Para ejecutar
desde la raíz:

```bash
python -m pregunta_2.main
```

Las figuras se guardan en [`pregunta_2/output/`](./pregunta_2/output/).

## Dependencias

Se requiere Python y los paquetes `numpy`, `scipy`, `matplotlib` e `imageio`.
Para el filtro bilateral de la pregunta 2, se requiere además `opencv-python`
o bien `scikit-image` (fall-back).
