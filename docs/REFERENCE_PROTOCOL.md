# Protocolo de interpretación de la muestra de referencia 2023→2024

**Objetivo.** Estimar sin sesgo, con un diseño probabilístico (Olofsson et al., 2014), qué parte del cambio de grupo de cobertura que el CDL muestra entre 2023 y 2024 es cambio real. La muestra la genera `scripts/07a_reference_sample.py` con los parámetros de `config/experiment.yaml` (`reference_sample`, semilla 42). **La interpretación tiene que hacerla una persona.** No se puede automatizar sin introducir de nuevo el error que se quiere medir.

## Ficheros

| Fichero | Contenido |
|---|---|
| `results/reference_sample/sample_2023_2024.csv` | 575 puntos (centro de píxel de 30 m). Por cada punto: `point_id`, estrato, coordenadas EPSG:5070 y WGS84, condado (`GEOID`), clases CDL de 2023 y 2024, y columnas vacías para la interpretación |
| `results/reference_sample/strata_2023_2024.csv` | Tamaño de cada estrato en CONUS (N_s, píxeles), candidatos en las 800 teselas elegidas (M_s) y tamaño muestral (n_s) |

## Diseño

1. **Estratos.** Se definen por la transición de grupo en el producto NASS 30 m: cultivo estable, pastizal estable, otros estables, cultivo→pastizal, pastizal→cultivo, bosque/matorral→pastizal y otros cambios. Los estratos de cambio, que son raros, están sobremuestreados.
2. **Primera etapa.** Muestreo aleatorio simple de 800 teselas de 512 × 512 píxeles de 30 m, sobre la partición nacional y entre las teselas con superficie de CONUS.
3. **Segunda etapa.** Muestreo aleatorio simple de n_s píxeles entre todos los píxeles del estrato en esas teselas, mediante un reservorio de claves aleatorias. Dentro de cada estrato, todos los píxeles tienen la misma probabilidad de inclusión.

## Qué se interpreta en cada punto

Para cada año (2023 y 2024), el **grupo de cobertura real** del píxel de 30 m centrado en las coordenadas. Se usan los mismos siete grupos del artículo: `crop`, `grass_pasture`, `forest_shrub`, `developed`, `water`, `wetland`, `barren_other`. El heno, la alfalfa y el barbecho cuentan como `crop`, igual que en el artículo (Apéndice A). La pradera permanente no labrada cuenta como `grass_pasture`.

- `ref_group_2023`, `ref_group_2024`: grupo de cada año. Si el píxel cae sobre una mezcla, se asigna el grupo que ocupa más de la mitad del píxel.
- `ref_confidence`: `high`, `medium` o `low`.
- `imagery_used`: fuentes consultadas, por ejemplo `NAIP2023; NAIP2024; S2-NDVI-2024`.
- `interpreter`: iniciales.
- `notes`: lo que justifique la decisión (labrado visible, siega, rotación, etc.).

## Fuentes de imagen recomendadas

1. **NAIP**, ortofoto de 0,6–1 m del USDA. Cada estado se vuela cada 2 o 3 años, así que no siempre habrá NAIP de 2023 y de 2024: se anota qué año se ha usado.
2. **Series temporales de Sentinel-2 y Landsat** (NDVI mensual o perfiles en Copernicus Browser, o HLS): distinguen un cultivo anual (siembra y cosecha) de la pradera permanente (verde más largo y sin labrado).
3. **Imágenes históricas de Google Earth Pro**, como apoyo.

No se consulta el CDL de ningún año ni otros mapas derivados del CDL (CSB, CropScape) mientras se interpreta; es decir, se interpreta "a ciegas". Las columnas `cdl_2023` y `cdl_2024` se ocultan en la hoja de trabajo.

## Control de calidad

- Al menos dos intérpretes independientes en un 20 % de los puntos elegido al azar, para estimar el acuerdo (kappa) y revisar los desacuerdos.
- Los puntos de confianza `low` se revisan entre los dos intérpretes antes del análisis.

## Análisis previsto, cuando la hoja esté completa

- Matriz de error estratificada entre el cambio cartografiado y el cambio de referencia.
- Estimador de área de Olofsson et al. (2014) con los pesos W_s = N_s / ΣN, e intervalos de confianza.
- Estimandos:
  - la superficie de cambio real de cultivo a pastizal (y a la inversa) entre 2023 y 2024;
  - la proporción del cambio cartografiado que es falso en cada estrato.

Estos resultados sustituirían a la frase de limitaciones del artículo y permitirían una afirmación cuantitativa sobre el cambio falso.
