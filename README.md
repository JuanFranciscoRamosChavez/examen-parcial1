# Análisis de Datos: Quién es Quién en los Precios (PROFECO)

Este Examen parcial es un sistema completo de análisis de datos diseñado para procesar y analizar las bases de datos históricas del programa "Quién es Quién en los Precios" de la PROFECO. El sistema utiliza técnicas de web scraping, procesamiento ETL y análisis exploratorio para extraer insights valiosos sobre el comportamiento de precios en el mercado mexicano.

## Características Principales

- **Web Scraping Automatizado**: Descarga automática de todos los archivos históricos de la PROFECO con manejo robusto de errores y reintentos.
- **Procesamiento ETL Eficiente**: Limpieza, unificación y homologación de datos usando Polars para optimización de memoria.
- **Análisis Exploratorio Completo**: Incluye análisis univariado, bivariado, multivariado y temporal de los datos de precios.
- **Visualizaciones **: Generación automática de gráficos 
- **Manejo de Grandes Volúmenes**: Procesamiento en lotes y control estricto de uso de memoria.

## Objetivos del Proyecto

1. **Acceder a la Fuente de Datos**
   - Ingresar a la ruta oficial de datos abiertos de PROFECO: https://datos.profeco.gob.mx/datos_abiertos/qqp.php

2. **Descarga de Archivos**
   - Utilizar técnicas de web scraping para descargar todos los archivos de datos históricos disponibles en la página.

3. **Procesamiento de Datos (ETL)**
   - Realizar la limpieza, unificación y homologación de todos los conjuntos de datos para asegurar su consistencia y calidad.

4. **Análisis Exploratorio de Datos (EDA)**
   - Ejecutar un análisis detallado que incluya:
     - **Análisis Univariado**: Examinar cada variable de forma individual para entender su distribución.
     - **Análisis Bivariado**: Estudiar la relación entre pares de variables para identificar correlaciones.
     - **Análisis Multivariado**: Investigar las interacciones entre tres o más variables para descubrir patrones complejos.

## Flujo del Proyecto y Estructura de Archivos

El proyecto se divide en tres fases principales: **recolección de datos**, **procesamiento ETL** y **análisis exploratorio**.

### Diagrama del Pipeline de Datos

```
[Web Scraping] -> 01_web_scraping.py -> [archivos_profeco_csv/]
|
v
[Procesamiento ETL] -> 02_etl_procesamiento.py -> [datos_procesados/]
|
v
[Análisis Exploratorio] -> 03_analisis_exploratorio.py -> [analisis_imagenes/]
```


### Descripción de Archivos

- **Scripts del Pipeline de Datos**:
  - `01_web_scraping.py`: Extrae y descarga todos los archivos históricos de la página de PROFECO, descomprime los RAR y consolida los CSV en carpetas respectivamente sus años.
  - `02_etl_procesamiento.py`: Realiza limpieza, transformación y unificación de todos los datasets usando procesamiento en lotes.
  - `03_analisis_exploratorio.py`: Ejecuta análisis estadístico completo y genera visualizaciones.

- **Directorios Generados**:
  - `archivos_profeco_csv/`: Contiene todos los archivos CSV descargados y extraídos.
  - `datos_procesados/`: Almacena el dataset consolidado y limpio listo para análisis.
  - `analisis_imagenes/`: Contiene todas las visualizaciones generadas por el análisis.

- **Configuración**:
  - `requirements.txt`: Lista de dependencias de Python necesarias.

## Scraper de Datos Históricos de Precios - PROFECO

El objetivo es automatizar la recolección de todos los archivos CSV en una única carpeta, listos para su análisis.

### Descripción del Funcionamiento

El script (`01_web_scraping.py`) realiza las siguientes tareas de forma automatizada:

1. **Conexión y Rastreo**: Se conecta a la página de datos abiertos de la PROFECO.
2. **Identificación de Enlaces**: Analiza el contenido HTML de la página para encontrar todos los enlaces de descarga de las bases de datos.
3. **Descarga Inteligente**: Para cada enlace, comprueba si el archivo (`.rar`) ya ha sido descargado. Si no existe, lo descarga mostrando una barra de progreso.
4. **Extracción Centralizada**: Descomprime el contenido de cada archivo `.rar` (que son archivos `.csv`) en una única carpeta de salida.
5. **Limpieza Automática**: Una vez que los archivos CSV han sido extraídos, el script elimina el archivo `.rar` temporal para ahorrar espacio en disco.

El script (`02_etl_procesamiento.py`) realiza el procesamiento ETL (Extract, Transform, Load) para limpiar, transformar y consolidar los archivos CSV descargados.

1. **Carga Unificada**: Lee y combina todos los archivos CSV descargados en un único DataFrame utilizando Polars para máximo rendimiento.
3. **Limpieza Inteligente de Texto**:
   - Normaliza nombres de productos y tiendas (minúsculas, elimina espacios extras).
   - Elimina caracteres especiales y unifica formatos.
5. **Procesamiento de Precios**:
   - Detecta y elimina valores atípicos extremos.
   - Filtra precios dentro de rangos razonables.
   - Identifica y maneja valores missing.
7. **Enriquecimiento de Datos**:
   - Calcula métricas derivadas como longitud de nombres.
   - Identifica productos con unidades de medida.
9. **Exportación Optimizada**: Guarda el dataset limpio en formato Parquet para análisis posteriores.

El script (`03_analisis_exploratorio.py`) realiza un análisis exploratorio completo en tres fases:

1. **Análisis Básicos**:
   - Univariado: Distribución de productos, tiendas y precios.
   - Bivariado: Relación precio-tienda y productos estratégicos.
   - Identificación de productos con mayor variación de precios.
3. **Análisis Multivariado**:
   - Segmentación de Mercado: Clasificación en Económico, Medio y Premium.
   - Análisis de Valor: Identificación de productos estratégicos por valor y cobertura.
   - Distribución Detallada: Análisis estadístico completo de precios.
4. **Análisis Avanzados**:
   - Evolución Temporal: Análisis de volatilidad y tendencias de precios.
   - Visualización Integral: Generación de gráficos.
   - Resumen Ejecutivo: Hallazgos clave y recomendaciones estratégicas.

## Cómo Ejecutar el Proyecto

La manera recomendada de ejecutar la aplicación es de forma local usando un entorno virtual.

### Requisitos Previos

- **Python 3.11.9**: Descargar desde: https://www.python.org/downloads/release/python-3119/

### Ejecución Local

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/JuanFranciscoRamosChavez/examen-parcial1.git
   cd proyecto-profeco
   ```
   
2. **Crear y activar un entorno virtual**:
    ```bash
   # Crear el entorno en Windows
   py -3.11 -m venv profeco_env

   # Crear el entorno en macOS/Linux
   python3.11 -m venv profeco_env

   # Activar en Windows
   profeco_env\Scripts\activate

   # Activar en macOS/Linux
   source profeco_env/bin/activate
    ```
    
3.  **Instalar las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar el pipeline de datos**:
    ```bash
      Fase 1: Descarga de datos
      python 01_web_scraping.py
   
      Fase 2: Procesamiento y limpieza
      python 02_etl_procesamiento.py

      Fase 3: Análisis exploratorio
      python 03_analisis_exploratorio.py
    ```
   
##  Tecnologías Utilizadas

-   **Python**
-   **Streamlit**: Para la interfaz de usuario.
-   **Hugging Face Transformers**: Para los modelos de búsqueda semántica y resumen de texto.
-   **spaCy**: Para el procesamiento de lenguaje natural en el pipeline.
-   **Selenium**: Para el web scraping.
-   **Pandas**: Para la manipulación de datos.


## Estructura de Análisis Incluida
- **Análisis Univariado**
   - Distribución de productos y frecuencias
   - Análisis de tiendas por volumen y precios
   - Estadísticas descriptivas de precios
   - Histogramas y diagramas de caja

- **Análisis Bivariado**
   - Relación entre precios y tiendas
   - Identificación de productos con mayor variación de precios
   - Boxplots de distribución de precios por tienda
   - Análisis de correlaciones

- **Análisis Multivariado**
   - Segmentación de mercado (Económico/Medio/Premium)
   - Matriz de valor vs cobertura
   - Identificación de productos estratégicos
   - Análisis de componentes principales

- **Análisis Temporal**
   - Evolución y volatilidad de precios
   - Productos con mayor variación temporal
   - Análisis de estacionalidad
   - Tendencias y patrones temporales

## Tecnologías Utilizadas
- **Python 3.11**: Lenguaje de programación principal

- **Polars**: Procesamiento eficiente de datos

- **BeautifulSoup4**: Web scraping y parsing HTML

- **Requests**: Descarga de archivos HTTP

- **Matplotlib/Seaborn**: Visualizaciones y gráficos

- **Pandas**: Manipulación de datos adicional

- **Scipy**: Análisis estadístico avanzado

- **Rarfile**: Manejo de archivos comprimidos

- **TQDM**: Barras de progreso para monitoreo

## Resultados Esperados
- Dataset consolidado con todos los datos históricos de PROFECO
- 15+ visualizaciones profesionales en formato PNG
- Análisis completo de distribución de precios y productos
- Segmentación del mercado mexicano en categorías de precio
- Identificación de productos estratégicos y patrones de precios
- Reportes ejecutivos con hallazgos principales
