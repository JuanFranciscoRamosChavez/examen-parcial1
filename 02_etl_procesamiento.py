"""
ETL Procesamiento de Datos - PROFECO
=====================================

Script para procesar, limpiar y consolidar los archivos CSV descargados
del programa "Quién es Quién en los Precios" de la PROFECO.

Características:
- Procesamiento en lotes para manejo eficiente de memoria
- Múltiples estrategias de lectura para diferentes formatos
- Limpieza y estandarización de datos
- Combinación segura de archivos en dataset final
- Monitoreo de uso de memoria
"""

import polars as pl
import os
import glob
import shutil
from datetime import datetime
import warnings
import gc

# Configuración de advertencias
warnings.filterwarnings('ignore')

class Config:
    """
    Configuración para el procesamiento de datos con control de memoria.
    
    Attributes:
        ESTRUCTURAS_COLUMNAS (list): Lista de posibles estructuras de columnas
        CODIFICACIONES (list): Codificaciones de archivo a probar
        FILES_PER_BATCH (int): Archivos por lote para procesamiento
        ROWS_PER_CHUNK (int): Filas por chunk para procesamiento
        MAX_MEMORY_CHUNKS (int): Máximo de chunks en memoria
        REPORT_EVERY (int): Frecuencia de reportes de progreso
    """
    
    # Múltiples estructuras posibles para flexibilidad
    ESTRUCTURAS_COLUMNAS = [
        # Estructura 1 (original)
        [
            'producto', 'caracteristicas', 'marca', 'departamento', 'tipo_producto',
            'precio', 'fecha_adquisicion', 'tienda', 'departamento_tienda', 'sucursal',
            'direccion', 'ciudad', 'estado', 'coordenadas_latitud', 'coordenadas_longitud'
        ],
        # Estructura 2 (alternativa común)
        [
            'producto', 'marca', 'precio', 'tienda', 'sucursal', 'ciudad', 'estado'
        ],
        # Estructura 3 (mínima)
        ['producto', 'precio', 'tienda']
    ]
    
    # Codificaciones de archivo a probar
    CODIFICACIONES = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    # Configuración para control de memoria
    FILES_PER_BATCH = 3
    ROWS_PER_CHUNK = 5000
    MAX_MEMORY_CHUNKS = 1
    REPORT_EVERY = 20

def procesar_archivo_memoria_segura(archivo: str, temp_dir: str, file_id: int) -> int:
    """
    Procesa un archivo CSV con múltiples estrategias de lectura y control de memoria.
    
    Args:
        archivo (str): Ruta al archivo CSV a procesar
        temp_dir (str): Directorio temporal para guardar chunks
        file_id (int): Identificador único para el archivo
        
    Returns:
        int: Número de registros procesados exitosamente
    """
    
    print(f"   Leyendo: {os.path.basename(archivo)}")
    
    # Estrategia 1: Intentar detectar estructura automáticamente
    for encoding in Config.CODIFICACIONES:
        try:
            # Leer primeras líneas para diagnóstico
            with open(archivo, 'r', encoding=encoding) as f:
                primeras_lineas = [f.readline().strip() for _ in range(3)]
            
            print(f"      Encoding: {encoding}, Columnas detectadas: {len(primeras_lineas[0].split(','))}")
            
            # Intentar leer con diferentes estructuras de columnas
            for estructura in Config.ESTRUCTURAS_COLUMNAS:
                try:
                    df = pl.read_csv(
                        archivo,
                        encoding=encoding,
                        has_header=False,
                        new_columns=estructura,
                        infer_schema_length=100,
                        ignore_errors=True,
                        low_memory=True,
                        n_threads=1,
                        rechunk=False,
                        skip_rows=0,
                        null_values=['', ' ', 'NA', 'N/A', 'NULL', 'null'],
                        truncate_ragged_lines=True
                    )
                    
                    if len(df) > 0:
                        print(f"      Exito con {len(estructura)} columnas, registros: {len(df)}")
                        
                        # Aplicar limpieza mínima
                        df_limpio = _limpieza_minima(df, estructura)
                        
                        if len(df_limpio) > 0:
                            # Guardar inmediatamente para liberar memoria
                            temp_file = os.path.join(temp_dir, f"chunk_{file_id:06d}.parquet")
                            df_limpio.write_parquet(temp_file, compression='snappy')
                            
                            registros = len(df_limpio)
                            
                            # Liberar memoria explícitamente
                            del df, df_limpio
                            gc.collect()
                            
                            return registros
                        else:
                            print(f"      Datos filtrados completamente")
                    
                except Exception as e_interno:
                    continue
            
        except Exception as e:
            continue
    
    # Estrategia 2: Lectura básica como último recurso
    try:
        print(f"      Intentando lectura de emergencia...")
        df = pl.read_csv(archivo, has_header=True, ignore_errors=True)
        if len(df) > 0:
            # Renombrar columnas genéricas
            new_columns = [f'col_{i}' for i in range(len(df.columns))]
            df = df.rename(dict(zip(df.columns, new_columns)))
            
            # Buscar columnas que parezcan producto, precio, tienda
            producto_col = None
            precio_col = None
            tienda_col = None
            
            for col in df.columns:
                sample_val = str(df[col][0]) if len(df) > 0 else ""
                if any(keyword in sample_val.lower() for keyword in ['producto', 'nombre', 'desc']):
                    producto_col = col
                elif any(keyword in sample_val.lower() for keyword in ['precio', 'price', 'cost']):
                    precio_col = col
                elif any(keyword in sample_val.lower() for keyword in ['tienda', 'store', 'shop']):
                    tienda_col = col
            
            # Si encontramos las columnas clave, procesar
            if producto_col and precio_col:
                temp_file = os.path.join(temp_dir, f"chunk_{file_id:06d}.parquet")
                
                df_limpio = df.select([
                    pl.col(producto_col).alias("producto_clean"),
                    pl.col(precio_col).alias("precio_clean"),
                    pl.col(tienda_col).alias("tienda_clean") if tienda_col else pl.lit("desconocido").alias("tienda_clean")
                ]).with_columns([
                    pl.col("producto_clean").str.to_lowercase().str.strip_chars(),
                    pl.col("precio_clean").cast(pl.Utf8).str.replace_all(r"[^\d\.]", "").cast(pl.Float64),
                    pl.col("tienda_clean").str.to_lowercase().str.strip_chars()
                ]).filter(
                    (pl.col("precio_clean") > 0) & 
                    (pl.col("precio_clean") < 1000000) &
                    (pl.col("producto_clean").is_not_null())
                )
                
                if len(df_limpio) > 0:
                    df_limpio.write_parquet(temp_file, compression='snappy')
                    registros = len(df_limpio)
                    del df, df_limpio
                    gc.collect()
                    return registros
                    
    except Exception as e_emergencia:
        print(f"      Error de emergencia: {str(e_emergencia)[:100]}")
    
    return 0


def _limpieza_minima(df: pl.DataFrame, estructura: list) -> pl.DataFrame:
    """
    Aplica limpieza básica adaptativa según la estructura de columnas.
    
    Args:
        df (pl.DataFrame): DataFrame a limpiar
        estructura (list): Lista de nombres de columnas esperados
        
    Returns:
        pl.DataFrame: DataFrame limpio con columnas estandarizadas
    """
    
    # Determinar índices de columnas clave
    try:
        producto_idx = estructura.index('producto') if 'producto' in estructura else 0
        precio_idx = estructura.index('precio') if 'precio' in estructura else (
            estructura.index('precio_clean') if 'precio_clean' in estructura else 1
        )
        tienda_idx = estructura.index('tienda') if 'tienda' in estructura else (
            estructura.index('tienda_clean') if 'tienda_clean' in estructura else 2
        )
    except:
        # Fallback a columnas por posición
        producto_idx = min(0, len(estructura) - 1)
        precio_idx = min(1, len(estructura) - 1)
        tienda_idx = min(2, len(estructura) - 1)
    
    return (
        df
        .with_columns([
            pl.col(df.columns[producto_idx]).str.to_lowercase().str.strip_chars().alias("producto_clean"),
            pl.col(df.columns[tienda_idx]).str.to_lowercase().str.strip_chars().alias("tienda_clean"),
        ])
        .with_columns([
            pl.col(df.columns[precio_idx])
                .cast(pl.Utf8)
                .str.replace_all(r"[^\d\.]", "")
                .cast(pl.Float64)
                .alias("precio_clean"),
        ])
        .filter(
            (pl.col("precio_clean") > 0) & 
            (pl.col("precio_clean") < 1000000) &
            (pl.col("producto_clean").is_not_null())
        )
        .select(["producto_clean", "tienda_clean", "precio_clean"])
    )

def diagnosticar_archivos(archivos: list):
    """
    Analiza la estructura de los archivos antes del procesamiento completo.
    
    Args:
        archivos (list): Lista de rutas a archivos CSV a diagnosticar
    """
    print("\nDIAGNOSTICO INICIAL DE ARCHIVOS:")
    print("=" * 50)
    
    for i, archivo in enumerate(archivos[:5]):  # Solo primeros 5 para diagnóstico
        print(f"\nArchivo {i+1}: {os.path.basename(archivo)}")
        
        for encoding in Config.CODIFICACIONES:
            try:
                with open(archivo, 'r', encoding=encoding) as f:
                    primera_linea = f.readline().strip()
                    segunda_linea = f.readline().strip()
                
                columnas = primera_linea.split(',')
                print(f"  Encoding {encoding}: {len(columnas)} columnas")
                print(f"  Muestra primera linea: {primera_linea[:100]}...")
                if segunda_linea:
                    print(f"  Muestra datos: {segunda_linea[:100]}...")
                break
                
            except Exception as e:
                continue

def ejecutar_pipeline_ultra_seguro():
    """
    Ejecuta el pipeline completo de ETL con control estricto de memoria.
    
    Procesa archivos en lotes, aplica limpieza y combina en dataset final.
    """
    print("INICIANDO PIPELINE - CON DIAGNOSTICO")
    print("=" * 70)
    
    # Configurar directorios
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    directorio_csv = os.path.join(directorio_script, "archivos_profeco_csv")
    output_dir = os.path.join(directorio_script, "datos_procesados")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(directorio_csv):
        print(f"Directorio no encontrado: {directorio_csv}")
        return
    
    try:
        inicio_total = datetime.now()
        
        # Detectar archivos CSV
        archivos = glob.glob(os.path.join(directorio_csv, "**", "*.csv"), recursive=True)
        if not archivos:
            print("No se encontraron archivos CSV")
            return
        
        print(f"Archivos totales: {len(archivos):,}")
        
        # Ejecutar diagnóstico inicial
        diagnosticar_archivos(archivos)
        
        print("\nIniciando procesamiento ultra-seguro...")
        
        # Procesar en lotes
        temp_dir = os.path.join(output_dir, "temp_chunks")
        os.makedirs(temp_dir, exist_ok=True)
        
        total_registros = 0
        archivos_procesados = 0
        lote_num = 0
        
        for i in range(0, len(archivos), Config.FILES_PER_BATCH):
            lote_num += 1
            lote_archivos = archivos[i:i + Config.FILES_PER_BATCH]
            
            print(f"\nLOTE {lote_num}: {len(lote_archivos)} archivos")
            
            for archivo in lote_archivos:
                try:
                    registros = procesar_archivo_memoria_segura(
                        archivo, temp_dir, archivos_procesados
                    )
                    total_registros += registros
                    archivos_procesados += 1
                    
                    if registros > 0:
                        print(f"      {registros} registros procesados")
                    else:
                        print(f"      0 registros - posible problema de formato")
                    
                    # Reporte periódico de progreso
                    if archivos_procesados % Config.REPORT_EVERY == 0:
                        print(f"   Progreso: {archivos_procesados:,}/{len(archivos):,} archivos")
                        print(f"   {total_registros:,} registros acumulados")
                        gc.collect()
                    
                except Exception as e:
                    print(f"   Error en {os.path.basename(archivo)}: {str(e)[:100]}...")
                    continue
            
            # Pausa estratégica para liberar memoria
            if lote_num % 5 == 0:
                print(f"   Pausa estrategica para liberar memoria...")
                gc.collect()
        
        # Verificar si hay datos antes de combinar
        chunks = glob.glob(os.path.join(temp_dir, "*.parquet"))
        
        if not chunks:
            print(f"\nADVERTENCIA: No se generaron chunks validos")
            print("Posibles soluciones:")
            print("   1. Verificar que los archivos CSV tengan datos")
            print("   2. Revisar el formato de columnas")
            print("   3. Verificar los encodings de archivos")
            return
        
        # Combinación final de chunks
        print(f"\nCOMBINANDO {len(chunks):,} CHUNKS...")
        archivo_final = _combinacion_segura(temp_dir, output_dir, total_registros)
        
        # Limpiar directorio temporal
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        tiempo_total = datetime.now() - inicio_total
        print(f"\nPROCESAMIENTO COMPLETADO EN: {tiempo_total}")
        print(f"REGISTROS TOTALES: {total_registros:,}")
        print(f"ARCHIVO FINAL: {archivo_final}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


def _combinacion_segura(temp_dir: str, output_dir: str, total_registros: int) -> str:
    """
    Combina chunks de datos de forma segura para alta utilización de memoria.
    
    Args:
        temp_dir (str): Directorio con chunks temporales
        output_dir (str): Directorio de salida para archivo final
        total_registros (int): Total de registros procesados
        
    Returns:
        str: Ruta al archivo final combinado
        
    Raises:
        ValueError: Si no hay chunks para combinar
    """
    chunks = glob.glob(os.path.join(temp_dir, "*.parquet"))
    
    if not chunks:
        raise ValueError("No hay chunks para combinar")
    
    print(f"Combinando {len(chunks):,} chunks...")
    
    # Estrategia: combinar en grupos pequeños
    grupos = [chunks[i:i + 10] for i in range(0, len(chunks), 10)]
    archivo_temp_actual = None
    
    for i, grupo in enumerate(grupos, 1):
        print(f"   Grupo {i}/{len(grupos)}: {len(grupo)} chunks")
        
        # Cargar y combinar grupo
        dfs_grupo = []
        for chunk in grupo:
            try:
                df_chunk = pl.read_parquet(chunk)
                dfs_grupo.append(df_chunk)
            except Exception as e:
                print(f"      Error cargando {chunk}: {e}")
                continue
        
        if not dfs_grupo:
            continue
            
        df_grupo_combinado = pl.concat(dfs_grupo)
        
        # Guardar resultado temporal
        archivo_temp_nuevo = os.path.join(output_dir, f"merge_temp_{i}.parquet")
        df_grupo_combinado.write_parquet(archivo_temp_nuevo, compression='snappy')
        
        # Limpiar archivo temporal anterior
        if archivo_temp_actual and os.path.exists(archivo_temp_actual):
            os.remove(archivo_temp_actual)
        
        archivo_temp_actual = archivo_temp_nuevo
        
        # Liberar memoria
        del dfs_grupo, df_grupo_combinado
        gc.collect()
    
    # Crear archivo final
    archivo_final = os.path.join(output_dir, "datos_profeco_final.parquet")
    if archivo_temp_actual:
        shutil.move(archivo_temp_actual, archivo_final)
    else:
        # Fallback: usar primer grupo
        shutil.move(grupos[0][0], archivo_final)
    
    return archivo_final


def monitorear_memoria():
    """
    Monitorea el uso de memoria del sistema.
    
    Returns:
        bool: True si el uso de memoria es crítico, False en caso contrario
    """
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"MONITOR RAM: {ram.percent}% usado")
        
        if ram.percent > 85:
            print("ALTO USO DE RAM - CONSIDERAR PAUSA")
            return True
        elif ram.percent > 90:
            print("CRITICO USO DE RAM - PAUSA RECOMENDADA")
            return True
            
        return False
    except:
        return False


if __name__ == "__main__":
    print("VERIFICANDO ESTADO INICIAL...")
    if monitorear_memoria():
        print("Alta RAM detectada - Ejecutando modo seguro")

    ejecutar_pipeline_ultra_seguro()