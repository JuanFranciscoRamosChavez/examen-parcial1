"""
Web Scraper para Datos Históricos de Precios - PROFECO
=====================================================

Script para descargar y extraer automáticamente las bases de datos históricas
del programa "Quién es Quién en los Precios" de la PROFECO.

Características:
- Descarga todos los archivos RAR disponibles en la página oficial
- Extrae los archivos CSV a una carpeta unificada
- Elimina los archivos RAR temporales después de la extracción
- Incluye barra de progreso para monitorear descargas
- Manejo robusto de errores y verificaciones
- Reintentos automáticos en caso de fallos de descarga
"""

import os
import requests
import rarfile
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

# Configuración global
URL_BASE = "https://datos.profeco.gob.mx/datos_abiertos/qqp.php"
DIRECTORIO_SALIDA = "archivos_profeco_csv"
MAX_REINTENTOS = 3
TIEMPO_ESPERA = 5  # segundos entre reintentos

def descargar_archivo(url_descarga, ruta_destino):
    """
    Descarga un archivo mostrando una barra de progreso.
    
    Args:
        url_descarga (str): URL del archivo a descargar
        ruta_destino (str): Ruta local donde guardar el archivo
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    try:
        file_response = requests.get(url_descarga, stream=True)
        file_response.raise_for_status()
        total_size = int(file_response.headers.get('content-length', 0))
        
        # Barra de progreso para la descarga
        with tqdm(total=total_size, unit='B', unit_scale=True, 
                 desc=os.path.basename(ruta_destino), leave=False) as progress_bar:
            with open(ruta_destino, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error en la descarga: {e}")
        return False

def descargar_con_reintentos(url_descarga, ruta_destino, max_reintentos=MAX_REINTENTOS):
    """
    Descarga con múltiples reintentos en caso de error.
    
    Args:
        url_descarga (str): URL del archivo a descargar
        ruta_destino (str): Ruta local donde guardar el archivo
        max_reintentos (int): Número máximo de reintentos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    for intento in range(max_reintentos):
        try:
            print(f"Intento {intento + 1} de {max_reintentos}...")
            if descargar_archivo(url_descarga, ruta_destino):
                return True
        except Exception as e:
            print(f"Intento {intento + 1} fallido: {e}")
            if intento < max_reintentos - 1:
                print(f"Reintentando en {TIEMPO_ESPERA} segundos...")
                time.sleep(TIEMPO_ESPERA)
    return False

def verificar_integridad_archivo(ruta_archivo):
    """
    Verifica si el archivo RAR está completo y no corrupto.
    
    Args:
        ruta_archivo (str): Ruta al archivo RAR a verificar
        
    Returns:
        bool: True si el archivo es válido, False si está corrupto
    """
    try:
        # Verificar que el archivo tenga tamaño mínimo razonable
        tamaño = os.path.getsize(ruta_archivo)
        if tamaño < 1000:  # Menos de 1KB probablemente está corrupto
            print(f"Archivo demasiado pequeño ({tamaño} bytes) - probablemente corrupto")
            return False
            
        # Intentar abrir el archivo RAR para verificar integridad
        with rarfile.RarFile(ruta_archivo, 'r') as rf:
            # Verificar que se puede leer la lista de archivos
            rf.namelist()
        return True
    except Exception as e:
        print(f"Archivo corrupto o incompleto: {e}")
        # Eliminar archivo corrupto
        if os.path.exists(ruta_archivo):
            os.remove(ruta_archivo)
        return False

def main():
    """Función principal que ejecuta el proceso completo de scraping."""
    
    # Crear el directorio principal si no existe
    if not os.path.exists(DIRECTORIO_SALIDA):
        os.makedirs(DIRECTORIO_SALIDA)
        print(f"Directorio creado: {DIRECTORIO_SALIDA}")

    try:
        print(f"Obteniendo información de: {URL_BASE}")
        response = requests.get(URL_BASE)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        links_container = soup.find('div', class_='form-group')

        if links_container:
            file_links = links_container.find_all('a')
            print(f"Se encontraron {len(file_links)} archivos. Iniciando proceso...")

            # Iterar sobre cada enlace de descarga
            for link in file_links:
                relative_url = link.get('href')
                absolute_url = urljoin(URL_BASE, relative_url)
                
                nombre_base = link.text.strip()
                nombre_rar = f"{nombre_base}.rar"
                ruta_rar = os.path.join(DIRECTORIO_SALIDA, nombre_rar)

                # 1. DESCARGAR CON REINTENTOS
                if not os.path.exists(ruta_rar):
                    print(f"\nDescargando '{nombre_rar}'...")
                    
                    if descargar_con_reintentos(absolute_url, ruta_rar):
                        print(f"'{nombre_rar}' descargado correctamente.")
                    else:
                        print(f"ERROR: No se pudo descargar '{nombre_rar}' después de {MAX_REINTENTOS} intentos")
                        continue
                else:
                    print(f"\nEl archivo '{nombre_rar}' ya existe. Omitiendo descarga.")

                # 2. VERIFICAR INTEGRIDAD ANTES DE EXTRAER
                if not verificar_integridad_archivo(ruta_rar):
                    print(f"Archivo corrupto, saltando: '{nombre_rar}'")
                    continue

                # 3. EXTRAER EN LA CARPETA PRINCIPAL
                print(f"Descomprimiendo '{nombre_rar}'...")
                try:
                    with rarfile.RarFile(ruta_rar, 'r') as rf:
                        # Extrae todo directamente en el directorio de salida único
                        rf.extractall(path=DIRECTORIO_SALIDA) 
                    
                    print(f"Archivos CSV extraídos en: '{DIRECTORIO_SALIDA}/'")
                    
                    # 4. ELIMINAR EL ARCHIVO .RAR TEMPORAL
                    os.remove(ruta_rar)
                    print(f"Archivo temporal '{nombre_rar}' eliminado.")

                except Exception as e:
                    print(f"Error al descomprimir '{nombre_rar}': {e}")
                    # Si falla la extracción, eliminar el archivo corrupto
                    if os.path.exists(ruta_rar):
                        os.remove(ruta_rar)

        else:
            print("No se pudo encontrar el contenedor de los enlaces.")

        print("\n¡Proceso completado! Todos los archivos CSV están en la carpeta 'archivos_profeco_csv'.")

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la página: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()