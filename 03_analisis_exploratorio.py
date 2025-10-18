"""
ANÁLISIS EXPLORATORIO COMPLETO - SISTEMA DE PRECIOS PROFECO
===========================================================
Script unificado para análisis exploratorio de datos del dataset
"Quién es Quién en los Precios" de PROFECO.

Análisis incluidos:
- Univariado: Productos, Tiendas, Precios
- Bivariado: Precio vs Tienda, Productos Estratégicos  
- Multivariado: Segmentación de mercado, Análisis de valor
- Temporal: Evolución y volatilidad de precios
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
import warnings

# Configuración de entorno
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuración de visualización
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def configurar_entorno():
    """
    Configura el entorno de trabajo creando directorios necesarios
    y cargando los datos.
    
    Returns:
        pl.DataFrame: DataFrame con los datos de Profeco
    """
    # Crear directorio para imágenes
    directorio_imagenes = "analisis_imagenes"
    if not os.path.exists(directorio_imagenes):
        os.makedirs(directorio_imagenes)
        print(f"Directorio creado: {directorio_imagenes}")
    
    # Cargar datos
    try:
        df = pl.read_parquet("datos_procesados/datos_profeco_final.parquet")
        print(f"Datos cargados correctamente: {len(df):,} registros")
        print(f"Columnas disponibles: {df.columns}")
        
        # Mostrar información básica del dataset
        print(f"\nRESUMEN INICIAL DEL DATASET:")
        print(f"Total de productos únicos: {df['producto_clean'].n_unique():,}")
        print(f"Total de tiendas únicas: {df['tienda_clean'].n_unique():,}")
        print(f"Rango de precios: ${df['precio_clean'].min():.2f} - ${df['precio_clean'].max():.2f}")
        print(f"Precio promedio: ${df['precio_clean'].mean():.2f}")
        print(f"Precio mediano: ${df['precio_clean'].median():.2f}")
        
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def mostrar_resumen_ejecutivo(df):
    """
    Muestra un resumen ejecutivo con los hallazgos más importantes.
    
    Args:
        df (pl.DataFrame): DataFrame con los datos
    """
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO - HALLAZGOS PRINCIPALES")
    print("="*80)
    
    # Estadísticas clave
    total_productos = len(df)
    productos_unicos = df['producto_clean'].n_unique()
    tiendas_unicas = df['tienda_clean'].n_unique()
    precio_promedio = df['precio_clean'].mean()
    precio_mediano = df['precio_clean'].median()
    
    # Productos más comunes
    top_productos = (df.group_by("producto_clean")
                     .agg(pl.count().alias("frecuencia"))
                     .sort("frecuencia", descending=True)
                     .head(5))
    
    # Tiendas más grandes
    top_tiendas = (df.group_by("tienda_clean")
                   .agg(pl.count().alias("total_productos"))
                   .sort("total_productos", descending=True)
                   .head(5))
    
    print(f"\nMETRICAS CLAVE:")
    print(f"   Total de registros: {total_productos:,}")
    print(f"   Productos únicos: {productos_unicos:,}")
    print(f"   Tiendas únicas: {tiendas_unicas:,}")
    print(f"   Precio promedio: ${precio_promedio:.2f}")
    print(f"   Precio mediano: ${precio_mediano:.2f}")
    
    print(f"\nTOP 5 PRODUCTOS MAS FRECUENTES:")
    for i, (producto, freq) in enumerate(top_productos.rows(), 1):
        print(f"   {i}. {producto[:35]:35} ({freq:,} registros)")
    
    print(f"\nTOP 5 TIENDAS POR VOLUMEN:")
    for i, (tienda, total) in enumerate(top_tiendas.rows(), 1):
        print(f"   {i}. {tienda[:30]:30} ({total:,} productos)")

def analisis_univariado_productos(df):
    """
    Realiza análisis univariado de la variable producto.
    
    Args:
        df (pl.DataFrame): DataFrame con los datos de productos
    """
    print("\n" + "="*60)
    print("ANALISIS UNIVARIADO - PRODUCTOS")
    print("="*60)
    
    # Top productos más frecuentes
    top_productos = (df.group_by("producto_clean")
                     .agg(pl.count().alias("frecuencia"))
                     .sort("frecuencia", descending=True)
                     .head(20))
    
    print("Top 20 productos mas frecuentes:")
    for i, (producto, freq) in enumerate(top_productos.rows(), 1):
        print(f"   {i:2d}. {producto[:40]:40} {freq:,} registros")
    
    # Estadísticas de distribución
    stats_productos = (df.group_by("producto_clean")
                       .agg(pl.count().alias("freq"))
                       .select(pl.col("freq")))
    
    freq_data = stats_productos["freq"].to_numpy()
    
    print(f"\nESTADISTICAS DE DISTRIBUCION:")
    print(f"   Productos con solo 1 registro: {np.sum(freq_data == 1):,}")
    print(f"   Productos con 2-10 registros: {np.sum((freq_data >= 2) & (freq_data <= 10)):,}")
    print(f"   Productos con 11-100 registros: {np.sum((freq_data >= 11) & (freq_data <= 100)):,}")
    print(f"   Productos con mas de 100 registros: {np.sum(freq_data > 100):,}")
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma de frecuencias (escala logarítmica)
    ax1.hist(np.log1p(freq_data), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_title('Distribucion de Frecuencia de Productos (log)')
    ax1.set_xlabel('log(Frecuencia + 1)')
    ax1.set_ylabel('Numero de Productos')
    ax1.grid(True, alpha=0.3)
    
    # Percentiles de frecuencia
    percentiles = np.percentile(freq_data, [25, 50, 75, 90, 95, 99])
    ax2.bar(range(len(percentiles)), percentiles, color='lightcoral', alpha=0.7)
    ax2.set_title('Percentiles de Frecuencia de Productos')
    ax2.set_xlabel('Percentil')
    ax2.set_ylabel('Frecuencia')
    ax2.set_xticks(range(len(percentiles)))
    ax2.set_xticklabels(['25%', '50%', '75%', '90%', '95%', '99%'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_imagenes/distribucion_frecuencia_productos.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Imagen guardada: analisis_imagenes/distribucion_frecuencia_productos.png")

def analisis_univariado_tiendas(df):
    """
    Realiza análisis univariado de la variable tienda.
    
    Args:
        df (pl.DataFrame): DataFrame con los datos de tiendas
    """
    print("\n" + "="*60)
    print("ANALISIS UNIVARIADO - TIENDAS")
    print("="*60)
    
    # Top tiendas por volumen
    top_tiendas = (df.group_by("tienda_clean")
                   .agg([
                       pl.count().alias("total_productos"),
                       pl.col("precio_clean").mean().alias("precio_promedio"),
                       pl.col("precio_clean").median().alias("precio_mediano")
                   ])
                   .sort("total_productos", descending=True)
                   .head(15))
    
    print("Top 15 tiendas por volumen:")
    for i, (tienda, total, precio_prom, precio_med) in enumerate(top_tiendas.rows(), 1):
        print(f"   {i:2d}. {tienda[:30]:30} {total:>8,} productos | ${precio_prom:.2f} prom | ${precio_med:.2f} med")
    
    # Estadísticas de distribución de tiendas
    stats_tiendas = (df.group_by("tienda_clean")
                     .agg(pl.count().alias("total_productos")))
    
    total_data = stats_tiendas["total_productos"].to_numpy()
    
    print(f"\nESTADISTICAS DE DISTRIBUCION POR TIENDA:")
    print(f"   Tiendas con menos de 100 productos: {np.sum(total_data < 100):,}")
    print(f"   Tiendas con 100-1,000 productos: {np.sum((total_data >= 100) & (total_data < 1000)):,}")
    print(f"   Tiendas con 1,000-10,000 productos: {np.sum((total_data >= 1000) & (total_data < 10000)):,}")
    print(f"   Tiendas con mas de 10,000 productos: {np.sum(total_data >= 10000):,}")
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    tiendas = [t[:20] for t in top_tiendas["tienda_clean"].to_list()]
    volumen = top_tiendas["total_productos"].to_list()
    precios_promedio = top_tiendas["precio_promedio"].to_list()
    
    # Gráfico de volumen
    bars1 = ax1.barh(tiendas, volumen, color='lightgreen', alpha=0.7)
    ax1.set_title('Top Tiendas por Volumen de Productos')
    ax1.set_xlabel('Numero de Productos')
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, valor in zip(bars1, volumen):
        ax1.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2, 
                f'{valor:,}', ha='left', va='center', fontsize=9)
    
    # Gráfico de precios promedio
    bars2 = ax2.barh(tiendas, precios_promedio, color='lightcoral', alpha=0.7)
    ax2.set_title('Precio Promedio por Tienda')
    ax2.set_xlabel('Precio Promedio ($)')
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, valor in zip(bars2, precios_promedio):
        ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'${valor:.0f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('analisis_imagenes/analisis_tiendas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Imagen guardada: analisis_imagenes/analisis_tiendas.png")

def analisis_bivariado_precio_tienda(df):
    """
    Analiza la relación entre precios y tiendas.
    
    Args:
        df (pl.DataFrame): DataFrame con los datos
    """
    print("\n" + "="*60)
    print("ANÁLISIS BIVARIADO - PRECIO VS TIENDA")
    print("="*60)
    
    # Precios promedio por tienda
    precios_por_tienda = (df.group_by("tienda_clean")
                          .agg([
                              pl.col("precio_clean").mean().alias("precio_promedio"),
                              pl.col("precio_clean").std().alias("precio_std"),
                              pl.count().alias("n_productos")
                          ])
                          .filter(pl.col("n_productos") > 1000)
                          .sort("precio_promedio", descending=True)
                          .head(10))
    
    print("Precios promedio por tienda (más de 1000 productos):")
    for i, (tienda, promedio, std, n) in enumerate(precios_por_tienda.rows(), 1):
        print(f"   {i:2d}. {tienda[:25]:25} ${promedio:>6.2f} ± ${std:.2f} (n={n:,})")
    
    # Boxplot de precios por tienda
    top_tiendas_list = precios_por_tienda["tienda_clean"].head(8).to_list()
    datos_boxplot = []
    etiquetas = []
    
    for tienda in top_tiendas_list:
        precios_tienda = (df.filter(pl.col("tienda_clean") == tienda)
                         .filter(pl.col("precio_clean") <= 500)
                         ["precio_clean"].to_numpy())
        if len(precios_tienda) > 0:
            datos_boxplot.append(precios_tienda)
            etiquetas.append(tienda[:15])
    
    plt.figure(figsize=(14, 8))
    box_plot = plt.boxplot(datos_boxplot, labels=etiquetas, patch_artist=True)
    
    # Colorear las cajas
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 
              'lightyellow', 'lightcyan', 'lavender', 'peachpuff']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Distribución de Precios por Tienda', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Precio ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('analisis_imagenes/boxplot_precios_tiendas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Imagen guardada: analisis_imagenes/boxplot_precios_tiendas.png")

def analisis_productos_estrategicos(df):
    """
    Identifica productos con mayor variación de precios entre tiendas.
    
    Args:
        df (pl.DataFrame): DataFrame con los datos
    """
    print("\n" + "="*60)
    print("ANALISIS DE PRODUCTOS ESTRATEGICOS")
    print("="*60)
    
    # Productos presentes en múltiples tiendas
    productos_comparables = (df.group_by("producto_clean")
                            .agg(pl.col("tienda_clean").n_unique().alias("n_tiendas"))
                            .filter(pl.col("n_tiendas") >= 5)
                            .select("producto_clean"))
    
    # Calcular variación de precios por producto
    variacion_precios = (df.join(productos_comparables, on="producto_clean")
                         .group_by("producto_clean")
                         .agg([
                             pl.col("precio_clean").mean().alias("precio_promedio"),
                             pl.col("precio_clean").std().alias("precio_std"),
                             pl.col("precio_clean").max().alias("precio_max"),
                             pl.col("precio_clean").min().alias("precio_min"),
                             (pl.col("precio_clean").max() - pl.col("precio_clean").min()).alias("rango"),
                             (pl.col("precio_clean").std() / pl.col("precio_clean").mean()).alias("coef_variacion"),
                             pl.col("tienda_clean").n_unique().alias("n_tiendas"),
                             pl.count().alias("total_observaciones")
                         ])
                         .filter(pl.col("precio_promedio") > 0)
                         .sort("coef_variacion", descending=True)
                         .head(15))
    
    print("Productos con mayor variacion de precios (presentes en al menos 5 tiendas):")
    for i, (producto, prom, std, max_p, min_p, rango, cv, n_tiendas, total_obs) in enumerate(variacion_precios.rows(), 1):
        diferencia_absoluta = max_p - min_p
        diferencia_relativa = (diferencia_absoluta / prom) * 100
        print(f"   {i:2d}. {producto[:35]:35}")
        print(f"        CV: {cv:.2f} | ${min_p:.2f}-${max_p:.2f} | Dif: ${diferencia_absoluta:.2f} ({diferencia_relativa:.1f}%)")
        print(f"        Tiendas: {n_tiendas} | Observaciones: {total_obs:,}")
    
    # Visualización de productos estratégicos
    if len(variacion_precios) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        productos_nombres = [p[:25] + '...' if len(p) > 25 else p for p in variacion_precios["producto_clean"].to_list()]
        coeficientes_variacion = variacion_precios["coef_variacion"].to_list()
        rangos_precios = variacion_precios["rango"].to_list()
        
        # Coeficientes de variación
        bars1 = ax1.barh(productos_nombres, coeficientes_variacion, color='lightsteelblue', alpha=0.7)
        ax1.set_title('Coeficiente de Variacion de Precios por Producto', fontsize=14)
        ax1.set_xlabel('Coeficiente de Variacion')
        ax1.grid(True, alpha=0.3)
        
        for bar, valor in zip(bars1, coeficientes_variacion):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{valor:.2f}', ha='left', va='center', fontsize=9)
        
        # Rangos de precios
        bars2 = ax2.barh(productos_nombres, rangos_precios, color='lightcoral', alpha=0.7)
        ax2.set_title('Rango de Precios por Producto (Maximo - Minimo)', fontsize=14)
        ax2.set_xlabel('Rango de Precios ($)')
        ax2.grid(True, alpha=0.3)
        
        for bar, valor in zip(bars2, rangos_precios):
            ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                    f'${valor:.0f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('analisis_imagenes/productos_variacion_precios.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Imagen guardada: analisis_imagenes/productos_variacion_precios.png")

def analisis_multivariado_segmentacion(df):
    """
    ANÁLISIS MULTIVARIADO - SEGMENTACIÓN DE MERCADO
    ===============================================
    Objetivo: Segmentar el mercado en categorías de precio y analizar
    características de cada segmento.
    """
    
    print("\n" + "="*60)
    print("ANALISIS MULTIVARIADO - SEGMENTACION")
    print("="*60)
    
    # Muestra representativa para análisis
    total_registros = len(df)
    tamaño_muestra = min(50000, total_registros)
    muestra = df.sample(tamaño_muestra, seed=42)
    print(f"Muestra utilizada: {tamaño_muestra:,} registros")
    
    # Crear segmentos de precio
    segmentos = muestra.with_columns([
        pl.when(pl.col("precio_clean") <= 50).then(pl.lit("Económico"))
        .when(pl.col("precio_clean") <= 200).then(pl.lit("Medio"))
        .otherwise(pl.lit("Premium")).alias("segmento_precio"),
        
        pl.col("producto_clean").str.len_chars().alias("longitud_nombre"),
        pl.col("producto_clean").str.contains("kg|kilo|litro|lt|gr|gramo").alias("tiene_unidades")
    ])
    
    # Análisis de segmentos
    analisis_segmentos = (segmentos.group_by("segmento_precio")
                          .agg([
                              pl.count().alias("n_productos"),
                              pl.col("precio_clean").mean().alias("precio_promedio"),
                              pl.col("precio_clean").median().alias("precio_mediano"),
                              pl.col("precio_clean").std().alias("precio_std"),
                              pl.col("tienda_clean").n_unique().alias("n_tiendas"),
                              pl.col("longitud_nombre").mean().alias("long_nombre_promedio"),
                              pl.col("tiene_unidades").mean().alias("porcentaje_con_unidades")
                          ])
                          .sort("precio_promedio"))
    
    print("Segmentacion de mercado por precio:")
    for segmento, n, promedio, mediano, std, n_tiendas, long_nombre, porc_unidades in analisis_segmentos.rows():
        print(f"   {segmento:10}")
        print(f"        Productos: {n:>6,} | Tiendas: {n_tiendas:>3}")
        print(f"        Precio: ${promedio:>6.2f} promedio | ${mediano:>6.2f} mediano | ±${std:.2f}")
        print(f"        Long. nombre: {long_nombre:.1f} chars | Con unidades: {porc_unidades:.1%}")
    
    # Identificación de productos más comunes por segmento
    productos_comunes_por_segmento = {}
    for segmento in analisis_segmentos["segmento_precio"].to_list():
        productos_segmento = (segmentos.filter(pl.col("segmento_precio") == segmento)
                            .group_by("producto_clean")
                            .agg(pl.count().alias("frecuencia"))
                            .sort("frecuencia", descending=True)
                            .head(3))
        productos_comunes_por_segmento[segmento] = productos_segmento
    
    print("\nPRODUCTOS MÁS COMUNES POR SEGMENTO:")
    for segmento, productos_df in productos_comunes_por_segmento.items():
        if len(productos_df) > 0:
            print(f"  {segmento:10}: ", end="")
            productos_lista = []
            for producto, freq in productos_df.rows():
                productos_lista.append(f"{producto[:20]}... ({freq})")
            print(" | ".join(productos_lista))
    
    # VISUALIZACIÓN - Figura organizada en grid 3x2
    if len(analisis_segmentos) > 0:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])
        
        # Configuración de colores y datos
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # Verde, Azul, Rojo
        segment_labels = analisis_segmentos["segmento_precio"].to_list()
        segment_counts = analisis_segmentos["n_productos"].to_list()
        
        # 1. Gráfico de torta - Distribución de segmentos
        ax1 = fig.add_subplot(gs[0, 0])
        def formato_porcentajes(valores):
            def autopct_func(pct):
                total = sum(valores)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val:,})'
            return autopct_func
        
        wedges, texts, autotexts = ax1.pie(
            segment_counts, 
            labels=segment_labels, 
            autopct=formato_porcentajes(segment_counts), 
            colors=colors, 
            startangle=90
        )
        plt.setp(autotexts, size=8, weight="bold")
        ax1.set_title('Distribución de Segmentos de Precio\n(50,000 productos muestreados)', 
                     fontsize=11, fontweight='bold')
        
        # 2. Tabla de métricas por segmento
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        # Preparar datos para tabla
        metricas_data = []
        for segmento, n, precio, mediano, std, n_tiendas, long_nombre, porc_unidades in analisis_segmentos.rows():
            metricas_data.append([
                segmento, 
                f"{n:,}", 
                f"${precio:.2f}", 
                f"${mediano:.2f}",
                f"{n_tiendas}",
                f"{long_nombre:.1f}",
                f"{porc_unidades:.1%}"
            ])
        
        # Crear tabla
        tabla_metricas = ax2.table(
            cellText=metricas_data,
            colLabels=['Segmento', 'Productos', 'Precio Prom', 'Precio Med', 'Tiendas', 'Chars', 'Unidades'],
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.3, 0.8, 0.6]
        )
        tabla_metricas.auto_set_font_size(False)
        tabla_metricas.set_fontsize(8)
        tabla_metricas.scale(1, 1.8)
        ax2.set_title('Métricas Detalladas por Segmento', fontsize=11, fontweight='bold')
        
        # 3. Comparativa de precios y cobertura
        ax3 = fig.add_subplot(gs[1, 0])
        x = np.arange(len(segment_labels))
        width = 0.35
        
        precios = analisis_segmentos["precio_promedio"].to_list()
        tiendas = analisis_segmentos["n_tiendas"].to_list()
        
        # Normalizar tiendas para escala compatible
        tiendas_norm = [t/max(tiendas)*max(precios)*0.3 for t in tiendas]
        
        bars1 = ax3.bar(x - width/2, precios, width, label='Precio Promedio ($)', color=colors, alpha=0.8)
        bars2 = ax3.bar(x + width/2, tiendas_norm, width, label='Tiendas (escala normalizada)', 
                       color=[c.replace('7', '4') for c in colors], alpha=0.6)
        
        ax3.set_ylabel('Precio ($)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(segment_labels)
        ax3.set_title('Comparativa: Precio vs Cobertura por Segmento', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=8)
        
        # Añadir valores en barras de precio
        for bar, precio in zip(bars1, precios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'${precio:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 4. Productos más comunes por segmento
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        productos_text = "PRODUCTOS MÁS COMUNES POR SEGMENTO\n\n"
        for segmento, productos_df in productos_comunes_por_segmento.items():
            productos_text += f"{segmento}:\n"
            for i, (producto, freq) in enumerate(productos_df.rows()):
                productos_text += f"  {i+1}. {producto[:25]}... ({freq} ocurrencias)\n"
            productos_text += "\n"
        
        ax4.text(0.05, 0.98, productos_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=1", 
                facecolor="lightgray", alpha=0.3))
        ax4.set_title('Productos Más Frecuentes por Segmento', fontsize=11, fontweight='bold')
        
        # 5. Resumen ejecutivo y hallazgos
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Cálculo de estadísticas globales para el resumen
        precio_global_muestra = muestra["precio_clean"].mean()
        tiendas_unicas_muestra = muestra["tienda_clean"].n_unique()
        productos_unicos_muestra = muestra["producto_clean"].n_unique()
        
        resumen_text = f"""
        RESUMEN EJECUTIVO - SEGMENTACIÓN DE MERCADO
        
        ESTADÍSTICAS GLOBALES:
        • Total productos analizados: {tamaño_muestra:,}
        • Precio promedio global: ${precio_global_muestra:,.2f}
        • Tiendas únicas en muestra: {tiendas_unicas_muestra:,}
        • Productos únicos en muestra: {productos_unicos_muestra:,}
        
        HALLAZGOS PRINCIPALES:
        • Segmento Económico: Mayor volumen ({segment_counts[0]:,} productos - {segment_counts[0]/tamaño_muestra*100:.1f}%)
        • Segmento Premium: Precio {precios[2]/precios[0]:.0f} veces mayor que económico
        • Cobertura similar entre segmentos (~{np.mean(tiendas):.0f} tiendas promedio)
        • Nombres más descriptivos en segmento Medio
        
        IMPLICACIONES ESTRATÉGICAS:
        • Mercado claramente estratificado por precio
        • Oportunidad en segmento medio para crecimiento
        • Productos premium ofrecen margen significativo
        • Estrategias de precio deben considerar esta segmentación
        """
        
        ax5.text(0.02, 0.98, resumen_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=1.5", 
                facecolor="lightblue", alpha=0.2))
        
        plt.tight_layout()
        plt.savefig('analisis_imagenes/segmentacion_mercado.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Imagen guardada: analisis_imagenes/segmentacion_mercado.png")
    
    return segmentos

def analisis_univariado_precios(df):
    """
    ANÁLISIS UNIVARIADO - DISTRIBUCIÓN DE PRECIOS
    =============================================
    Objetivo: Analizar la distribución, tendencia central y dispersión
    de la variable precio en el dataset.
    """
    
    print("\n" + "="*60)
    print("ANALISIS UNIVARIADO - PRECIOS")
    print("="*60)
    
    # Cálculo de estadísticas descriptivas
    precio_stats = df.select([
        pl.col("precio_clean").mean().alias("media"),
        pl.col("precio_clean").median().alias("mediana"),
        pl.col("precio_clean").std().alias("desviacion"),
        pl.col("precio_clean").min().alias("minimo"),
        pl.col("precio_clean").max().alias("maximo"),
        pl.col("precio_clean").quantile(0.25).alias("q25"),
        pl.col("precio_clean").quantile(0.75).alias("q75"),
        pl.col("precio_clean").skew().alias("asimetria")
    ])
    
    stats_names = ["Media", "Mediana", "Desviación", "Mínimo", "Máximo", "Q25", "Q75", "Asimetría"]
    stats_values = precio_stats.row(0)
    
    # Análisis de distribución por rangos de precio
    rangos_precio = df.with_columns([
        pl.when(pl.col("precio_clean") <= 50).then(pl.lit("0-50"))
        .when(pl.col("precio_clean") <= 100).then(pl.lit("50-100"))
        .when(pl.col("precio_clean") <= 500).then(pl.lit("100-500"))
        .when(pl.col("precio_clean") <= 1000).then(pl.lit("500-1000"))
        .otherwise(pl.lit("1000+")).alias("rango_precio")
    ]).group_by("rango_precio").agg(pl.count().alias("cantidad")).sort("rango_precio")
    
    # Resultados en consola
    print("ESTADÍSTICAS DESCRIPTIVAS DE PRECIOS:")
    for name, val in zip(stats_names, stats_values):
        if name == "Asimetría":
            print(f"  {name:12}: {val:+.3f}")
        else:
            print(f"  {name:12}: ${val:,.2f}")
    
    # Interpretación de asimetría
    asimetria = stats_values[7]
    if asimetria > 1:
        interpretacion = "Distribución muy asimétrica positiva - muchos productos baratos, pocos muy caros"
    elif asimetria > 0.5:
        interpretacion = "Distribución asimétrica positiva"
    else:
        interpretacion = "Distribución relativamente simétrica"
    
    print(f"\nINTERPRETACIÓN: {interpretacion}")
    
    print("\nDISTRIBUCIÓN POR RANGOS DE PRECIO:")
    for rango, cantidad in rangos_precio.rows():
        porcentaje = (cantidad / len(df)) * 100
        print(f"  {rango:10}: {cantidad:>8,} productos ({porcentaje:.1f}%)")
    
    # VISUALIZACIÓN - Análisis univariado completo
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Filtrar precios extremos para mejor visualización
    precios_filtrados = df.filter(pl.col("precio_clean") <= 1000)
    hist_data = precios_filtrados["precio_clean"].to_numpy()
    
    # 1. Histograma de distribución
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(hist_data, bins=50, alpha=0.7, edgecolor='black', 
                               color='skyblue', density=True)
    ax1.set_title('Distribución de Precios\n(Precios menores a $1,000 para mejor visualización)', 
                 fontsize=11, fontweight='bold')
    ax1.set_xlabel('Precio ($)', fontweight='bold')
    ax1.set_ylabel('Densidad de Probabilidad', fontweight='bold')
    
    # Líneas de referencia para media y mediana
    media, mediana = stats_values[0], stats_values[1]
    ax1.axvline(mediana, color='red', linestyle='--', linewidth=2, 
                label=f'Mediana: ${mediana:.0f}')
    ax1.axvline(media, color='green', linestyle='--', linewidth=2, 
                label=f'Media: ${media:.0f}')
    ax1.legend(fontsize=8)
    
    # 2. Boxplot para identificar outliers y dispersión
    ax2 = fig.add_subplot(gs[0, 1])
    box = ax2.boxplot([hist_data], patch_artist=True, labels=['Distribución de Precios'])
    box['boxes'][0].set_facecolor('lightgreen')
    ax2.set_title('Diagrama de Caja - Distribución y Valores Atípicos', 
                 fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precio ($)', fontweight='bold')
    
    # 3. Distribución por rangos de precio
    ax3 = fig.add_subplot(gs[1, 0])
    rangos = rangos_precio["rango_precio"].to_list()
    cantidades = rangos_precio["cantidad"].to_list()
    porcentajes = [(c/len(df))*100 for c in cantidades]
    
    # Colores semánticos para rangos
    colores_rangos = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    bars = ax3.bar(rangos, cantidades, color=colores_rangos)
    ax3.set_title('Distribución de Productos por Rango de Precio', 
                 fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cantidad de Productos', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Anotar valores y porcentajes en las barras
    for bar, cantidad, porcentaje in zip(bars, cantidades, porcentajes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10000,
                f'{cantidad:,}\n({porcentaje:.1f}%)', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
    
    # 4. Resumen estadístico y análisis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Preparar texto de resumen
    resumen_text = f"""
    RESUMEN EJECUTIVO - ANÁLISIS DE PRECIOS
    
    CONTEXTO DEL DATASET:
    • Total de productos: {len(df):,}
    • Rango de precios: ${stats_values[3]:.2f} - ${stats_values[4]:,.2f}
    • Dispersión: Alta (desviación estándar: ${stats_values[2]:,.2f})
    
    HALLAZGOS PRINCIPALES:
    • Concentración en bajo costo: {porcentajes[0]:.1f}% productos en rango 0-50
    • Segmento premium limitado: {porcentajes[4]:.1f}% productos arriba de $1,000
    • Gran diferencia entre media y mediana indica presencia de outliers
    • Distribución: {interpretacion}
    
    IMPLICACIONES:
    • Estrategias de precio deben considerar alta concentración en segmento económico
    • Mercado dominado por productos de bajo costo
    • Oportunidad en segmentos medios y premium
    """
    
    ax4.text(0.05, 0.95, resumen_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=1", 
            facecolor="lightyellow", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('analisis_imagenes/analisis_precios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Imagen guardada: analisis_imagenes/analisis_precios.png")
    return precios_filtrados

def analisis_valor_productos(df):
    """
    ANÁLISIS DE VALOR - IDENTIFICACIÓN DE PRODUCTOS ESTRATÉGICOS
    ============================================================
    Objetivo: Identificar productos de alto valor y alta cobertura
    que representan oportunidades estratégicas.
    """
    
    print("\n" + "="*60)
    print("ANALISIS DE VALOR DE PRODUCTOS")
    print("="*60)
    
    # Calcular métricas de valor por producto
    metricas_productos = (df.group_by("producto_clean")
                          .agg([
                              pl.col("precio_clean").mean().alias("precio_promedio"),
                              pl.col("precio_clean").std().alias("precio_std"),
                              pl.col("tienda_clean").n_unique().alias("cobertura_tiendas"),
                              pl.count().alias("frecuencia_escaneo")
                          ])
                          .filter(
                              (pl.col("cobertura_tiendas") >= 3) &
                              (pl.col("frecuencia_escaneo") >= 10)
                          ))
    
    print(f"Productos analizados: {len(metricas_productos):,}")
    
    if len(metricas_productos) > 0:
        # Cálculo de percentiles para clasificación
        q25_precio = metricas_productos["precio_promedio"].quantile(0.25)
        q75_precio = metricas_productos["precio_promedio"].quantile(0.75)
        q25_cobertura = metricas_productos["cobertura_tiendas"].quantile(0.25)
        q75_cobertura = metricas_productos["cobertura_tiendas"].quantile(0.75)
        
        print(f"Umbrales de clasificación - Precio: ${q25_precio:.2f} / ${q75_precio:.2f}")
        print(f"Umbrales de clasificación - Cobertura: {q25_cobertura:.0f} / {q75_cobertura:.0f} tiendas")
        
        # Clasificación de productos en matriz de valor-cobertura
        productos_categorizados = metricas_productos.with_columns([
            pl.when(pl.col("precio_promedio") >= q75_precio)
              .then(pl.lit("Alto Valor"))
              .when(pl.col("precio_promedio") <= q25_precio)
              .then(pl.lit("Bajo Valor"))
              .otherwise(pl.lit("Valor Medio")).alias("categoria_valor"),
            pl.when(pl.col("cobertura_tiendas") >= q75_cobertura)
              .then(pl.lit("Alta Cobertura"))
              .when(pl.col("cobertura_tiendas") <= q25_cobertura)
              .then(pl.lit("Baja Cobertura"))
              .otherwise(pl.lit("Cobertura Media")).alias("categoria_cobertura")
        ])
        
        # Matriz de categorías
        matriz_categorias = (productos_categorizados.group_by(["categoria_valor", "categoria_cobertura"])
                            .agg(pl.count().alias("n_productos"))
                            .sort("n_productos", descending=True))
        
        # Productos estratégicos: Alto Valor + Alta Cobertura
        productos_estrategicos = (productos_categorizados
                                 .filter(
                                     (pl.col("categoria_valor") == "Alto Valor") &
                                     (pl.col("categoria_cobertura") == "Alta Cobertura")
                                 )
                                 .sort("precio_promedio", descending=True)
                                 .head(10))
        
        # Resultados en consola
        print("\nMATRIZ DE VALOR/COBERTURA:")
        total_productos = sum(n for _, _, n in matriz_categorias.rows())
        for (valor, cobertura, n) in matriz_categorias.rows():
            porcentaje = (n / total_productos) * 100
            print(f"  {valor:12} + {cobertura:15} | {n:>5,} productos ({porcentaje:.1f}%)")
        
        print(f"\nPRODUCTOS ESTRATEGICOS IDENTIFICADOS (Alto Valor + Alta Cobertura): {len(productos_estrategicos)}")
        for i, row in enumerate(productos_estrategicos.rows(), 1):
            producto, precio, std, cobertura, freq, cat_valor, cat_cob = row
            print(f"  {i:2d}. {producto[:40]:40} ${precio:>6.2f} | {cobertura:>2} tiendas | {freq:>4} escaneos")
        
        # VISUALIZACIÓN - Análisis de valor estratégico
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        
        # 1. Matriz de calor - Valor vs Cobertura
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Preparar datos para heatmap
        categorias_valor = ["Bajo Valor", "Valor Medio", "Alto Valor"]
        categorias_cobertura = ["Baja Cobertura", "Cobertura Media", "Alta Cobertura"]
        matriz_heatmap = np.zeros((3, 3))
        
        for (valor, cobertura, n) in matriz_categorias.rows():
            i = categorias_valor.index(valor)
            j = categorias_cobertura.index(cobertura)
            matriz_heatmap[i, j] = n
        
        # Crear heatmap
        im = ax1.imshow(matriz_heatmap, cmap='YlOrRd', aspect='auto')
        
        # Añadir valores en celdas
        for i in range(3):
            for j in range(3):
                text = ax1.text(j, i, f'{matriz_heatmap[i, j]:.0f}',
                               ha="center", va="center", color="black", 
                               fontweight='bold', fontsize=10)
        
        ax1.set_xticks(range(3))
        ax1.set_yticks(range(3))
        ax1.set_xticklabels(categorias_cobertura, rotation=45, ha='right')
        ax1.set_yticklabels(categorias_valor)
        ax1.set_title('Matriz de Valor vs Cobertura', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Número de Productos', shrink=0.8)
        
        # 2. Dispersión Precio vs Cobertura
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Muestra para visualización clara
        muestra_viz = productos_categorizados.sample(min(200, len(productos_categorizados)), seed=42)
        precios = muestra_viz["precio_promedio"].to_numpy()
        coberturas = muestra_viz["cobertura_tiendas"].to_numpy()
        categorias = muestra_viz["categoria_valor"].to_list()
        
        # Mapeo de colores por categoría de valor
        color_map = {"Bajo Valor": "green", "Valor Medio": "blue", "Alto Valor": "red"}
        colors = [color_map[cat] for cat in categorias]
        
        scatter = ax2.scatter(precios, coberturas, c=colors, alpha=0.6, s=40)
        ax2.set_xlabel('Precio Promedio ($)', fontweight='bold')
        ax2.set_ylabel('Cobertura (Número de Tiendas)', fontweight='bold')
        ax2.set_title('Dispersión: Precio vs Cobertura de Productos', 
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Leyenda para categorías de valor
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=6, label='Bajo Valor'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=6, label='Valor Medio'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=6, label='Alto Valor')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 3. Tabla de productos estratégicos
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        if len(productos_estrategicos) > 0:
            # Preparar datos para tabla
            tabla_data = []
            for i, row in enumerate(productos_estrategicos.rows(), 1):
                producto, precio, std, cobertura, freq, cat_valor, cat_cob = row
                tabla_data.append([
                    f"{i}", 
                    producto[:25], 
                    f"${precio:.0f}", 
                    f"{cobertura}", 
                    f"{freq:,}"
                ])
            
            # Crear tabla
            tabla = ax3.table(
                cellText=tabla_data,
                colLabels=['#', 'Producto', 'Precio', 'Tiendas', 'Escaneos'],
                cellLoc='left',
                loc='center',
                bbox=[0.1, 0.1, 0.9, 0.8]
            )
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(8)
            tabla.scale(1, 1.5)
            ax3.set_title('Top 10 Productos Estratégicos\n(Alto Valor + Alta Cobertura)', 
                         fontsize=11, fontweight='bold')
        
        # 4. Resumen y recomendaciones estratégicas
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Calcular métricas para el resumen
        if len(productos_estrategicos) > 0:
            precio_promedio_estrategicos = productos_estrategicos["precio_promedio"].mean()
            cobertura_promedio_estrategicos = productos_estrategicos["cobertura_tiendas"].mean()
            escaneos_promedio_estrategicos = productos_estrategicos["frecuencia_escaneo"].mean()
        else:
            precio_promedio_estrategicos = cobertura_promedio_estrategicos = escaneos_promedio_estrategicos = 0
        
        resumen_text = f"""
        RESUMEN - ANÁLISIS DE VALOR ESTRATÉGICO
        
        CONTEXTO:
        • Productos analizados: {len(metricas_productos):,}
        • Umbral Precio: ${q25_precio:.0f} / ${q75_precio:.0f}
        • Umbral Cobertura: {q25_cobertura:.0f} / {q75_cobertura:.0f} tiendas
        
        PRODUCTOS ESTRATÉGICOS: {len(productos_estrategicos)}
        • Precio promedio: ${precio_promedio_estrategicos:.0f}
        • Cobertura promedio: {cobertura_promedio_estrategicos:.0f} tiendas
        • Escaneos promedio: {escaneos_promedio_estrategicos:.0f}
        
        RECOMENDACIONES ESTRATÉGICAS:
        • Fortalecer presencia en productos estratégicos identificados
        • Desarrollar estrategias para segmento Bajo Valor + Alta Cobertura
        • Optimizar precios en categoría Valor Medio
        • Monitorear competencia en productos premium
        """
        
        ax4.text(0.05, 0.95, resumen_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=1", 
                facecolor="lightgreen", alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('analisis_imagenes/analisis_valor_productos.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Imagen guardada: analisis_imagenes/analisis_valor_productos.png")
        return productos_categorizados
    
    print("No hay productos que cumplan los criterios de filtrado para análisis de valor")
    return None

def analisis_temporal_evolucion_precios(df):
    """
    Análisis de cómo han evolucionado los precios en el tiempo.
    
    Args:
        df (pl.DataFrame): DataFrame con los datos
    """
    print("\n" + "="*60)
    print("ANALISIS TEMPORAL - EVOLUCION DE PRECIOS")
    print("="*60)
    
    # Estrategia: Agrupar por productos y analizar variaciones
    productos_temporales = (df.group_by("producto_clean")
                            .agg([
                                pl.col("precio_clean").mean().alias("precio_promedio"),
                                pl.col("precio_clean").std().alias("volatilidad"),
                                pl.col("precio_clean").max().alias("precio_max"),
                                pl.col("precio_clean").min().alias("precio_min"),
                                pl.count().alias("n_observaciones")
                            ])
                            .filter(pl.col("n_observaciones") >= 100)
                            .sort("volatilidad", descending=True))
    
    print("Productos con mayor volatilidad temporal (min. 100 observaciones):")
    for i, (producto, promedio, vol, max_p, min_p, n_obs) in enumerate(productos_temporales.head(10).rows(), 1):
        variacion_absoluta = max_p - min_p
        variacion_relativa = ((max_p - min_p) / promedio) * 100
        print(f"   {i:2d}. {producto[:35]:35}")
        print(f"        Precio: ${promedio:.2f} ± ${vol:.2f}")
        print(f"        Rango: ${min_p:.2f}-${max_p:.2f} (${variacion_absoluta:.2f}, {variacion_relativa:.1f}%)")
        print(f"        Observaciones: {n_obs:,}")
    
    # Productos con alta variación
    productos_estacionales = (df.group_by("producto_clean")
                              .agg([
                                  pl.col("precio_clean").mean().alias("precio_global"),
                                  (pl.col("precio_clean").std() / pl.col("precio_clean").mean()).alias("coef_variacion"),
                                  pl.count().alias("n_observaciones")
                              ])
                              .filter(pl.col("coef_variacion") > 0.3)
                              .sort("coef_variacion", descending=True)
                              .head(15))
    
    # Visualización de productos más volátiles
    plt.figure(figsize=(15, 10))
    
    top_volatiles = productos_estacionales.head(8)
    
    for i, (producto, precio_global, cv, n_obs) in enumerate(top_volatiles.rows()):
        datos_producto = (df.filter(pl.col("producto_clean") == producto)
                         .select("precio_clean")
                         .head(500))
        
        if len(datos_producto) > 0:
            plt.subplot(2, 4, i+1)
            precios = datos_producto["precio_clean"].to_numpy()
            plt.hist(precios, bins=30, alpha=0.7, edgecolor='black', color='lightblue')
            plt.title(f'{producto[:20]}...\nCV: {cv:.2f}\nn={n_obs:,}')
            plt.xlabel('Precio ($)')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_imagenes/volatilidad_precios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Imagen guardada: analisis_imagenes/volatilidad_precios.png")
    return productos_temporales

def ejecutar_analisis_completo():
    """
    Función principal que ejecuta todos los análisis en secuencia.
    """
    print("INICIANDO ANALISIS COMPLETO DE DATOS PROFECO")
    print("="*70)
    
    # Configurar entorno y cargar datos
    df = configurar_entorno()
    if df is None:
        print("No se pudieron cargar los datos. Verifique la ruta del archivo.")
        return
    
    # Mostrar resumen ejecutivo inicial
    mostrar_resumen_ejecutivo(df)
    
    # FASE 1: ANÁLISIS BÁSICOS
    print("\n" + "="*70)
    print("FASE 1: ANALISIS BASICOS")
    print("="*70)
    
    analisis_univariado_productos(df)
    analisis_univariado_tiendas(df)
    analisis_bivariado_precio_tienda(df)
    analisis_productos_estrategicos(df)
    
    # FASE 2: ANÁLISIS MULTIVARIADO Y SEGMENTACIÓN
    print("\n" + "="*70)
    print("FASE 2: ANALISIS MULTIVARIADO Y SEGMENTACION")
    print("="*70)
    
    analisis_multivariado_segmentacion(df)
    precios_filtrados = analisis_univariado_precios(df)
    analisis_valor_productos(df)
    
    # FASE 3: ANÁLISIS AVANZADOS
    print("\n" + "="*70)
    print("FASE 3: ANALISIS AVANZADOS")
    print("="*70)
    
    analisis_temporal_evolucion_precios(df)
    
    print("\n" + "="*70)
    print("ANALISIS COMPLETADO EXITOSAMENTE")
    print("Todas las imagenes se guardaron en la carpeta: analisis_imagenes/")
    print("="*70)

if __name__ == "__main__":
    ejecutar_analisis_completo()
