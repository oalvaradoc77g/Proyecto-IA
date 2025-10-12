"""
Script principal para análisis de movimientos financieros
Proyecto: Predicción Hipoteca y Análisis Financiero
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os

# Verificar dependencias antes de importar
def verificar_dependencias():
    """Verifica que todas las dependencias estén instaladas"""
    dependencias = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'plotly': 'plotly'
    }
    
    faltantes = []
    for nombre, paquete in dependencias.items():
        try:
            __import__(paquete)
        except ImportError:
            faltantes.append(nombre)
    
    if faltantes:
        print("❌ Faltan las siguientes dependencias:")
        for dep in faltantes:
            print(f"   - {dep}")
        print("\n💡 Ejecuta uno de estos comandos para instalar:")
        print("   python install_dependencies.py")
        print("   pip install -r requirements.txt")
        print(f"   pip install {' '.join(faltantes)}")
        sys.exit(1)

# Verificar dependencias
verificar_dependencias()

import pandas as pd
import numpy as np

# Agregar el directorio base al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos de análisis
from analisis import (
    analizar_tendencias,
    analizar_categorias,
    proyectar_tendencias,
    analizar_patrones_estacionales,
    calcular_ratios_financieros,
    analizar_sensibilidad,
    crear_dashboard_interactivo
)

RANDOM_STATE = 42


def main():
    """Función principal - Análisis de tendencias financieras"""
    # Ruta actualizada
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_path, 'data', 'raw', 'Datos Movimientos Financieros.csv')
    
    if not os.path.exists(path):
        print(f"❌ Error: El archivo no se encuentra en la ruta: {path}")
        return
    
    try:
        df = pd.read_csv(path)
        print(f"📊 Datos cargados: {len(df)} registros\n")
        
        # Conversión de fechas mejorada
        month_dict = {
            'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
        }
        
        def convert_date_smart(date_str):
            try:
                parts = date_str.split()
                if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
                    year, month, day = parts[0], parts[1], parts[2]
                else:
                    month, day = parts[0], parts[1]
                    year = str(pd.Timestamp.now().year)
                
                return f"{year}-{month_dict.get(month, '01')}-{day.zfill(2)}"
            except Exception as e:
                print(f"⚠️ Error procesando fecha '{date_str}': {e}")
                return f"{pd.Timestamp.now().year}-01-01"

        df['Fecha'] = df['Fecha'].apply(convert_date_smart)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Limpiar valores numéricos
        def clean_numeric(x):
            if pd.isna(x) or x == '':
                return 0.0
            if isinstance(x, str):
                return float(x.replace(',', '').strip())
            return float(x)
        
        df['Débitos'] = df['Débitos'].apply(clean_numeric)
        df['Créditos'] = df['Créditos'].apply(clean_numeric)
        df['Saldo'] = df['Saldo'].apply(clean_numeric)
    
        # Dashboards interactivos
        print("\n📊 Generando dashboards interactivos...\n")
        crear_dashboard_interactivo(df.copy())
        
        # Análisis de tendencias
        print("🔍 Iniciando análisis de tendencias...\n")
        analizar_tendencias(df.copy())
        
        # Análisis de categorías
        print("\n🏷️ Iniciando análisis de categorías...\n")
        analizar_categorias(df.copy())
        
        # Análisis de patrones estacionales
        print("\n🌡️ Iniciando análisis de patrones estacionales...\n")
        analizar_patrones_estacionales(df.copy())
        
        # Análisis de ratios financieros
        print("\n📊 Iniciando análisis de ratios financieros...\n")
        calcular_ratios_financieros(df.copy())
        
        # Análisis de sensibilidad
        print("\n🔍 Iniciando análisis de sensibilidad...\n")
        analizar_sensibilidad(df.copy(), meses_proyeccion=12)
        
        # Proyecciones futuras
        print("\n🔮 Iniciando proyecciones futuras...\n")
        proyectar_tendencias(df.copy(), meses_futuro=6)

    except Exception as e:
        print(f"❌ Error en el procesamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
