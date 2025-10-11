import os
import sys
# Add src directory to Python path to fix import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def main():
    """
    Análisis de datos de estudiantes con nuevas columnas y cálculos
    """
    try:
        # Configurar rutas
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        input_path = os.path.join(base_dir, 'src', 'Ejercicios', 'estudiantes_ejercicio.csv')
        output_dir = os.path.join(base_dir, 'data', 'processed')
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Cargar y verificar archivo
        print("\n=== 1. VERIFICACIÓN DE DATOS ===")
        df = pd.read_csv(input_path)
        columnas_requeridas = ['Nombre', 'Edad', 'Carrera', 'Promedio', 'Promedio_10', 'Promedio_100', 'Aprobado']
        
        print("Columnas en el archivo:")
        for col in df.columns:
            status = "✅" if col in columnas_requeridas else "❌"
            print(f"{status} {col}")
        
        # 2. Análisis de Promedio_10
        print("\n=== 2. ANÁLISIS DE PROMEDIO_10 ===")
        media = df['Promedio_10'].mean()
        mediana = df['Promedio_10'].median()
        moda = df['Promedio_10'].mode()[0]
        
        print(f"""
        Medidas de tendencia central:
        - Media: {media:.2f} 
          → Representa el promedio general del grupo
          → Afectada por valores extremos
          
        - Mediana: {mediana:.2f}
          → Valor central que divide al grupo en dos partes iguales
          → Menos sensible a valores extremos
          
        - Moda: {moda:.2f}
          → Calificación más frecuente
          → Indica el valor más común en el grupo
        """)
        
        # 3. Filtrar estudiantes aprobados
        print("\n=== 3. ESTUDIANTES APROBADOS ===")
        aprobados = df[df['Aprobado'] == True]['Nombre'].tolist()
        print("Lista de estudiantes aprobados:")
        for i, nombre in enumerate(aprobados, 1):
            print(f"{i}. {nombre}")
        
        # 4. Carrera con mejor promedio
        print("\n=== 4. ANÁLISIS POR CARRERA ===")
        # Calcular promedio por carrera y ordenar de mayor a menor
        promedios_carrera = df.groupby('Carrera')['Promedio'].agg(['mean', 'count', 'std']).round(2)
        promedios_carrera = promedios_carrera.sort_values('mean', ascending=False)
        
        mejor_carrera = promedios_carrera.iloc[0]
        print(f"""
        Carrera con mejor desempeño: {promedios_carrera.index[0]}
        - Promedio: {mejor_carrera['mean']:.2f}
        - Cantidad de estudiantes: {mejor_carrera['count']}
        - Desviación estándar: {mejor_carrera['std']:.2f}
        
        Método de cálculo:
        1. Agrupar datos por carrera
        2. Calcular promedio (mean) para cada grupo
        3. Ordenar de mayor a menor
        4. Seleccionar la primera carrera (mejor promedio)
        """)
        
        # 5. Crear y analizar columna Diferencia
        print("\n=== 5. ANÁLISIS DE DIFERENCIA ===")
        df['Diferencia'] = df['Promedio_100'] - (df['Edad'] * 2)
        
        print("""
        Interpretación de la columna 'Diferencia':
        - Valores positivos: El rendimiento supera el doble de la edad
        - Valores negativos: El rendimiento está por debajo del doble de la edad
        - Cuanto mayor sea el valor, mejor es el rendimiento en relación a la edad
        """)
        
        print("\nPrimeros 5 ejemplos:")
        ejemplos = df[['Nombre', 'Edad', 'Promedio_100', 'Diferencia']].head()
        print(ejemplos)
        
        # 6. Guardar subconjunto
        print("\n=== 6. GUARDANDO SUBCONJUNTO DE DATOS ===")
        columnas_subset = ['Nombre', 'Carrera', 'Promedio_10', 'Aprobado']
        df_subset = df[columnas_subset]
        
        output_path = os.path.join(output_dir, 'estudiantes_subset.csv')
        df_subset.to_csv(output_path, index=False)
        print(f"✅ Archivo guardado en: {output_path}")
        print("\nContenido del nuevo archivo:")
        print(df_subset)
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {input_path}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
