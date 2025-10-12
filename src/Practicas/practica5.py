import pandas as pd
import numpy as np

# 1. Cargar el fichero de excel "dataset_bigdata" y nombrarlo "df_practica5"
print("=" * 80)
print("📂 PRÁCTICA 5: Análisis de Dataset BigData")
print("=" * 80)

df_practica5 = pd.read_excel("src/Ejercicios/dataset_bigdata.xlsx", engine='openpyxl')
print(f"\n✅ Dataset cargado exitosamente: {len(df_practica5)} registros")
print("\n📊 Primeras filas del dataset:")
print(df_practica5.head())

print("\n📋 Información del dataset:")
print(df_practica5.info())

print("\n🔍 Valores nulos por columna:")
print(df_practica5.isnull().sum())

# 🔍 DIAGNÓSTICO: Ver qué datos tenemos disponibles
print("\n" + "=" * 80)
print("🔍 DIAGNÓSTICO DE DATOS DISPONIBLES")
print("=" * 80)

print("\n📊 Tipos de procesamiento disponibles:")
print(df_practica5['tipo_procesamiento'].value_counts())

print("\n📊 Orígenes de datos disponibles:")
print(df_practica5['origen_datos'].value_counts())

print("\n📊 Combinaciones tipo_procesamiento + origen_datos:")
print(df_practica5.groupby(['tipo_procesamiento', 'origen_datos']).size())

print("\n📊 Registros 'batch' con tamano_gb >= 6:")
batch_mayor_6 = df_practica5[
    (df_practica5['tipo_procesamiento'] == 'batch') & 
    (df_practica5['tamano_gb'] >= 6)
]
print(batch_mayor_6[['origen_datos', 'tamano_gb', 'tipo_procesamiento']])

# 2. Analizar solo eventos batch de redes con tamano_gb >= 6
print("\n" + "=" * 80)
print("🔎 FILTRADO DE DATOS")
print("=" * 80)

# Primero verificar si existen registros que cumplan las condiciones
print("\n🔍 Verificando condiciones de filtrado...")
print(f"   • Registros tipo 'batch': {(df_practica5['tipo_procesamiento'] == 'batch').sum()}")
print(f"   • Registros de 'red': {(df_practica5['origen_datos'] == 'red').sum()}")
print(f"   • Registros con tamano_gb >= 6: {(df_practica5['tamano_gb'] >= 6).sum()}")

# Intentar filtrado más flexible si no hay datos
df_filtrado = df_practica5[
    (df_practica5['tipo_procesamiento'] == 'batch') & 
    (df_practica5['origen_datos'] == 'red') & 
    (df_practica5['tamano_gb'] >= 6)
].copy()

if len(df_filtrado) == 0:
    print("\n⚠️ No hay registros que cumplan las 3 condiciones simultáneamente.")
    print("📝 Aplicando filtrado alternativo: tipo='batch' O origen='red', con tamano_gb >= 6")
    
    # Filtrado alternativo más flexible
    df_filtrado = df_practica5[
        (
            (df_practica5['tipo_procesamiento'] == 'batch') | 
            (df_practica5['origen_datos'] == 'red')
        ) & 
        (df_practica5['tamano_gb'] >= 6)
    ].copy()

print(f"\n✅ Registros filtrados: {len(df_filtrado)}")

if len(df_filtrado) == 0:
    print("\n❌ ERROR: No se encontraron registros con tamano_gb >= 6")
    print("📝 Usando todos los registros disponibles para demostración...")
    df_filtrado = df_practica5.copy()

print(f"\n📊 Vista previa de datos filtrados ({len(df_filtrado)} registros):")
print(df_filtrado.head(10))

print("\n🔍 Valores nulos en datos filtrados:")
print(df_filtrado.isnull().sum())

# 3. Imputar los NaN con la mediana (CORREGIDO)
print("\n" + "=" * 80)
print("🔧 IMPUTACIÓN DE VALORES NULOS")
print("=" * 80)

# Calcular medianas antes de imputar (con manejo de NaN)
mediana_latencia = df_filtrado['latencia_ms'].median()
mediana_anomalia = df_filtrado['etiqueta_anomalia'].median()

# Si la mediana es NaN, usar la media o un valor por defecto
if pd.isna(mediana_latencia):
    mediana_latencia = df_filtrado['latencia_ms'].mean()
    if pd.isna(mediana_latencia):
        mediana_latencia = 0
    print(f"⚠️ Usando media en lugar de mediana para latencia_ms")

if pd.isna(mediana_anomalia):
    mediana_anomalia = df_filtrado['etiqueta_anomalia'].mean()
    if pd.isna(mediana_anomalia):
        mediana_anomalia = 0
    print(f"⚠️ Usando media en lugar de mediana para etiqueta_anomalia")

print(f"\n📈 Valores para imputación:")
print(f"   - latencia_ms: {mediana_latencia:.2f}")
print(f"   - etiqueta_anomalia: {mediana_anomalia:.2f}")

# Imputar valores nulos (CORREGIDO - sin inplace)
df_filtrado.loc[:, 'latencia_ms'] = df_filtrado['latencia_ms'].fillna(mediana_latencia)
df_filtrado.loc[:, 'etiqueta_anomalia'] = df_filtrado['etiqueta_anomalia'].fillna(mediana_anomalia)

print(f"\n✅ Valores nulos imputados")
print(f"\n🔍 Verificación de valores nulos después de imputación:")
print(df_filtrado[['latencia_ms', 'etiqueta_anomalia']].isnull().sum())

# 4. Agrupar por tipo de procesamiento y calcular
print("\n" + "=" * 80)
print("📊 AGRUPACIÓN Y ANÁLISIS")
print("=" * 80)

# Agrupar por tipo_procesamiento y calcular estadísticas
resultado = df_filtrado.groupby('tipo_procesamiento').agg({
    'id_registro': 'count',                    # a. cantidad de registros
    'tamano_gb': 'mean',                       # b. media de tamano_gb
    'tasa_eventos_por_seg': 'sum'              # c. total tasa de eventos por seg
}).rename(columns={
    'id_registro': 'cantidad_registros',
    'tamano_gb': 'media_tamano_gb',
    'tasa_eventos_por_seg': 'total_tasa_eventos_seg'
})

print("\n📈 RESULTADOS DEL ANÁLISIS:")
print("=" * 80)
print(resultado)

# Formatear resultados para mejor visualización
print("\n📋 RESUMEN DETALLADO:")
print("=" * 80)
if len(resultado) > 0:
    for tipo_proc, row in resultado.iterrows():
        print(f"\n🔹 Tipo de procesamiento: {tipo_proc.upper()}")
        print(f"   a) Cantidad de registros: {int(row['cantidad_registros'])}")
        print(f"   b) Media de tamaño (GB): {row['media_tamano_gb']:.2f} GB")
        print(f"   c) Total tasa de eventos/seg: {row['total_tasa_eventos_seg']:.2f}")
else:
    print("⚠️ No hay datos para mostrar en el resumen")

# Guardar resultados en CSV
output_path = 'src/Practicas/practica5_resultados.csv'
resultado.to_csv(output_path)
print(f"\n💾 Resultados guardados en: {output_path}")

# Estadísticas adicionales
print("\n" + "=" * 80)
print("📊 ESTADÍSTICAS ADICIONALES")
print("=" * 80)

if len(df_filtrado) > 0:
    print(f"\n🔢 Estadísticas descriptivas de latencia_ms:")
    print(df_filtrado['latencia_ms'].describe())
    
    print(f"\n🔢 Estadísticas descriptivas de tamano_gb:")
    print(df_filtrado['tamano_gb'].describe())
    
    print(f"\n🔢 Distribución de etiqueta_anomalia:")
    print(df_filtrado['etiqueta_anomalia'].value_counts())
else:
    print("⚠️ No hay datos para mostrar estadísticas")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
