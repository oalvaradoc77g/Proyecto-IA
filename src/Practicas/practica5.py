
#1. Cargar documento datatset_bigdata.xlsx y verificar que se cargó correctamente
import pandas as pd
import os

df = pd.read_excel("src/Ejercicios/dataset_bigdata.xlsx", engine='openpyxl')

print(df.head())

#~Convertir columnas en tipo fecha
df['Fecha de Nacimiento'] = pd.to_datetime(df['fecha_hora'], errors='coerce')

print("\n=== Verificación de tipos de datos ===")
print(df.dtypes)
print("\n=== Información del DataFrame ===")
print(df.info())
print("\n=== Verificación de datos nulos ===")
print(df.isnull().sum())    

# Punto 2 Segmentacion de dataframe (seleccion y filtrado   de datos)
# Queremos analizar solo eventos "stream" de sensores con "tamano_gb" mayor a 10 gb
# Seleccion de columnas
columnas_interes = ['id_registro', 'origen_datos','tamano_gb']
df = df[columnas_interes]
df_stream = df[(df['id_registro'] == 'stream') & (df['tamano_gb'] > 10)]