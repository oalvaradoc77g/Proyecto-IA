import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo CSV usando ruta relativa
df = pd.read_csv('src/Ejercicios/estudiantes_original.csv')

# 1. Análisis Estadístico Básico
print("\n=== Estadísticas Descriptivas ===")
print(df.describe())

print("\n=== Información del DataFrame ===")
print(df.info())

print("\n=== Primeras 5 Filas del DataFrame ===")
print(df.head())

print("\n=== Últimas 5 Filas del DataFrame ===")
print(df.tail())

print("\n=== Valores Nulos por Columna ===")
print(df.isnull().sum())

print("\n=== Dimensiones del DataFrame ===")
print(df.shape())

# 2. Análisis por Carrera
print("\n=== Análisis por Carrera ===")
print("\nPromedio por Carrera:")
print(df.groupby('Carrera')['Promedio'].mean().sort_values(ascending=False))

print("\nEdad Promedio por Carrera:")
print(df.groupby('Carrera')['Edad'].mean().sort_values(ascending=False))

print("\nCantidad de Estudiantes por Carrera:")
print(df['Carrera'].value_counts())

# 3. Visualizaciones
plt.figure(figsize=(12, 8))

# Gráfico de dispersión: Edad vs Promedio
plt.subplot(2, 2, 1)
plt.scatter(df['Edad'], df['Promedio'])
plt.xlabel('Edad')
plt.ylabel('Promedio')
plt.title('Relación Edad-Promedio')

# Gráfico de barras: Promedio por Carrera
plt.subplot(2, 2, 2)
df.groupby('Carrera')['Promedio'].mean().plot(kind='bar')
plt.title('Promedio por Carrera')
plt.xticks(rotation=45)

# Histograma de Edades
plt.subplot(2, 2, 3)
plt.hist(df['Edad'], bins=10)
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.title('Distribución de Edades')

# Boxplot de Promedios
plt.subplot(2, 2, 4)
plt.boxplot(df['Promedio'])
plt.title('Distribución de Promedios')

plt.tight_layout()

# 4. Correlaciones
print("\n=== Correlación entre Edad y Promedio ===")
correlation = df['Edad'].corr(df['Promedio'])
print(f"Coeficiente de correlación: {correlation:.2f}")

# 5. Estadísticas Adicionales
print("\n=== Estadísticas Adicionales ===")
print(f"\nEstudiante con mejor promedio:")
print(df.loc[df['Promedio'].idxmax()])

print(f"\nEstudiante con menor promedio:")
print(df.loc[df['Promedio'].idxmin()])

print(f"\nEdad promedio de los estudiantes: {df['Edad'].mean():.2f}")
print(f"Promedio general: {df['Promedio'].mean():.2f}")

# Mostrar todas las gráficas
plt.show()
