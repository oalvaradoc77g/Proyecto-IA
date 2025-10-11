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


# 3. Calculo de las estadistricas descriptivas de variable promedio
print("\n=== Estadísticas Descriptivas de Promedio ===")
print(f"Media: {df['Promedio'].mean():.2f}")
print(f"Mediana: {df['Promedio'].median():.2f}")
print(f"Moda: {df['Promedio'].mode()[0]:.2f}")
print(f"Desviación Estándar: {df['Promedio'].std():.2f}")
print(f"Varianza: {df['Promedio'].var():.2f}")
print(f"Rango: {df['Promedio'].max() - df['Promedio'].min():.2f}")
print(f"Coeficiente de Variación: {(df['Promedio'].std() / df['Promedio'].mean() * 100):.2f}%")
print(f"Skewness: {df['Promedio'].skew():.2f}")
print(f"Kurtosis: {df['Promedio'].kurt():.2f}")
print(f"Percentiles (25%, 50%, 75%): {df['Promedio'].quantile([0.25, 0.5, 0.75]).values}")
print(f"Valores Atípicos (IQR):")
Q1 = df['Promedio'].quantile(0.25)
Q2 = df['Promedio'].quantile(0.50)
Q3 = df['Promedio'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Promedio'] < (Q1 - 1.5 * IQR)) | (df['Promedio'] > (Q3 + 1.5 * IQR))]
print(outliers)

# Filtro y extraccion de datos
filtered_df = df[(df['Edad'] > 20) & (df['Promedio'] > 8)]
print(f"\nEstudiantes con Edad > 20 y Promedio > 8: {len(filtered_df)}")
print(filtered_df)  

# Manipulación de datos: Añadir columna de Aprobación
df['Aprobado'] = np.where(df['Promedio'] >= 8, 'Sí', 'No')
print("\n=== DataFrame con Columna de Aprobación ===")

print(df.head())



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
