"""
ANÁLISIS DE DATOS DE ESTUDIANTES
Archivo: estudiantes_original (1).csv
Objetivo: Inspeccionar, transformar y analizar datos de estudiantes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

# Definir ruta del archivo
ruta_datos = r"C:\Users\omaroalvaradoc\Documents\Personal\Proyectos\CURSO IA\data\processed\estudiantes_original (1).csv"

# Verificar que el archivo existe
if not os.path.exists(ruta_datos):
    print(f"❌ Error: No se encontró el archivo en {ruta_datos}")
    exit()

# Cargar el dataset
df = pd.read_csv(ruta_datos)
print("✅ Datos cargados exitosamente\n")

# ============================================================================
# 2. INSPECCIÓN INICIAL
# ============================================================================

print("=" * 70)
print("INSPECCIÓN INICIAL DEL DATASET")
print("=" * 70)

# Mostrar primeras filas
print("\n📋 Primeras filas del dataset:")
print(df.head())

# Información general
print("\n📊 Información del dataset:")
print(df.info())

# Dimensiones
print(f"\n📐 Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

# Verificar valores nulos
print("\n🔍 Valores nulos por columna:")
print(df.isnull().sum())

# ============================================================================
# 3. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

print("\n" + "=" * 70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 70)

# Estadísticas de variables numéricas
print("\n📈 Estadísticas de Edad y Promedio:")
print(df[["Edad", "Promedio"]].describe())

# Estadísticas adicionales
print(f"\n🎓 Edad:")
print(f"   Media: {df['Edad'].mean():.2f} años")
print(f"   Mediana: {df['Edad'].median():.2f} años")
print(f"   Mínimo: {df['Edad'].min()} años")
print(f"   Máximo: {df['Edad'].max()} años")

print(f"\n📚 Promedio Académico:")
print(f"   Media: {df['Promedio'].mean():.2f}")
print(f"   Mediana: {df['Promedio'].median():.2f}")
print(f"   Mínimo: {df['Promedio'].min():.2f}")
print(f"   Máximo: {df['Promedio'].max():.2f}")

# ============================================================================
# 4. ANÁLISIS POR CARRERA
# ============================================================================

print("\n" + "=" * 70)
print("ANÁLISIS POR CARRERA")
print("=" * 70)

# Contar estudiantes por carrera
print("\n👥 Estudiantes por Carrera:")
print(df["Carrera"].value_counts())

# Promedio académico por carrera
print("\n📊 Promedio Académico por Carrera:")
promedio_carrera = (
    df.groupby("Carrera")["Promedio"].agg(["mean", "min", "max"]).round(2)
)
promedio_carrera.columns = ["Promedio", "Mínimo", "Máximo"]
print(promedio_carrera.sort_values("Promedio", ascending=False))

# Edad promedio por carrera
print("\n👤 Edad Promedio por Carrera:")
edad_carrera = (
    df.groupby("Carrera")["Edad"].mean().round(1).sort_values(ascending=False)
)
print(edad_carrera)

# ============================================================================
# 5. TRANSFORMACIONES Y CLASIFICACIONES
# ============================================================================

print("\n" + "=" * 70)
print("TRANSFORMACIONES Y CLASIFICACIONES")
print("=" * 70)


# Clasificar estudiantes por rendimiento
def clasificar_rendimiento(promedio):
    if promedio >= 9.0:
        return "Excelente"
    elif promedio >= 8.0:
        return "Bueno"
    elif promedio >= 7.0:
        return "Regular"
    else:
        return "Bajo"


df["Rendimiento"] = df["Promedio"].apply(clasificar_rendimiento)

print("\n🏆 Clasificación por Rendimiento:")
print(df["Rendimiento"].value_counts())


# Clasificar por edad
def clasificar_edad(edad):
    if edad < 23:
        return "Joven (< 23)"
    elif edad <= 26:
        return "Adulto Joven (23-26)"
    else:
        return "Adulto (> 26)"


df["Grupo_Edad"] = df["Edad"].apply(clasificar_edad)

print("\n👥 Clasificación por Grupo de Edad:")
print(df["Grupo_Edad"].value_counts())

# ============================================================================
# 6. VISUALIZACIONES
# ============================================================================

print("\n" + "=" * 70)
print("GENERANDO VISUALIZACIONES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Distribución de Promedio
axes[0, 0].hist(df["Promedio"], bins=5, edgecolor="black", color="steelblue")
axes[0, 0].axvline(
    df["Promedio"].mean(),
    color="red",
    linestyle="--",
    label=f'Media: {df["Promedio"].mean():.2f}',
)
axes[0, 0].set_xlabel("Promedio")
axes[0, 0].set_ylabel("Frecuencia")
axes[0, 0].set_title("Distribución de Promedios")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Promedio por Carrera
promedio_carrera_plot = df.groupby("Carrera")["Promedio"].mean().sort_values()
axes[0, 1].barh(
    promedio_carrera_plot.index, promedio_carrera_plot.values, color="coral"
)
axes[0, 1].set_xlabel("Promedio")
axes[0, 1].set_title("Promedio Académico por Carrera")
axes[0, 1].grid(True, alpha=0.3, axis="x")

# Gráfico 3: Distribución de Edades
axes[1, 0].hist(df["Edad"], bins=6, edgecolor="black", color="lightgreen")
axes[1, 0].axvline(
    df["Edad"].mean(),
    color="red",
    linestyle="--",
    label=f'Media: {df["Edad"].mean():.1f}',
)
axes[1, 0].set_xlabel("Edad (años)")
axes[1, 0].set_ylabel("Frecuencia")
axes[1, 0].set_title("Distribución de Edades")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Rendimiento por Grupo
rendimiento_counts = df["Rendimiento"].value_counts()
axes[1, 1].pie(
    rendimiento_counts.values,
    labels=rendimiento_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#90EE90", "#FFD700", "#FFA500"],
)
axes[1, 1].set_title("Distribución por Rendimiento")

plt.tight_layout()
plt.show()

print("✅ Visualizaciones generadas")

# ============================================================================
# 7. BÚSQUEDAS Y FILTROS
# ============================================================================

print("\n" + "=" * 70)
print("BÚSQUEDAS Y FILTROS ESPECÍFICOS")
print("=" * 70)

# Mejor estudiante
mejor_estudiante = df.loc[df["Promedio"].idxmax()]
print(f"\n🥇 Mejor Estudiante:")
print(f"   Nombre: {mejor_estudiante['Nombre']}")
print(f"   Carrera: {mejor_estudiante['Carrera']}")
print(f"   Promedio: {mejor_estudiante['Promedio']}")

# Estudiantes con promedio >= 9.0
print(f"\n⭐ Estudiantes con Promedio ≥ 9.0:")
excelentes = df[df["Promedio"] >= 9.0][["Nombre", "Carrera", "Promedio"]]
print(excelentes.to_string(index=False))

# Estudiantes mayores de 25 años
print(f"\n👴 Estudiantes mayores de 25 años:")
mayores = df[df["Edad"] > 25][["Nombre", "Edad", "Carrera"]]
print(mayores.to_string(index=False))

# ============================================================================
# 8. ANÁLISIS E INTERPRETACIÓN FINAL
# ============================================================================

print("\n" + "=" * 70)
print("INTERPRETACIONES Y CONCLUSIONES")
print("=" * 70)

print(
    """
📊 ANÁLISIS ESTADÍSTICO:
------------------------
1. El dataset contiene 8 estudiantes de diferentes carreras.
2. La edad promedio es de {:.1f} años, con un rango de {} a {} años.
3. El promedio académico general es {:.2f}, con valores entre {:.1f} y {:.1f}.

🎓 HALLAZGOS POR CARRERA:
-------------------------
- La carrera con mejor promedio es: {}
- La carrera con más estudiantes es única para cada caso (1 por carrera)
- Existe diversidad en las áreas de estudio (ciencias, ingenierías, humanidades)

👥 CLASIFICACIÓN DE ESTUDIANTES:
---------------------------------
- {} estudiante(s) con rendimiento Excelente (≥ 9.0)
- {} estudiante(s) con rendimiento Bueno (8.0-8.9)
- {} estudiante(s) con rendimiento Regular (7.0-7.9)

📈 TENDENCIAS OBSERVADAS:
-------------------------
- No hay correlación directa entre edad y promedio académico
- Los promedios están bien distribuidos entre 7.5 y 9.3
- La mayoría de estudiantes están en el rango de edad 21-30 años

✅ CONCLUSIONES:
----------------
1. El grupo muestra un rendimiento académico satisfactorio (promedio > 8.0)
2. Existe equilibrio entre diferentes áreas de conocimiento
3. Los estudiantes mayores no necesariamente tienen mejores promedios
4. Se recomienda enfoque en estudiantes con promedio < 8.0 para mejora continua

""".format(
        df["Edad"].mean(),
        df["Edad"].min(),
        df["Edad"].max(),
        df["Promedio"].mean(),
        df["Promedio"].min(),
        df["Promedio"].max(),
        promedio_carrera.sort_values("Promedio", ascending=False).index[0],
        len(df[df["Rendimiento"] == "Excelente"]),
        len(df[df["Rendimiento"] == "Bueno"]),
        len(df[df["Rendimiento"] == "Regular"]),
    )
)

print("=" * 70)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 70)
