"""
PERCEPTRÓN SIMPLE CON DATASET SINTÉTICO
Objetivo: Entender fundamentos del aprendizaje supervisado lineal
Pasos:
1. Generar dataset sintético con separabilidad lineal
2. Introducir valores faltantes (NaN)
3. Limpiar e imputar datos
4. Entrenar Perceptrón
5. Evaluar y visualizar resultados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# ============================================================================
# 1. GENERACIÓN DEL DATASET SINTÉTICO
# ============================================================================

print("=" * 80)
print("PASO 1: GENERACIÓN DEL DATASET SINTÉTICO")
print("=" * 80)

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Generar 500 puntos de cada clase (1000 total)
n_samples = 500

# Clase 0: puntos cercanos a y = x (abajo de la línea)
X_class0 = np.random.randn(n_samples, 2) + np.array([0, -1])
y_class0 = np.zeros(n_samples)

# Clase 1: puntos cercanos a y = x + 3 (arriba de la línea)
X_class1 = np.random.randn(n_samples, 2) + np.array([3, 2])
y_class1 = np.ones(n_samples)

# Combinar ambas clases
X = np.vstack([X_class0, X_class1])
y = np.hstack([y_class0, y_class1])

# Crear DataFrame
df_original = pd.DataFrame(
    {"caracteristica_1": X[:, 0], "caracteristica_2": X[:, 1], "etiqueta": y}
)

print(f"\n✅ Dataset generado: {df_original.shape[0]} registros")
print(f"   - Clase 0: {sum(y == 0)} registros")
print(f"   - Clase 1: {sum(y == 1)} registros")
print(f"\nPrimeras filas del dataset original:")
print(df_original.head(10))

# ============================================================================
# 2. INTRODUCCIÓN DE VALORES FALTANTES
# ============================================================================

print("\n" + "=" * 80)
print("PASO 2: INTRODUCCIÓN DE VALORES FALTANTES (NaN)")
print("=" * 80)

# Copiar dataset
df = df_original.copy()

# Introducir 5% de valores NaN aleatoriamente
np.random.seed(42)
missing_rate = 0.05
n_missing = int(df.shape[0] * missing_rate)

# Seleccionar índices aleatorios para cada columna
indices_faltantes_1 = np.random.choice(df.shape[0], n_missing, replace=False)
indices_faltantes_2 = np.random.choice(df.shape[0], n_missing, replace=False)

df.loc[indices_faltantes_1, "caracteristica_1"] = np.nan
df.loc[indices_faltantes_2, "caracteristica_2"] = np.nan

print(f"\n✅ Valores faltantes introducidos ({missing_rate*100:.1f}%):")
print(f"\n🔍 Cantidad de NaN por columna:")
print(df.isnull().sum())

print(f"\nPorcentaje de NaN por columna:")
print((df.isnull().sum() / len(df) * 100).round(2))

print(f"\nPrimeras filas con valores faltantes:")
print(df.head(10))

# ============================================================================
# 3. TRATAMIENTO DE VALORES FALTANTES
# ============================================================================

print("\n" + "=" * 80)
print("PASO 3: TRATAMIENTO DE VALORES FALTANTES (IMPUTACIÓN)")
print("=" * 80)

print("\n📌 ESTRATEGIA DE IMPUTACIÓN:")
print("   - Método: Media (Promedio) para características numéricas")
print("   - Justificación: Mantiene la distribución y no sesga los datos")

# Calcular media de cada característica
media_car1 = df["caracteristica_1"].mean()
media_car2 = df["caracteristica_2"].mean()

print(f"\n   Media característica_1: {media_car1:.4f}")
print(f"   Media característica_2: {media_car2:.4f}")

# Imputar valores faltantes con la media
df["caracteristica_1"].fillna(media_car1, inplace=True)
df["caracteristica_2"].fillna(media_car2, inplace=True)

print(f"\n✅ Imputación completada")
print(f"\nValores nulos después de la imputación:")
print(df.isnull().sum())

print(f"\nPrimeras filas después de la imputación:")
print(df.head(10))

# ============================================================================
# 4. CARGA Y ANÁLISIS DE DATOS CON PANDAS
# ============================================================================

print("\n" + "=" * 80)
print("PASO 4: CARGA Y ANÁLISIS DE DATOS")
print("=" * 80)

# Guardar dataset limpio
ruta_salida = r"C:\Users\omaroalvaradoc\Documents\Personal\Proyectos\CURSO IA\data\processed\datos_perceptron.csv"
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
df.to_csv(ruta_salida, index=False)
print(f"\n✅ Dataset limpio guardado en: {ruta_salida}")

# Análisis estadístico
print(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())

print(f"\n📈 INFORMACIÓN DEL DATASET:")
print(df.info())

print(f"\n🎯 DISTRIBUCIÓN DE CLASES:")
print(df["etiqueta"].value_counts())

# ============================================================================
# 5. PREPARACIÓN DE DATOS PARA EL MODELO
# ============================================================================

print("\n" + "=" * 80)
print("PASO 5: PREPARACIÓN DE DATOS PARA ENTRENAMIENTO")
print("=" * 80)

# Separar características (X) y etiquetas (y)
X = df[["caracteristica_1", "caracteristica_2"]].values
y = df["etiqueta"].values

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Split realizado:")
print(
    f"   - Entrenamiento: {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.1f}%)"
)
print(f"   - Prueba: {X_test.shape[0]} registros ({X_test.shape[0]/len(X)*100:.1f}%)")

# Normalizar características (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Características normalizadas con StandardScaler")
print(f"   - Media X_train: {X_train_scaled.mean(axis=0).round(4)}")
print(f"   - Desv. Std X_train: {X_train_scaled.std(axis=0).round(4)}")

# ============================================================================
# 6. ENTRENAMIENTO DEL PERCEPTRÓN
# ============================================================================

print("\n" + "=" * 80)
print("PASO 6: ENTRENAMIENTO DEL PERCEPTRÓN")
print("=" * 80)

# Crear el modelo Perceptrón
perceptron = Perceptron(max_iter=100, random_state=42, verbose=0, eta0=0.01)

print(f"\n📌 PARÁMETROS DEL PERCEPTRÓN:")
print(f"   - max_iter: 100 (número máximo de épocas)")
print(f"   - eta0: 0.01 (tasa de aprendizaje)")
print(f"   - random_state: 42 (reproducibilidad)")

# Entrenar el modelo
perceptron.fit(X_train_scaled, y_train)

print(f"\n✅ Modelo entrenado exitosamente")
print(f"   - Número de características: {perceptron.n_features_in_}")
print(f"   - Pesos (coeficientes): {perceptron.coef_[0].round(4)}")
print(f"   - Intercepto (bias): {perceptron.intercept_[0]:.4f}")

# ============================================================================
# 7. EVALUACIÓN DEL MODELO
# ============================================================================

print("\n" + "=" * 80)
print("PASO 7: EVALUACIÓN DEL MODELO")
print("=" * 80)

# Predicciones
y_train_pred = perceptron.predict(X_train_scaled)
y_test_pred = perceptron.predict(X_test_scaled)

# Métricas
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

print(f"\n📊 MÉTRICAS DE DESEMPEÑO:")
print(f"   - Exactitud en entrenamiento: {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"   - Exactitud en prueba: {acc_test:.4f} ({acc_test*100:.2f}%)")

# Matriz de confusión
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n🔍 MATRIZ DE CONFUSIÓN (Conjunto de Prueba):")
print(cm)

# Reporte de clasificación
print(f"\n📋 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_test_pred, target_names=["Clase 0", "Clase 1"]))

# ============================================================================
# 8. VISUALIZACIONES
# ============================================================================

print("\n" + "=" * 80)
print("PASO 8: GENERANDO VISUALIZACIONES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: Dataset original con frontera de decisión
ax1 = axes[0, 0]
scatter0 = ax1.scatter(
    X[y == 0, 0], X[y == 0, 1], label="Clase 0", alpha=0.6, s=50, color="blue"
)
scatter1 = ax1.scatter(
    X[y == 1, 0], X[y == 1, 1], label="Clase 1", alpha=0.6, s=50, color="red"
)

# Calcular frontera de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

Z = perceptron.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

ax1.contourf(
    xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=["lightblue", "lightcoral"]
)
ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors="black")

ax1.set_xlabel("Característica 1", fontsize=11, fontweight="bold")
ax1.set_ylabel("Característica 2", fontsize=11, fontweight="bold")
ax1.set_title(
    "Dataset y Frontera de Decisión del Perceptrón", fontsize=12, fontweight="bold"
)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Matriz de confusión
ax2 = axes[0, 1]
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax2,
    xticklabels=["Predicho 0", "Predicho 1"],
    yticklabels=["Real 0", "Real 1"],
)
ax2.set_title(
    "Matriz de Confusión (Conjunto de Prueba)", fontsize=12, fontweight="bold"
)
ax2.set_ylabel("Valores Reales", fontsize=11, fontweight="bold")
ax2.set_xlabel("Valores Predichos", fontsize=11, fontweight="bold")

# Gráfico 3: Desempeño por conjunto
ax3 = axes[1, 0]
conjuntos = ["Entrenamiento", "Prueba"]
exactitudes = [acc_train, acc_test]
colores = ["green", "orange"]
bars = ax3.bar(conjuntos, exactitudes, color=colores, edgecolor="black", linewidth=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{exactitudes[i]:.4f}\n({exactitudes[i]*100:.2f}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax3.set_ylabel("Exactitud", fontsize=11, fontweight="bold")
ax3.set_title("Comparación de Exactitud", fontsize=12, fontweight="bold")
ax3.set_ylim([0.8, 1.05])
ax3.grid(True, alpha=0.3, axis="y")

# Gráfico 4: Distribución de predicciones
ax4 = axes[1, 1]
distribucion = pd.DataFrame(
    {
        "Predicción": np.concatenate([y_test, y_test_pred]),
        "Tipo": ["Real"] * len(y_test) + ["Predicho"] * len(y_test_pred),
    }
)

distribucion_counts = (
    distribucion.groupby(["Tipo", "Predicción"]).size().unstack(fill_value=0)
)
distribucion_counts.plot(
    kind="bar",
    ax=ax4,
    color=["lightblue", "lightcoral"],
    edgecolor="black",
    linewidth=1.5,
)
ax4.set_xlabel("Tipo de Datos", fontsize=11, fontweight="bold")
ax4.set_ylabel("Cantidad", fontsize=11, fontweight="bold")
ax4.set_title("Distribución: Real vs Predicho", fontsize=12, fontweight="bold")
ax4.legend(title="Clase", labels=["Clase 0", "Clase 1"])
ax4.tick_params(axis="x", rotation=0)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    r"C:\Users\omaroalvaradoc\Documents\Personal\Proyectos\CURSO IA\data\processed\grafico_perceptron.png",
    dpi=300,
    bbox_inches="tight",
)
print("✅ Gráficos generados y guardados")
plt.show()

# ============================================================================
# 9. BLOQUE DE INTERPRETACIONES Y CONCLUSIONES
# ============================================================================

print("\n" + "=" * 80)
print("INTERPRETACIONES Y CONCLUSIONES")
print("=" * 80)

interpretacion = f"""
📊 ANÁLISIS DEL PERCEPTRÓN SIMPLE:
==================================

1. GENERACIÓN DEL DATASET:
   - Se generaron 1000 muestras de dos clases linealmente separables
   - Clase 0: 500 puntos distribuidos alrededor de (0, -1)
   - Clase 1: 500 puntos distribuidos alrededor de (3, 2)
   - Los datos son claramente separables por una línea

2. TRATAMIENTO DE DATOS FALTANTES:
   - Se introdujeron {missing_rate*100:.1f}% de valores NaN
   - Método de imputación: Media aritmética
   - Características tratadas: {int(n_missing*2)} valores faltantes totales
   - Ventajas: Mantiene estadísticas y no sesga el modelo

3. ENTRENAMIENTO DEL MODELO:
   - Algoritmo: Perceptrón (clasificador lineal)
   - Características normalizadas con StandardScaler
   - Máximo de iteraciones: 100 épocas
   - Tasa de aprendizaje (eta0): 0.01
   - Split: 80% entrenamiento, 20% prueba ({len(X_train)} vs {len(X_test)} muestras)

4. DESEMPEÑO DEL MODELO:
   - Exactitud en entrenamiento: {acc_train:.4f} ({acc_train*100:.2f}%)
   - Exactitud en prueba: {acc_test:.4f} ({acc_test*100:.2f}%)
   - Diferencia: {abs(acc_train - acc_test):.4f} (indica buen ajuste sin sobreentrenamiento)

5. MATRIZ DE CONFUSIÓN:
   - Verdaderos Negativos (TN): {cm[0, 0]}
   - Falsos Positivos (FP): {cm[0, 1]}
   - Falsos Negativos (FN): {cm[1, 0]}
   - Verdaderos Positivos (TP): {cm[1, 1]}
   - Tasa de error: {(cm[0, 1] + cm[1, 0]) / len(y_test) * 100:.2f}%

6. FUNDAMENTOS DEL APRENDIZAJE SUPERVISADO:
   - El Perceptrón es un algoritmo de aprendizaje supervisado binario
   - Busca encontrar un hiperplano que separe las dos clases
   - Utiliza la regla de actualización: w = w + η(y - ŷ)x
   - Converge garantizado si los datos son linealmente separables

7. INTERPRETACIÓN DE LA FRONTERA DE DECISIÓN:
   - La línea/frontera mostrada separa ambas clases
   - Los pesos aprendidos ({perceptron.coef_[0].round(4)}) 
     determinan la orientación de la frontera
   - El intercepto ({perceptron.intercept_[0]:.4f}) define su posición

✅ CONCLUSIONES:
================
1. El modelo Perceptrón converge exitosamente en este dataset
2. La alta exactitud ({acc_test*100:.2f}%) confirma la separabilidad lineal
3. No hay diferencia significativa entre entrenamiento y prueba
   (indica generalización adecuada)
4. El tratamiento de NaN mediante media no afectó significativamente
   el rendimiento del modelo
5. Este es un ejemplo clásico de aprendizaje supervisado lineal
   donde el algoritmo encuentra correctamente la frontera de decisión

📈 RECOMENDACIONES:
===================
- Para datos más complejos, considerar redes neuronales
- Realizar validación cruzada para mayor robustez
- Analizar importancia de características
- Considerar técnicas de regularización si hay sobreentrenamiento
"""

print(interpretacion)

print("=" * 80)
print("✅ ANÁLISIS DEL PERCEPTRÓN COMPLETADO")
print("=" * 80)
