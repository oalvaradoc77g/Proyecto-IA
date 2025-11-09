import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from core.modelo_hibrido import ModeloHibrido
from core.modelo_series_temporales import ModeloSeriesTiempo
from utils.data_loader import DataLoader

import json

warnings.filterwarnings("ignore")


class ModeloHipoteca:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        # ✅ MEJORADO: Incluir TODAS las variables relevantes
        self.columnas_numericas = [
            "capital",
            "gastos_fijos",
            "tasa_uvr",
            "tasa_dtf",
            "inflacion_ipc",
        ]
        self.columnas_categoricas = ["tipo_pago"]
        self.todas_columnas = self.columnas_numericas + self.columnas_categoricas
        self.ultimas_predicciones = {}
        self.historial_errores = []
        self.data_loader = DataLoader()

    def cargar_datos(self, path):
        """Carga y valida los datos con mejor manejo de errores"""
        try:
            df = pd.read_excel(path)
            print(f"📊 Datos cargados: {len(df)} registros, {df.shape[1]} columnas")

            # Crear nueva variable gastos_fijos
            df["gastos_fijos"] = df["intereses"] + df["seguros"]

            # Agregar tipo_pago si no existe
            if "tipo_pago" not in df.columns:
                print("ℹ️ Agregando columna tipo_pago")
                # Calcular umbral usando Series en lugar de acceder directamente
                capital_medio = df["capital"].mean()
                df["tipo_pago"] = df["capital"].apply(
                    lambda x: "Abono extra" if x > capital_medio * 1.2 else "Ordinario"
                )

            # Verificar y convertir índice temporal
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"])
                df.set_index("fecha", inplace=True)
            else:
                print("⚠️ Creando índice temporal automático")
                df.index = pd.date_range(start="2025-01-01", periods=len(df), freq="M")

            # Enriquecer con datos macroeconómicos simulados
            print("🔄 Agregando variables macroeconómicas simuladas...")
            df_enriquecido = self.data_loader.enriquecer_datos(df)

            # Asegurar que todas las columnas necesarias existen
            columnas_requeridas = [
                "capital",
                "gastos_fijos",
                "total_mensual",
                "tipo_pago",
            ] + ["tasa_uvr", "tasa_dtf", "inflacion_ipc"]

            for col in columnas_requeridas:
                if col not in df_enriquecido.columns:
                    print(f"⚠️ Agregando columna faltante: {col}")
                    if col in self.data_loader.valores_actuales:
                        df_enriquecido[col] = self.data_loader.valores_actuales[col]
                    elif col == "tipo_pago":
                        df_enriquecido[col] = "Ordinario"  # valor por defecto

            return df_enriquecido

        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None

    def analizar_multicolinealidad(self, df):
        """Análisis completo de multicolinealidad"""
        # ✅ CORREGIDO: Filtrar columnas con varianza > 0
        X = df[self.columnas_numericas].copy()

        # Eliminar columnas constantes (sin variación)
        columnas_con_varianza = X.columns[X.std() > 0]
        X = X[columnas_con_varianza]

        if len(X.columns) < 2:
            print(
                "⚠️ No hay suficientes variables con varianza para análisis de multicolinealidad"
            )
            return None

        # Matriz de correlación
        print("\n🔗 Matriz de Correlación:")
        corr_matrix = X.corr()
        print(corr_matrix.round(3))

        # VIF (solo si hay más de 1 variable)
        if len(X.columns) > 1:
            vif = self.calcular_vif(X)
            print("\n📊 Factor de Inflación de Varianza (VIF):")
            for var, vif_val in vif.items():
                status = "⚠️ ALTO" if vif_val > 10 else "✅ OK"
                print(f"  {var}: {vif_val:.2f} {status}")
        else:
            vif = None
            print("\n⚠️ Solo hay una variable con varianza, VIF no calculable")

        return vif

    def calcular_vif(self, X):
        """Calcula VIF con validación"""
        try:
            X_const = sm.add_constant(X)
            vif_data = pd.Series(
                [
                    variance_inflation_factor(X_const.values, i)
                    for i in range(1, X_const.shape[1])
                ],
                index=X.columns,
            )
            return vif_data
        except Exception as e:
            print(f"⚠️ Error calculando VIF: {e}")
            return pd.Series()

    def preparar_datos(self, df, temporal=False, test_size=0.2):
        """✅ MEJORADO: Preparar datos con variables categóricas"""
        # One-hot encoding para tipo_pago
        df_encoded = pd.get_dummies(df, columns=["tipo_pago"], drop_first=True)

        # Seleccionar columnas numéricas + dummies
        columnas_modelo = self.columnas_numericas + [
            col for col in df_encoded.columns if "tipo_pago_" in col
        ]

        X = df_encoded[columnas_modelo]
        y = df_encoded["total_mensual"]

        if temporal and len(df) > 10:
            n_test = max(1, int(test_size * len(df)))
            X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
            y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
            print(f"⏰ Split temporal: Train={len(X_train)}, Test={len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=not temporal
            )
            print(f"📊 Split aleatorio: Train={len(X_train)}, Test={len(X_test)}")

        # Guardar columnas para predicción
        self.columnas_modelo = columnas_modelo
        return X_train, X_test, y_train, y_test

    def crear_modelo(self, X_train, y_train):
        """✅ MEJORADO: Modelo con regularización adaptativa"""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import ElasticNetCV

        # Opción 1: Ridge (tu actual)
        alphas = np.logspace(-3, 3, 50)
        pipeline_ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=5),
                ),
            ]
        )

        # ✅ NUEVO: Opción 2: ElasticNet (mejor para múltiples variables)
        pipeline_elastic = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "elastic",
                    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv=5),
                ),
            ]
        )

        # ✅ NUEVO: Opción 3: Gradient Boosting (captura no-linealidades)
        pipeline_gb = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gb",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Entrenar todos y comparar
        modelos = {
            "Ridge": pipeline_ridge,
            "ElasticNet": pipeline_elastic,
            "GradientBoosting": pipeline_gb,
        }

        scores = {}
        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            score = cross_val_score(modelo, X_train, y_train, cv=5, scoring="r2").mean()
            scores[nombre] = score
            print(f"   {nombre}: R² = {score:.4f}")

        # Seleccionar el mejor
        mejor_modelo = max(scores, key=scores.get)
        self.modelo = modelos[mejor_modelo]
        print(
            f"\n✅ Modelo seleccionado: {mejor_modelo} (R² = {scores[mejor_modelo]:.4f})"
        )

        return self.modelo

    def evaluar_modelo(self, modelo, X_test, y_test):
        """Evaluación completa del modelo"""
        y_pred = modelo.predict(X_test)

        # Métricas múltiples
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Porcentaje de error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print("\n📊 EVALUACIÓN DEL MODELO:")
        print(f"  R²: {r2:.4f}")
        print(f"  MSE: {mse:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return y_pred, {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

    def crear_graficos(self, df, y_test, y_pred, X_test):
        """Gráficos más informativos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Predicho vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        axes[0, 0].set_xlabel("Valor Real")
        axes[0, 0].set_ylabel("Valor Predicho")
        axes[0, 0].set_title("Predicciones vs Valores Reales")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuales
        residuos = y_test - y_pred
        axes[0, 1].plot(residuos, "o-", alpha=0.7)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Índice")
        axes[0, 1].set_ylabel("Residuales")
        axes[0, 1].set_title("Análisis de Residuales")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribución de residuales
        axes[1, 0].hist(residuos, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(x=0, color="r", linestyle="--")
        axes[1, 0].set_xlabel("Residuales")
        axes[1, 0].set_ylabel("Frecuencia")
        axes[1, 0].set_title("Distribución de Residuales")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Serie temporal (si aplica)
        if len(y_test) > 1:
            axes[1, 1].plot(y_test.values, label="Real", marker="o")
            axes[1, 1].plot(y_pred, label="Predicho", marker="s")
            axes[1, 1].set_xlabel("Tiempo")
            axes[1, 1].set_ylabel("Total Mensual")
            axes[1, 1].set_title("Evolución Temporal")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return residuos

    def predecir(self, capital, gastos_fijos, seguros):
        """Predicción con validación"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None

        try:
            ejemplo = np.array([[capital, gastos_fijos]])
            pred = self.modelo.predict(ejemplo)[0]
            print(f"\n🎯 PREDICCIÓN:")
            print(f"   Capital: {capital:,.2f}")
            print(f"   Gastos Fijos: {gastos_fijos:,.2f}")
            print(f"   Total Mensual Predicho: {pred:,.2f}")
            return pred
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None

    def validar_prediccion(self, prediccion, valor_real):
        """Valida predicción contra valor real y decide reentrenamiento"""
        error = abs((valor_real - prediccion) / valor_real)
        self.historial_errores.append(error)

        print(f"\n🎯 VALIDACIÓN DE PREDICCIÓN:")
        print(f"   Predicho: ${prediccion:,.2f}")
        print(f"   Real: ${valor_real:,.2f}")
        print(f"   Error: {error:.2%}")

        if error > 0.15:  # Error > 15%
            print("⚠️ Error significativo detectado")
            return False
        return True

    def guardar_predicciones(self, predicciones, ruta="predicciones.json"):
        """Guarda predicciones con metadata"""
        try:
            datos = {
                "fecha_generacion": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predicciones": (
                    predicciones.to_dict()
                    if hasattr(predicciones, "to_dict")
                    else predicciones
                ),
                "metricas": self.metricas if hasattr(self, "metricas") else {},
                "error_promedio": (
                    np.mean(self.historial_errores) if self.historial_errores else None
                ),
            }

            with open(ruta, "w") as f:
                json.dump(datos, f, indent=4)
            print(f"✅ Predicciones guardadas en {ruta}")

        except Exception as e:
            print(f"❌ Error guardando predicciones: {e}")

    def analizar_importancia_variables(self, X_train, y_train):
        """✅ NUEVO: Análisis de importancia de variables"""
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        importancias = pd.DataFrame(
            {"variable": X_train.columns, "importancia": rf.feature_importances_}
        ).sort_values("importancia", ascending=False)

        print("\n📊 IMPORTANCIA DE VARIABLES:")
        for _, row in importancias.iterrows():
            print(f"   {row['variable']}: {row['importancia']:.4f}")

        # Visualizar
        plt.figure(figsize=(10, 6))
        plt.barh(importancias["variable"], importancias["importancia"])
        plt.xlabel("Importancia")
        plt.title("Importancia de Variables en el Modelo")
        plt.tight_layout()
        plt.show()

        return importancias

    def predecir_con_escenarios(self, capital, gastos_fijos, tipo_pago="Ordinario"):
        """✅ NUEVO: Predicción con escenarios macroeconómicos"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None

        # Escenarios macroeconómicos
        escenarios = {
            "Optimista": {"tasa_uvr": 11.5, "tasa_dtf": 10.0, "inflacion_ipc": 3.0},
            "Base": {"tasa_uvr": 12.0, "tasa_dtf": 10.5, "inflacion_ipc": 5.5},
            "Pesimista": {"tasa_uvr": 13.0, "tasa_dtf": 11.5, "inflacion_ipc": 8.0},
        }

        print(f"\n🎯 PREDICCIONES BAJO DIFERENTES ESCENARIOS:")
        print(f"   Capital: ${capital:,.2f}")
        print(f"   Gastos Fijos: ${gastos_fijos:,.2f}")
        print(f"   Tipo Pago: {tipo_pago}\n")

        resultados = {}
        for nombre, macro in escenarios.items():
            # Crear datos de entrada
            datos = {
                "capital": capital,
                "gastos_fijos": gastos_fijos,
                "tasa_uvr": macro["tasa_uvr"],
                "tasa_dtf": macro["tasa_dtf"],
                "inflacion_ipc": macro["inflacion_ipc"],
            }

            # One-hot encoding para tipo_pago
            if tipo_pago == "Abono extra":
                datos["tipo_pago_Ordinario"] = 0
            else:
                datos["tipo_pago_Ordinario"] = 1

            # Asegurar orden correcto de columnas
            ejemplo = pd.DataFrame([datos])[self.columnas_modelo]

            pred = self.modelo.predict(ejemplo)[0]
            resultados[nombre] = pred

            print(
                f"   {nombre:12s}: ${pred:,.2f} (UVR: {macro['tasa_uvr']}%, DTF: {macro['tasa_dtf']}%, IPC: {macro['inflacion_ipc']}%)"
            )

        return resultados

    def optimizar_abono_extra(self, df):
        """✅ NUEVO: Recomendar cuándo hacer abono extra"""
        # Calcular diferencia entre abono extra y ordinario
        df_abono = df[df["tipo_pago"] == "Abono extra"].copy()
        df_ordinario = df[df["tipo_pago"] == "Ordinario"].copy()

        if len(df_abono) > 0 and len(df_ordinario) > 0:
            ahorro_promedio = (
                df_ordinario["total_mensual"].mean() - df_abono["total_mensual"].mean()
            )

            print(f"\n💰 ANÁLISIS DE ABONO EXTRA:")
            print(
                f"   Cuota promedio ordinaria: ${df_ordinario['total_mensual'].mean():,.2f}"
            )
            print(
                f"   Cuota promedio con abono: ${df_abono['total_mensual'].mean():,.2f}"
            )
            print(f"   Ahorro estimado: ${ahorro_promedio:,.2f}/mes")

            # Recomendar meses óptimos (baja inflación, baja UVR)
            df_sorted = df.sort_values(["inflacion_ipc", "tasa_uvr"])
            mejores_meses = df_sorted.head(3).index

            print(f"\n✅ MESES RECOMENDADOS PARA ABONO EXTRA:")
            for mes in mejores_meses:
                print(
                    f"   {mes.strftime('%B %Y')}: IPC={df.loc[mes, 'inflacion_ipc']:.2f}%, UVR={df.loc[mes, 'tasa_uvr']:.2f}%"
                )

    def estimar_proximo_extracto(self, df, cuota_total_predicha=None):
        """✅ NUEVO: Estima el próximo extracto con aporte a capital y meses restantes"""

        # Obtener últimos valores conocidos
        ultimo = df.iloc[-1]
        penultimo = df.iloc[-2] if len(df) > 1 else ultimo

        # Calcular tendencias
        capital_actual = ultimo["capital"]
        capital_anterior = penultimo["capital"]
        abono_capital_promedio = capital_anterior - capital_actual

        # Si no se proporciona cuota predicha, usar la última conocida
        if cuota_total_predicha is None:
            cuota_total_predicha = ultimo["total_mensual"]

        # Estimar gastos fijos (intereses + seguros) del próximo mes
        # Usar promedio de últimos 3 meses
        gastos_fijos_promedio = df["gastos_fijos"].tail(3).mean()

        # Calcular abono a capital estimado
        abono_capital_estimado = cuota_total_predicha - gastos_fijos_promedio

        # Capital restante después del próximo pago
        capital_proyectado = capital_actual - abono_capital_estimado

        # Estimar meses restantes
        if abono_capital_estimado > 0:
            meses_restantes = int(np.ceil(capital_proyectado / abono_capital_estimado))
        else:
            meses_restantes = float("inf")

        # Calcular fecha de finalización
        fecha_ultimo = df.index[-1]
        fecha_finalizacion = fecha_ultimo + pd.DateOffset(months=meses_restantes)

        print("\n" + "=" * 80)
        print("📋 ESTIMACIÓN DEL PRÓXIMO EXTRACTO")
        print("=" * 80)

        print(f"\n💰 SITUACIÓN ACTUAL (último extracto):")
        print(f"   Fecha: {fecha_ultimo.strftime('%B %Y')}")
        print(f"   Capital pendiente: ${capital_actual:,.2f}")
        print(f"   Cuota total: ${ultimo['total_mensual']:,.2f}")
        print(f"   └─ Capital: ${ultimo['capital']:,.2f}")
        print(f"   └─ Gastos fijos: ${ultimo['gastos_fijos']:,.2f}")

        print(f"\n🔮 PROYECCIÓN PRÓXIMO MES:")
        print(f"   Cuota total estimada: ${cuota_total_predicha:,.2f}")
        print(f"   └─ Gastos fijos estimados: ${gastos_fijos_promedio:,.2f}")
        print(f"   └─ Abono a capital estimado: ${abono_capital_estimado:,.2f}")

        print(f"\n📊 DESPUÉS DEL PRÓXIMO PAGO:")
        print(f"   Capital restante: ${capital_proyectado:,.2f}")
        print(
            f"   Reducción de deuda: ${abono_capital_estimado:,.2f} ({(abono_capital_estimado/capital_actual*100):.2f}%)"
        )

        print(f"\n📅 PROYECCIÓN DE FINALIZACIÓN:")
        print(f"   Meses restantes estimados: {meses_restantes}")
        print(
            f"   Fecha estimada de finalización: {fecha_finalizacion.strftime('%B %Y')}"
        )

        # Análisis de tendencia
        print(f"\n📈 ANÁLISIS DE TENDENCIA:")
        abonos_recientes = []
        for i in range(min(3, len(df) - 1)):
            idx = len(df) - 1 - i
            abono = df.iloc[idx - 1]["capital"] - df.iloc[idx]["capital"]
            abonos_recientes.append(abono)

        if len(abonos_recientes) > 0:
            abono_min = min(abonos_recientes)
            abono_max = max(abonos_recientes)
            print(f"   Abono a capital últimos meses:")
            print(f"   └─ Mínimo: ${abono_min:,.2f}")
            print(f"   └─ Máximo: ${abono_max:,.2f}")
            print(f"   └─ Promedio: ${np.mean(abonos_recientes):,.2f}")

        # Retornar resumen
        return {
            "fecha_proximo": fecha_ultimo + pd.DateOffset(months=1),
            "cuota_total": cuota_total_predicha,
            "gastos_fijos": gastos_fijos_promedio,
            "abono_capital": abono_capital_estimado,
            "capital_restante": capital_proyectado,
            "meses_restantes": meses_restantes,
            "fecha_finalizacion": fecha_finalizacion,
            "porcentaje_completado": (
                (df.iloc[0]["capital"] - capital_proyectado)
                / df.iloc[0]["capital"]
                * 100
            ),
        }

    def simular_abono_extra(self, df, abono_extra):
        """✅ NUEVO: Simula impacto de hacer un abono extra"""

        estimacion_normal = self.estimar_proximo_extracto(df)

        # Simular con abono extra
        capital_con_extra = estimacion_normal["capital_restante"] - abono_extra
        abono_capital_promedio = estimacion_normal["abono_capital"]

        if abono_capital_promedio > 0:
            meses_con_extra = int(np.ceil(capital_con_extra / abono_capital_promedio))
        else:
            meses_con_extra = float("inf")

        meses_ahorrados = estimacion_normal["meses_restantes"] - meses_con_extra
        intereses_ahorrados = meses_ahorrados * estimacion_normal["gastos_fijos"]

        print("\n" + "=" * 80)
        print(f"💡 SIMULACIÓN: ABONO EXTRA DE ${abono_extra:,.2f}")
        print("=" * 80)

        print(f"\n📊 COMPARACIÓN:")
        print(f"   {'Concepto':<30} {'Sin Abono Extra':>20} {'Con Abono Extra':>20}")
        print(f"   {'-'*30} {'-'*20} {'-'*20}")
        print(
            f"   {'Capital restante':<30} ${estimacion_normal['capital_restante']:>19,.2f} ${capital_con_extra:>19,.2f}"
        )
        print(
            f"   {'Meses restantes':<30} {estimacion_normal['meses_restantes']:>20} {meses_con_extra:>20}"
        )
        print(
            f"   {'Fecha finalización':<30} {estimacion_normal['fecha_finalizacion'].strftime('%m/%Y'):>20} {(estimacion_normal['fecha_finalizacion'] - pd.DateOffset(months=int(meses_ahorrados))).strftime('%m/%Y'):>20}"
        )

        print(f"\n✅ BENEFICIOS DEL ABONO EXTRA:")
        print(f"   Meses ahorrados: {meses_ahorrados}")
        print(f"   Intereses ahorrados: ${intereses_ahorrados:,.2f}")
        print(f"   ROI del abono: {(intereses_ahorrados/abono_extra*100):.2f}%")

        return {
            "capital_final": capital_con_extra,
            "meses_restantes": meses_con_extra,
            "meses_ahorrados": meses_ahorrados,
            "intereses_ahorrados": intereses_ahorrados,
            "roi": (intereses_ahorrados / abono_extra * 100),
        }


def main():
    # Modelo de regresión
    modelo_hipoteca = ModeloHipoteca()

    # Cargar datos
    path = r"C:\Users\omaroalvaradoc\Documents\Personal\hipoteca_extractos_ene_sep_2025.xlsx"
    df = modelo_hipoteca.cargar_datos(path)

    if df is None or len(df) == 0:
        print("No se pudieron cargar los datos")
        return

    # Preparar y entrenar modelo
    X_train, X_test, y_train, y_test = modelo_hipoteca.preparar_datos(df, temporal=True)

    # ✅ NUEVO: Análisis de importancia (ANTES de crear_modelo)
    importancias = modelo_hipoteca.analizar_importancia_variables(X_train, y_train)

    # ✅ CORREGIDO: Análisis de multicolinealidad (solo si hay datos)
    if len(df) > 0:
        vif = modelo_hipoteca.analizar_multicolinealidad(df)

    modelo = modelo_hipoteca.crear_modelo(X_train, y_train)

    # Evaluar
    y_pred, metricas = modelo_hipoteca.evaluar_modelo(modelo, X_test, y_test)

    # Gráficos
    residuos = modelo_hipoteca.crear_graficos(df, y_test, y_pred, X_test)

    # Predicción para octubre
    # Tomamos los últimos valores conocidos del dataset
    ultimo_registro = df.iloc[-1]
    capital_octubre = ultimo_registro["capital"]
    gastos_fijos_octubre = ultimo_registro["gastos_fijos"]

    print("\n🗓️ PREDICCIÓN PARA OCTUBRE:")
    print("Usando los últimos valores conocidos:")
    modelo_hipoteca.predecir(
        capital_octubre, gastos_fijos_octubre, 0
    )  # El tercer parámetro ya no se usa

    # Modelo de series temporales
    print("\n⏰ MODELOS DE SERIES TEMPORALES")
    modelo_temporal = ModeloSeriesTiempo()
    df_temporal = modelo_temporal.preparar_datos(df)

    # Entrenar modelos
    modelo_temporal.entrenar_arima(df_temporal)
    modelo_temporal.entrenar_prophet(df_temporal)

    # Realizar predicciones
    predicciones = modelo_temporal.predecir_proximas_cuotas(df_temporal)
    print("\n🔮 PREDICCIONES PARA LOS PRÓXIMOS 6 MESES:")
    print(predicciones)

    # Visualizar resultados
    modelo_temporal.visualizar_predicciones(df_temporal, predicciones)

    print("\n🔄 ENTRENANDO MODELO HÍBRIDO")
    modelo_hibrido = ModeloHibrido()

    try:
        # Validar columnas requeridas
        columnas_requeridas = [
            "capital",
            "gastos_fijos",
            "total_mensual",
            "tasa_uvr",
            "tasa_dtf",
            "inflacion_ipc",
            "tipo_pago",
        ]

        for col in columnas_requeridas:
            if col not in df.columns:
                print(f"⚠️ Falta columna {col}")
                return

        # Entrenamiento inicial
        if modelo_hibrido.entrenar(df):
            # 🔄 Validación con dato real
            try:
                mes_actual = df["total_mensual"].iloc[-1]  # Último valor conocido
                prediccion_anterior = modelo_hibrido.predecir_futuro(n_predicciones=1)[
                    0
                ]

                error_relativo = abs((mes_actual - prediccion_anterior) / mes_actual)

                print("\n🎯 VALIDACIÓN DEL MODELO:")
                print(f"   Valor Real: ${mes_actual:,.2f}")
                print(f"   Predicción: ${prediccion_anterior:,.2f}")
                print(f"   Error: {error_relativo:.2%}")

                if error_relativo > 0.15:  # Error mayor al 15%
                    print("\n⚠️  ERROR ALTO - Reentrenando modelo...")
                    # Agregar nuevo dato y reentrenar
                    df_actualizado = df.copy()
                    df_actualizado.loc[len(df_actualizado)] = {
                        "total_mensual": mes_actual
                    }
                    modelo_hibrido.entrenar(df_actualizado)
            except Exception as e:
                print(f"⚠️  Error en validación: {e}")

            # 🚀 DESPLIEGUE A PRODUCCIÓN
            print("\n🚀 MODELO EN PRODUCCIÓN")
            predicciones_futuras = modelo_hibrido.predecir_futuro(
                n_predicciones=6, retornar_componentes=True
            )

            if predicciones_futuras is not None:
                print("\n📊 PREDICCIONES OPERACIONALES:")
                print("=" * 50)
                for fecha, row in predicciones_futuras.iterrows():
                    print(f"   {fecha.strftime('%B %Y')}:")
                    print(f"      Predicción: ${row['prediccion_hibrida']:,.2f}")
                    if "ic_inferior" in row:
                        print(
                            f"      Rango: ${row['ic_inferior']:,.2f} - ${row['ic_superior']:,.2f}"
                        )
                print("=" * 50)

                # Guardar predicciones
                try:
                    predicciones_futuras.to_excel("predicciones_produccion.xlsx")
                    print(
                        "\n✅ Predicciones guardadas en 'predicciones_produccion.xlsx'"
                    )
                except Exception as e:
                    print(f"⚠️  Error guardando predicciones: {e}")

    except Exception as e:
        print(f"❌ Error en producción: {e}")

    # Sistema de validación y monitoreo
    if modelo_hibrido is not None and len(df) > 0:
        ultimo_valor_real = df["total_mensual"].iloc[-1]
        ultima_prediccion = modelo_hibrido.ultimas_predicciones.get(
            df.index[-1].strftime("%Y-%m"), None
        )

        if ultima_prediccion is not None:
            if not modelo_hibrido.validar_prediccion(
                ultima_prediccion, ultimo_valor_real
            ):
                print("🔄 Reentrenando modelo con nuevos datos...")
                modelo_hibrido.entrenar(df)  # Reentrenar

        # Guardar nuevas predicciones
        predicciones_futuras = modelo_hibrido.predecir_futuro(n_predicciones=3)
        if predicciones_futuras is not None:
            modelo_hibrido.guardar_predicciones(predicciones_futuras)

    # Entrenar modelo mejorado
    modelo = modelo_hipoteca.crear_modelo(X_train, y_train)

    # Evaluar
    y_pred, metricas = modelo_hipoteca.evaluar_modelo(modelo, X_test, y_test)

    # ✅ NUEVO: Predicción con escenarios
    ultimo_registro = df.iloc[-1]
    resultados_escenarios = modelo_hipoteca.predecir_con_escenarios(
        capital=ultimo_registro["capital"],
        gastos_fijos=ultimo_registro["gastos_fijos"],
        tipo_pago="Ordinario",
    )

    # ✅ NUEVO: Estimación del próximo extracto
    print("\n" + "=" * 80)
    print("🎯 ESTIMACIÓN DEL PRÓXIMO EXTRACTO")
    print("=" * 80)

    # Usar la predicción del modelo como cuota total
    cuota_predicha = y_pred[-1] if len(y_pred) > 0 else None
    estimacion = modelo_hipoteca.estimar_proximo_extracto(df, cuota_predicha)

    # ✅ NUEVO: Simulación de abonos extras
    print("\n" + "=" * 80)
    print("💰 ANÁLISIS DE ABONOS EXTRAS")
    print("=" * 80)

    # Simular diferentes montos de abono
    abonos_a_simular = [500000, 1000000, 2000000]  # Ajusta estos valores según tu caso

    for abono in abonos_a_simular:
        modelo_hipoteca.simular_abono_extra(df, abono)

    # ✅ NUEVO: Optimizar abonos
    modelo_hipoteca.optimizar_abono_extra(df)

    # ✅ NUEVO: Guardar reporte completo
    print("\n" + "=" * 80)
    print("💾 GUARDANDO REPORTE COMPLETO")
    print("=" * 80)

    reporte = {
        "fecha_generacion": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datos_actuales": {
            "capital_actual": float(df.iloc[-1]["capital"]),
            "cuota_actual": float(df.iloc[-1]["total_mensual"]),
            "gastos_fijos": float(df.iloc[-1]["gastos_fijos"]),
        },
        "estimacion_proximo_mes": {
            k: (
                float(v)
                if isinstance(v, (np.integer, np.floating))
                else v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v
            )
            for k, v in estimacion.items()
        },
        "predicciones_escenarios": {
            k: float(v) for k, v in resultados_escenarios.items()
        },
        "metricas_modelo": metricas,
    }

    try:
        import json

        with open("reporte_hipoteca.json", "w", encoding="utf-8") as f:
            json.dump(reporte, f, indent=4, ensure_ascii=False)
        print("✅ Reporte guardado en 'reporte_hipoteca.json'")
    except Exception as e:
        print(f"⚠️ Error guardando reporte: {e}")

    print("\n✅ ANÁLISIS COMPLETADO")


if __name__ == "__main__":
    main()
