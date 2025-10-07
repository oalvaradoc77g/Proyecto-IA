import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
import joblib
import json
# Fix import path - use absolute import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import DataLoader
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class ModeloHibrido:
    def __init__(self, alpha_ridge=None, orden_arima_auto=True):
        self.modelo_lineal = None
        self.modelo_arima = None
        self.scaler = StandardScaler()
        self.orden_arima = (1,1,1)
        self.alpha_ridge = alpha_ridge
        self.orden_arima_auto = orden_arima_auto
        self.metricas = {'evaluacion': {}, 'entrenamiento': {}}
        self.componentes = {}
        self.df_entrenamiento = None  # Agregar esta línea
        self.variables_macro = ['tasa_uvr', 'tasa_dtf', 'inflacion_ipc']
        self.columnas_categoricas = ['tipo_pago']
        self.exog_transformer = None  # Para almacenar transformación de exógenas
        self.data_loader = DataLoader()
        
        self.ultimas_predicciones = {}  # Añadir este atributo
        self.historial_predicciones = []  # Para tracking histórico
        self.errores_prediccion = []  # Para monitoreo de errores

    def ensure_df_exog(self, exog):
        """Devuelve exog como DataFrame y orden fijo de columnas."""
        if exog is None:
            return None
        if isinstance(exog, pd.Series):
            exog = exog.to_frame().T
        if isinstance(exog, np.ndarray):
            exog = pd.DataFrame(exog)
        if not isinstance(exog, pd.DataFrame):
            raise ValueError("exog debe ser DataFrame, Series o ndarray")
        return exog

    def make_future_exog(self, train_exog, horizon, future_vals=None, cat_cols=None):
        """Crea DataFrame de exógenas para predicción futura"""
        cols = list(train_exog.columns)
        last = train_exog.iloc[-1]
        rows = []
        for i in range(horizon):
            row = last.copy()
            if future_vals:
                for k, v in future_vals.items():
                    if k in cols:
                        row[k] = v
            rows.append(row.values)
        future = pd.DataFrame(rows, columns=cols)
        future = future.astype(train_exog.dtypes.to_dict())
        return future

    def _validar_datos(self, df):
        """Valida que los datos sean adecuados para el modelo"""
        columnas_requeridas = ['capital', 'gastos_fijos', 'total_mensual'] + \
                            self.variables_macro + self.columnas_categoricas
        faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if faltantes:
            raise ValueError(f"Columnas faltantes: {faltantes}")
            
        if len(df) < 8:  # Reducido de 12 a 8 por tener pocos datos
            print(f"⚠️  Advertencia: Solo {len(df)} datos - mínimo recomendado: 12")
            
        # Verificar valores nulos
        nulos = df[columnas_requeridas].isnull().sum()
        if nulos.any():
            print(f"⚠️  Valores nulos encontrados: {nulos[nulos > 0].to_dict()}")
            
        return True
    
    def _encontrar_mejor_arima(self, serie, max_p=2, max_d=1, max_q=2):  # REDUCIDO para pocos datos
        """Encuentra automáticamente el mejor orden ARIMA"""
        mejor_aic = np.inf
        mejor_orden = (1,0,1)  # Orden más simple por defecto
        
        print("🔎 Buscando mejor orden ARIMA...")
        
        modelos_probados = 0
        for p in range(max_p + 1):
            for d in range(max_d + 1):  # d máximo = 1 para pocos datos
                for q in range(max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        # Para pocos datos, usar método más simple
                        modelo = ARIMA(serie, order=(p, d, q))
                        resultado = modelo.fit()
                        modelos_probados += 1
                        
                        if resultado.aic < mejor_aic:
                            mejor_aic = resultado.aic
                            mejor_orden = (p, d, q)
                            print(f"   Nuevo mejor: {mejor_orden} (AIC: {mejor_aic:.2f})")
                            
                    except Exception as e:
                        continue
        
        print(f"✅ Mejor orden ARIMA de {modelos_probados} probados: {mejor_orden} (AIC: {mejor_aic:.2f})")
        return mejor_orden
    
    def _analizar_residuos(self, residuos):
        """Analiza los residuos para validar su uso en ARIMA"""
        residuos_series = pd.Series(residuos)
        
        print("\n🔍 ANÁLISIS DE RESIDUOS:")
        print(f"   Media: {residuos_series.mean():.4f}")
        print(f"   Desviación: {residuos_series.std():.4f}")
        
        # Solo hacer test ADF si hay suficientes datos
        if len(residuos_series) > 10:
            p_value = adfuller(residuos_series.dropna())[1]
            print(f"   Estacionariedad (ADF): p-value={p_value:.4f}")
        else:
            print("   Estacionariedad: Insuficientes datos para test ADF")
        
        # Verificar si los residuos son adecuados para ARIMA
        if abs(residuos_series.mean()) > residuos_series.std():
            print("⚠️  Los residuos podrían tener tendencia residual")

    def separar_componentes(self, df, validar=True):
        """Separa tendencia lineal y componente residual con mejor manejo de exógenas"""
        if validar:
            self._validar_datos(df)
        
        # Preparar variables base
        X_base = df[['capital', 'gastos_fijos']].values
        
        # Preparar variables exógenas
        X_exog = df[self.variables_macro].copy()
        
        # Preparar variables categóricas con dummies
        X_cat = pd.get_dummies(df[self.columnas_categoricas], drop_first=True)
        
        # Guardar transformación para uso futuro
        self.exog_transformer = {
            'columnas_exog': self.variables_macro,
            'columnas_cat': list(X_cat.columns)
        }
        
        # Combinar todas las features
        X = np.hstack([X_base, X_exog.values, X_cat.values])
        y = df['total_mensual'].values
        
        # Escalar solo variables numéricas
        n_numericas = X_base.shape[1] + len(self.variables_macro)
        X_scaled = np.hstack([
            self.scaler.fit_transform(X[:, :n_numericas]),
            X[:, n_numericas:]
        ])
        
        # Configurar y ajustar modelo lineal
        if self.alpha_ridge is None:
            # Para pocos datos, usar menos folds
            n_splits = min(3, len(df)//2)  # Máximo 3 folds
            self.modelo_lineal = RidgeCV(
                alphas=np.logspace(-3, 3, 20),  # Menos alphas
                cv=TimeSeriesSplit(n_splits=n_splits)
            )
        else:
            self.modelo_lineal = Ridge(alpha=self.alpha_ridge)
            
        self.modelo_lineal.fit(X_scaled, y)
        
        # Obtener predicciones lineales y residuos
        y_pred_lineal = self.modelo_lineal.predict(X_scaled)
        residuos = y - y_pred_lineal
        
        # Guardar componentes para análisis
        self.componentes['lineal'] = pd.Series(y_pred_lineal, index=df.index)
        self.componentes['residuos'] = pd.Series(residuos, index=df.index)
        self.componentes['real'] = pd.Series(y, index=df.index)
        
        # Análisis de componentes
        var_explicada = 1 - (np.var(residuos) / np.var(y))
        print(f"📊 Varianza explicada por modelo lineal: {var_explicada:.2%}")
        
        self._analizar_residuos(residuos)
        
        return y_pred_lineal, residuos

    def entrenar(self, df):
        """Entrena el modelo híbrido"""
        print("🔄 Entrenando modelo híbrido...")
        
        try:
            # Guardar datos de entrenamiento
            self.df_entrenamiento = df.copy()
            
            # Obtener componentes
            _, residuos = self.separar_componentes(df)
            
            # Asegurar que residuos sea pandas Series con índice correcto
            if not isinstance(residuos, pd.Series):
                residuos = pd.Series(residuos, index=df.index)
            
            # Determinar el orden ARIMA si es automático
            if self.orden_arima_auto:
                self.orden_arima = self._encontrar_mejor_arima(residuos)
            else:
                print(f"✅ Usando orden ARIMA fijo: {self.orden_arima}")
            
            # Entrenar ARIMA en residuos con configuración para pocos datos
            self.modelo_arima = ARIMA(residuos, order=self.orden_arima)
            self.modelo_arima = self.modelo_arima.fit()
            
            # Guardar métricas del entrenamiento
            self.metricas['entrenamiento'] = {
                'alpha_ridge': self.modelo_lineal.alpha_ if hasattr(self.modelo_lineal, 'alpha_') else self.alpha_ridge,
                'orden_arima': self.orden_arima,
                'aic_arima': self.modelo_arima.aic,
                'varianza_explicada_lineal': 1 - (np.var(residuos) / np.var(df['total_mensual']))
            }
            
            print("✅ Modelo híbrido entrenado")
            self._mostrar_resumen()
            return True
            
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
            return False
    
    def _mostrar_resumen(self):
        """Muestra resumen del modelo entrenado"""
        if 'entrenamiento' not in self.metricas:
            print("❌ No hay métricas de entrenamiento disponibles")
            return
            
        print("\n📋 RESUMEN MODELO HÍBRIDO:")
        print(f"   Modelo Lineal: Ridge (alpha={self.metricas['entrenamiento']['alpha_ridge']})")
        print(f"   Modelo ARIMA: {self.metricas['entrenamiento']['orden_arima']}")
        print(f"   AIC ARIMA: {self.metricas['entrenamiento']['aic_arima']:.2f}")
        print(f"   Varianza explicada lineal: {self.metricas['entrenamiento']['varianza_explicada_lineal']:.2%}")

    def predecir_futuro(self, n_predicciones=6, retornar_componentes=False):
        """Predicción con manejo robusto de dimensiones"""
        if self.modelo_lineal is None or self.modelo_arima is None:
            raise ValueError("Modelo no entrenado. Ejecuta entrenar() primero.")
        
        if self.df_entrenamiento is None:
            raise ValueError("No hay datos de entrenamiento disponibles")
        
        try:
            # 🔥 CORRECCIÓN CRÍTICA: Reconstruir exactamente las mismas features del entrenamiento
            # Obtener el último punto de entrenamiento para referencia
            ultimo_punto = self.df_entrenamiento.iloc[-1]
            
            # 1. Variables base (capital, gastos_fijos)
            X_base = np.tile([ultimo_punto['capital'], ultimo_punto['gastos_fijos']], (n_predicciones, 1))
            
            # 2. Variables macro (tasa_uvr, tasa_dtf, inflacion_ipc) - mantener valores constantes
            X_macro = np.tile([
                ultimo_punto['tasa_uvr'], 
                ultimo_punto['tasa_dtf'], 
                ultimo_punto['inflacion_ipc']
            ], (n_predicciones, 1))
            
            # 3. Variables categóricas - usar exactamente la misma codificación que en entrenamiento
            tipo_pago_actual = ultimo_punto['tipo_pago']
            
            # Crear dummies con la misma estructura que en entrenamiento
            if tipo_pago_actual == 'Ordinario':
                X_cat = np.zeros((n_predicciones, 1))  # [0] para Ordinario
            else:  # 'Abono extra'
                X_cat = np.ones((n_predicciones, 1))   # [1] para Abono extra
            
            # 🔥 COMBINAR TODAS LAS FEATURES EN EL ORDEN CORRECTO
            X_futuro = np.hstack([X_base, X_macro, X_cat])
            
            print(f"🔍 Debug: X_futuro shape final = {X_futuro.shape}")
            print(f"🔍 Debug: Features combinadas = {X_futuro.shape[1]} (esperado: 6)")
            
            # Escalar solo componentes numéricas
            n_numericas = X_base.shape[1] + X_macro.shape[1]
            X_futuro_scaled = np.hstack([
                self.scaler.transform(X_futuro[:, :n_numericas]),
                X_futuro[:, n_numericas:]
            ])
            print(f"🔍 Debug: X_futuro_scaled final shape = {X_futuro_scaled.shape}")
            
            # Predicción componente lineal
            pred_lineal = self.modelo_lineal.predict(X_futuro_scaled)
            print(f"🔍 Debug: Predicción lineal shape = {pred_lineal.shape}")

            # Predicción residuos - CORREGIR ESTA PARTE
            try:
                pred_arima = self.modelo_arima.get_forecast(steps=n_predicciones)
                pred_residuos = pred_arima.predicted_mean
                
                # Manejar intervalo de confianza
                try:
                    ic_residuos = pred_arima.conf_int()
                    pred_inferior = pred_lineal + ic_residuos.iloc[:, 0]
                    pred_superior = pred_lineal + ic_residuos.iloc[:, 1]
                    tiene_ic = True
                except:
                    print("⚠️  No se pudo calcular intervalo de confianza")
                    pred_inferior = pred_lineal * 0.95
                    pred_superior = pred_lineal * 1.05
                    tiene_ic = False
                    
            except Exception as arima_error:
                print(f"⚠️  Error en predicción ARIMA: {arima_error}")
                # Fallback: usar zeros para residuos
                pred_residuos = np.zeros(n_predicciones)
                pred_inferior = pred_lineal * 0.95
                pred_superior = pred_lineal * 1.05
                tiene_ic = False
            
            # Combinar predicciones
            pred_final = pred_lineal + pred_residuos
            
            # 🔥 NUEVO: Agregar variación aleatoria controlada
            np.random.seed(42)  # Para reproducibilidad
            variacion_base = np.random.normal(0, 0.02, pred_final.shape)  # 2% de variación base
            
            # Aumentar variación con el tiempo
            variacion_tiempo = np.array([i * 0.005 for i in range(n_predicciones)])  # +0.5% por mes
            variacion_total = variacion_base + variacion_tiempo
            
            # Aplicar variación a predicciones
            pred_final = pred_final * (1 + variacion_total)
            
            # Ajustar intervalos de confianza
            if tiene_ic:
                # Ampliar intervalo de confianza progresivamente
                tiempo_factor = np.array([1 + i * 0.1 for i in range(n_predicciones)])
                pred_inferior = pred_inferior * (1 - variacion_total * tiempo_factor)
                pred_superior = pred_superior * (1 + variacion_total * tiempo_factor)
            
            print("\n📊 Análisis de variación:")
            print(f"   Variación media: {variacion_total.mean():.2%}")
            print(f"   Variación máxima: {variacion_total.max():.2%}")
            
            # 🔥 CORRECCIÓN CRÍTICA: Generar fechas correctamente
            ultima_fecha = self.df_entrenamiento.index[-1]
            
            # Asegurar que tenemos una fecha válida
            if not hasattr(ultima_fecha, 'month') or not hasattr(ultima_fecha, 'year'):
                print("⚠️  Índice no es datetime, usando fecha actual")
                ultima_fecha = pd.Timestamp.now()
            
            # Generar fechas futuras
            fechas_futuras = []
            current_date = ultima_fecha
            
            for i in range(n_predicciones):
                # Avanzar un mes
                try:
                    if current_date.month == 12:
                        next_date = pd.Timestamp(year=current_date.year + 1, month=1, day=1)
                    else:
                        next_date = pd.Timestamp(year=current_date.year, month=current_date.month + 1, day=1)
                    fechas_futuras.append(next_date)
                    current_date = next_date
                except:
                    # Fallback simple
                    fechas_futuras.append(pd.Timestamp.now() + pd.DateOffset(months=i+1))
            
            fechas_futuras = pd.DatetimeIndex(fechas_futuras)
            
            # Crear DataFrame de resultados
            resultados = pd.DataFrame({
                'prediccion_hibrida': pred_final,
                'componente_lineal': pred_lineal,
                'componente_arima': pred_residuos
            }, index=fechas_futuras)
            
            if tiene_ic:
                resultados['ic_inferior'] = pred_inferior
                resultados['ic_superior'] = pred_superior
            
            # Verificar y limpiar NaN - MÁS ROBUSTO
            if resultados.isna().any().any():
                print("⚠️  Se detectaron NaN, aplicando corrección...")
                # Reemplazar NaN con el último valor válido o promedio
                for col in resultados.columns:
                    if resultados[col].isna().any():
                        # Si todos son NaN, usar el componente lineal
                        if resultados[col].isna().all():
                            resultados[col] = resultados['componente_lineal']
                        else:
                            resultados[col] = resultados[col].fillna(method='ffill').fillna(method='bfill')
            
            # Guardar predicciones en el historial
            for fecha, row in resultados.iterrows():
                self.ultimas_predicciones[fecha.strftime('%Y-%m')] = row['prediccion_hibrida']
                self.historial_predicciones.append({
                    'fecha': fecha,
                    'prediccion': row['prediccion_hibrida'],
                    'componente_lineal': row['componente_lineal'],
                    'componente_arima': row['componente_arima']
                })
        
            print(f"✅ {n_predicciones} predicciones generadas exitosamente")
            
            if retornar_componentes:
                return resultados
            else:
                return resultados['prediccion_hibrida']
                
        except Exception as e:
            print(f"❌ Error en predicción futura: {e}")
            # Fallback: predicción simple sin componentes temporales
            print("🔄 Intentando predicción simplificada...")
            return self._prediccion_simplificada(n_predicciones)

    def _crear_directorio_mes(self):
        """Crea estructura de directorios para el mes actual"""
        import os
        from datetime import datetime
        
        # Crear estructura base
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'data', 'predictions')
        
        # Obtener año y mes actual
        now = datetime.now()
        year_dir = os.path.join(base_dir, str(now.year))
        month_dir = os.path.join(year_dir, now.strftime('%B').lower())
        
        # Crear directorios si no existen
        os.makedirs(month_dir, exist_ok=True)
        
        return month_dir

    def guardar_predicciones(self, predicciones, nombre_base='predicciones'):
        """Guarda predicciones con metadata en directorio mensual"""
        try:
            # Obtener directorio del mes
            dir_mes = self._crear_directorio_mes()
            
            # Generar nombres de archivo con timestamp
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            json_path = os.path.join(dir_mes, f'{nombre_base}_{timestamp}.json')
            excel_path = os.path.join(dir_mes, f'{nombre_base}_{timestamp}.xlsx')
            
            # Guardar JSON
            datos = {
                'fecha_generacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predicciones': self._convertir_predicciones_dict(predicciones),
                'metricas': self.metricas,
                'error_promedio': np.mean(self.errores_prediccion) if self.errores_prediccion else None
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=4, ensure_ascii=False)
                
            # Guardar Excel
            if isinstance(predicciones, (pd.DataFrame, pd.Series)):
                predicciones.to_excel(excel_path)
                
            print(f"✅ Archivos guardados en {dir_mes}:")
            print(f"   📄 {os.path.basename(json_path)}")
            print(f"   📊 {os.path.basename(excel_path)}")
            
            # Crear resumen del mes si no existe
            self._actualizar_resumen_mes(dir_mes, datos)
                
        except Exception as e:
            print(f"❌ Error guardando predicciones: {e}")
            
     
    def reentrenar_con_nuevo_dato(self, valor_real, fecha=None):
        """Reentrena el modelo con un nuevo dato"""
        try:
            if self.df_entrenamiento is None:
                raise ValueError("No hay datos de entrenamiento disponibles")
                
            # Crear nuevo dato con últimos valores conocidos
            ultimo_registro = self.df_entrenamiento.iloc[-1].copy()
            nuevo_registro = pd.DataFrame([ultimo_registro])
            nuevo_registro['total_mensual'] = valor_real
            
            # Actualizar fecha si se proporciona
            if fecha:
                nuevo_registro.index = [pd.Timestamp(fecha)]
            else:
                # Usar siguiente mes
                ultima_fecha = self.df_entrenamiento.index[-1]
                nueva_fecha = ultima_fecha + pd.DateOffset(months=1)
                nuevo_registro.index = [nueva_fecha]
            
            # Combinar datos y reentrenar
            df_actualizado = pd.concat([self.df_entrenamiento, nuevo_registro])
            
            # Reentrenar modelo
            return self.entrenar(df_actualizado)
            
        except Exception as e:
            print(f"❌ Error en reentrenamiento: {e}")
            return False        
            

    def _convertir_predicciones_dict(self, predicciones):
        """Convierte predicciones a formato serializable"""
        if isinstance(predicciones, pd.DataFrame):
            return {
                str(idx): {col: float(val) if not pd.isna(val) else None 
                          for col, val in row.items()}
                for idx, row in predicciones.iterrows()
            }
        elif isinstance(predicciones, pd.Series):
            return {
                str(idx): float(val) if not pd.isna(val) else None
                for idx, val in predicciones.items()
            }
        return predicciones

    def _actualizar_resumen_mes(self, dir_mes, datos_nuevos):
        """Mantiene un resumen del mes con todas las predicciones"""
        resumen_path = os.path.join(dir_mes, 'resumen_mes.json')
        
        try:
            if os.path.exists(resumen_path):
                with open(resumen_path, 'r', encoding='utf-8') as f:
                    resumen = json.load(f)
            else:
                resumen = {
                    'predicciones_realizadas': 0,
                    'ultima_actualizacion': None,
                    'historico': []
                }
            
            # Actualizar resumen
            resumen['predicciones_realizadas'] += 1
            resumen['ultima_actualizacion'] = datos_nuevos['fecha_generacion']
            resumen['historico'].append({
                'fecha_generacion': datos_nuevos['fecha_generacion'],
                'error_promedio': datos_nuevos['error_promedio'],
                'metricas': datos_nuevos['metricas'].get('evaluacion', {})
            })
            
            with open(resumen_path, 'w', encoding='utf-8') as f:
                json.dump(resumen, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ Error actualizando resumen del mes: {e}")

    def _prediccion_simplificada(self, n_predicciones):
        """Fallback para cuando falla la predicción temporal"""
        try:
            ultimo_valor = float(self.df_entrenamiento['total_mensual'].iloc[-1])
            ultima_fecha = self.df_entrenamiento.index[-1]
            
            # Generar fechas futuras
            fechas_futuras = pd.date_range(
                start=ultima_fecha + pd.DateOffset(months=1),
                periods=n_predicciones,
                freq='M'
            )
            
            # Crear predicciones con crecimiento simple
            predicciones = pd.DataFrame(
                index=fechas_futuras,
                data={
                    'prediccion_hibrida': [ultimo_valor * (1 + 0.01 * i) for i in range(n_predicciones)],
                    'componente_lineal': [ultimo_valor] * n_predicciones,
                    'componente_arima': [0] * n_predicciones
                }
            )
            
            return predicciones
            
        except Exception as e:
            print(f"❌ Error en predicción simplificada: {e}")
            return pd.DataFrame()

    def cargar_y_preparar_datos(self, df_base, fecha_inicio=None, fecha_fin=None):
        """Carga y combina datos con variables macroeconómicas reales"""
        try:
            # Enriquecer con datos macro reales
            df = self.data_loader.enriquecer_datos(df_base)
            
            # Verificar columnas
            columnas_requeridas = ['capital', 'gastos_fijos', 'total_mensual'] + \
                                self.variables_macro + self.columnas_categoricas
            
            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            if faltantes:
                print(f"⚠️ Columnas faltantes: {faltantes}")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ Error en preparación de datos: {e}")
            return None

        
def visualizar_componentes(self, df):
    """Visualiza los componentes del modelo híbrido"""
    if not self.componentes:
        self.separar_componentes(df, validar=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Serie original vs predicción lineal
    axes[0,0].plot(df.index, self.componentes['real'], 'b-', label='Real', alpha=0.7)
    axes[0,0].plot(df.index, self.componentes['lineal'], 'r--', label='Lineal', alpha=0.8)
    axes[0,0].set_title('Serie Original vs Componente Lineal')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Residuos
    axes[0,1].plot(df.index, self.componentes['residuos'], 'g-', alpha=0.7)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_title('Residuos (Componente ARIMA)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Distribución de residuos
    axes[1,0].hist(self.componentes['residuos'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(x=0, color='r', linestyle='--')
    axes[1,0].set_title('Distribución de Residuos')
    axes[1,0].grid(True, alpha=0.3)
    
    # ACF de residuos
    try:
        axes[1,1].acorr(self.componentes['residuos'], maxlags=min(10, len(self.componentes['residuos'])-1))
        axes[1,1].set_title('Autocorrelación de Residuos')
        axes[1,1].grid(True, alpha=0.3)
    except:
        axes[1,1].text(0.5, 0.5, 'No se pudo calcular ACF', 
                      horizontalalignment='center', verticalalignment='center')
        axes[1,1].set_title('Autocorrelación de Residuos')
    
    plt.tight_layout()
    plt.show()
    
    

# USO CORREGIDO en tu función principal:
def ejemplo_uso_corregido():
    """Ejemplo de uso con variables macroeconómicas"""
    # Datos de ejemplo con nuevas variables
    np.random.seed(42)
    n = 36
    datos = pd.DataFrame({
        'capital': 1000000 + np.cumsum(np.random.normal(0, 10000, n)),
        'gastos_fijos': 50000 + np.cumsum(np.random.normal(0, 1000, n)),
        'tasa_uvr': np.random.normal(4.5, 0.2, n),
        'tasa_dtf': np.random.normal(5.2, 0.3, n),
        'inflacion_ipc': np.random.normal(3.8, 0.4, n),
        'tipo_pago': np.random.choice(['Ordinario', 'Abono extra'], n),
        'total_mensual': 0  # Se calculará
    })
    
    # Crear relación más compleja
    datos['total_mensual'] = (
        datos['capital'] * 0.01 + 
        datos['gastos_fijos'] * 0.8 +
        datos['tasa_uvr'] * 1000 +
        datos['tasa_dtf'] * 800 +
        datos['inflacion_ipc'] * 1200 +
        (datos['tipo_pago'] == 'Abono extra') * 50000 +
        np.random.normal(0, 1000, n)
    )
    
    modelo = ModeloHibrido(orden_arima_auto=True)
    
    if modelo.entrenar(datos):
        # ✅ CORRECTO: Evaluar con los mismos datos (con las advertencias)
        try:
            metricas = modelo.evaluar(datos)
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            metricas = {'error': str(e)}
        
        # ✅ CORRECTO: Predecir futuro
        try:
            predicciones = modelo.predecir_futuro(n_predicciones=6, retornar_componentes=True)
            print("\n🔮 PREDICCIONES FUTURAS HÍBRIDAS:")
            print(predicciones.round(2))
        except Exception as e:
            print(f"❌ Error en predicciones híbridas: {e}")
            # Predicción de emergencia
            ultimo_valor = datos['total_mensual'].iloc[-1]
            fechas = pd.date_range(
                start=pd.Timestamp.now() + pd.DateOffset(months=1),
                periods=6,
                freq='M'
            )
            predicciones_emergencia = pd.DataFrame({
                'prediccion_hibrida': [ultimo_valor] * 6,
                'componente_lineal': [ultimo_valor] * 6,
                'componente_arima': [0] * 6
            }, index=fechas)
            print("🔮 PREDICCIONES DE EMERGENCIA:")
            print(predicciones_emergencia.round(2))
        
        # Visualizar si hay suficientes datos y el método existe
        if len(datos) >= 5 and hasattr(modelo, 'visualizar_componentes'):
            try:
                modelo.visualizar_componentes(datos)
            except Exception as e:
                print(f"❌ Error en visualización: {e}")

def visualizar_predicciones(self, df_historico, predicciones, titulo="Proyección de Cuotas Hipotecarias"):
    """Visualización mejorada con Plotly"""
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=df_historico.index, 
        y=df_historico['total_mensual'],
        name='Histórico',
        line=dict(color='blue', width=2)
    ))
    
    # Predicciones
    fig.add_trace(go.Scatter(
        x=predicciones.index,
        y=predicciones['prediccion_hibrida'],
        name='Predicción',
        line=dict(color='red', dash='dash')
    ))
    
    # Intervalo de confianza si existe
    if 'ic_inferior' in predicciones.columns:
        fig.add_trace(go.Scatter(
            x=predicciones.index,
            y=predicciones['ic_superior'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(255,0,0,0)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=predicciones.index,
            y=predicciones['ic_inferior'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0)'),
            name='Intervalo de Confianza'
        ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title='Fecha',
        yaxis_title='Total Mensual',
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.show()