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
warnings.filterwarnings('ignore')

class ModeloHibrido:
    def __init__(self, alpha_ridge=None, orden_arima_auto=True):
        self.modelo_lineal = None
        self.modelo_arima = None
        self.scaler = StandardScaler()
        self.orden_arima = (1,1,1)
        self.alpha_ridge = alpha_ridge
        self.orden_arima_auto = orden_arima_auto
        self.metricas = {}
        self.componentes = {}
        self.df_entrenamiento = None  # Agregar esta línea
        
    def _validar_datos(self, df):
        """Valida que los datos sean adecuados para el modelo"""
        columnas_requeridas = ['capital', 'gastos_fijos', 'total_mensual']
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
        """Separa tendencia lineal y componente residual"""
        if validar:
            self._validar_datos(df)
        
        # Preparar features para modelo lineal
        X = df[['capital', 'gastos_fijos']].values
        y = df['total_mensual'].values
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
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
        """Predice valores FUTUROS (no para evaluación)"""
        if self.modelo_lineal is None or self.modelo_arima is None:
            raise ValueError("Modelo no entrenado. Ejecuta entrenar() primero.")
        
        if self.df_entrenamiento is None:
            raise ValueError("No hay datos de entrenamiento disponibles")
        
        try:
            # Usar el ÚLTIMO punto conocido para proyectar
            ultimo_punto = self.df_entrenamiento[['capital', 'gastos_fijos']].iloc[-1:].values
            
            # Crear escenarios futuros (asumiendo pequeña variación)
            X_futuro = np.tile(ultimo_punto, (n_predicciones, 1))
            
            # Aplicar variación mínima (1%)
            variacion = np.random.normal(0, 0.01, X_futuro.shape)
            X_futuro = X_futuro * (1 + variacion)
            
            X_futuro_scaled = self.scaler.transform(X_futuro)
            
            # Predicción componente lineal
            pred_lineal = self.modelo_lineal.predict(X_futuro_scaled)
            
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

    def _prediccion_simplificada(self, n_predicciones):
        """Fallback para cuando falla la predicción temporal"""
        ultimo_valor = self.df_entrenamiento['total_mensual'].iloc[-1]
        predicciones = [ultimo_valor * (1 + 0.01 * i) for i in range(n_predicciones)]  # 1% de crecimiento mensual
        
        fechas = pd.RangeIndex(start=0, stop=n_predicciones, name='periodo_futuro')
        return pd.Series(predicciones, index=fechas, name='prediccion_simplificada')

    def evaluar(self, df_test=None, retornar_detalles=False):
        """Evalúa el modelo usando backtesting simple"""
        if self.modelo_lineal is None or self.modelo_arima is None:
            raise ValueError("Modelo no entrenado")
        
        try:
            # Si no se proporciona test, usar los mismos datos (no ideal)
            if df_test is None:
                print("⚠️  Usando datos de entrenamiento para evaluación (puede sobrestimar rendimiento)")
                df_test = self.df_entrenamiento
            
            if len(df_test) == 0:
                raise ValueError("DataFrame de prueba está vacío")
                
            # Para evaluación, necesitamos separar train/test temporal
            if len(df_test) >= 4:  # Mínimo para hacer split
                split_point = max(1, len(df_test) - 2)  # Últimos 2 puntos para test
                df_train = df_test.iloc[:split_point]
                df_eval = df_test.iloc[split_point:]
                
                # Reentrenar modelo con datos de train
                modelo_temp = ModeloHibrido(orden_arima_auto=False)
                modelo_temp.entrenar(df_train)
                
                # Predecir los puntos de test
                predicciones = modelo_temp.predecir_futuro(n_predicciones=len(df_eval), retornar_componentes=True)
                y_true = df_eval['total_mensual'].values
                y_pred = predicciones['prediccion_hibrida'].values[:len(y_true)]
                
            else:
                # Con muy pocos datos, evaluación simple
                print("⚠️  Muy pocos datos para evaluación temporal, usando ajuste directo")
                y_true = df_test['total_mensual'].values
                
                # Usar el modelo actual para predecir (esto sobrestima)
                X_test = df_test[['capital', 'gastos_fijos']].values
                X_test_scaled = self.scaler.transform(X_test)
                pred_lineal = self.modelo_lineal.predict(X_test_scaled)
                
                # Para residuos, usar predicciones in-sample
                pred_residuos = self.modelo_arima.predict(start=0, end=len(df_test)-1)
                y_pred = pred_lineal + pred_residuos
            
            # Calcular métricas
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # MAPE con protección
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            
            self.metricas['evaluacion'] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            
            print("\n📊 EVALUACIÓN MODELO HÍBRIDO:")
            print(f"   MAE:  {mae:,.2f}")
            print(f"   RMSE: {rmse:,.2f}")
            print(f"   R²:   {r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")
            
            if retornar_detalles:
                return self.metricas['evaluacion'], predicciones
            else:
                return self.metricas['evaluacion']
                
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            # Evaluación simplificada como fallback
            print("📊 Evaluación simplificada:")
            residuos = self.componentes.get('residuos', pd.Series([0]))
            print(f"   Desviación de residuos: {residuos.std():.2f}")
            return {'error': str(e)}
        
        
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
    """Ejemplo de uso mejorado con manejo robusto de errores"""
    # TUS DATOS REALES (9 registros)
    # ... tu código de carga de datos ...
    
    modelo = ModeloHibrido(orden_arima_auto=True)
    
    if modelo.entrenar(tus_datos_reales):
        # ✅ CORRECTO: Evaluar con los mismos datos (con las advertencias)
        try:
            metricas = modelo.evaluar(tus_datos_reales)
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
            ultimo_valor = tus_datos_reales['total_mensual'].iloc[-1]
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
        if len(tus_datos_reales) >= 5 and hasattr(modelo, 'visualizar_componentes'):
            try:
                modelo.visualizar_componentes(tus_datos_reales)
            except Exception as e:
                print(f"❌ Error en visualización: {e}")