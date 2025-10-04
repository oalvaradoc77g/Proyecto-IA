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
import joblib  # Agregar al inicio del archivo
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
        
    def _validar_datos(self, df):
        """Valida que los datos sean adecuados para el modelo"""
        columnas_requeridas = ['capital', 'gastos_fijos', 'total_mensual']
        faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if faltantes:
            raise ValueError(f"Columnas faltantes: {faltantes}")
            
        if len(df) < 12:
            print("⚠️  Advertencia: Pocos datos para modelo híbrido")
            
        # Verificar valores nulos
        nulos = df[columnas_requeridas].isnull().sum()
        if nulos.any():
            print(f"⚠️  Valores nulos encontrados: {nulos[nulos > 0].to_dict()}")
            
        return True
    
    def _encontrar_mejor_arima(self, serie, max_p=3, max_d=2, max_q=3):
        """Encuentra automáticamente el mejor orden ARIMA"""
        mejor_aic = np.inf
        mejor_orden = (1,1,1)
        
        print("🔎 Buscando mejor orden ARIMA...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        modelo = ARIMA(serie, order=(p, d, q))
                        resultado = modelo.fit()
                        if resultado.aic < mejor_aic:
                            mejor_aic = resultado.aic
                            mejor_orden = (p, d, q)
                    except:
                        continue
        
        print(f"✅ Mejor orden ARIMA encontrado: {mejor_orden} (AIC: {mejor_aic:.2f})")
        return mejor_orden
    
    def _analizar_residuos(self, residuos):
        """Analiza los residuos para validar su uso en ARIMA"""
        # Convertir residuos a pandas Series
        residuos_series = pd.Series(residuos)
        
        print("\n🔍 ANÁLISIS DE RESIDUOS:")
        print(f"   Media: {residuos_series.mean():.4f}")
        print(f"   Desviación: {residuos_series.std():.4f}")
        print(f"   Estacionariedad (ADF): p-value={adfuller(residuos_series.dropna())[1]:.4f}")
        
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
            self.modelo_lineal = RidgeCV(
                alphas=np.logspace(-3, 3, 50),
                cv=TimeSeriesSplit(n_splits=min(5, len(df)//3))
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
            # Obtener componentes
            _, residuos = self.separar_componentes(df)
            
            # Asegurar que residuos sea pandas Series con índice correcto
            if not isinstance(residuos, pd.Series):
                residuos = pd.Series(residuos, index=df.index)
            else:
                # Verificar que el índice coincida
                if not residuos.index.equals(df.index):
                    residuos = pd.Series(residuos.values, index=df.index)
            
            # Determinar el orden ARIMA si es automático
            if self.orden_arima_auto:
                self.orden_arima = self._encontrar_mejor_arima(residuos)
            
            # Entrenar ARIMA en residuos
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

    def predecir(self, df_futuro, n_predicciones=6, retornar_componentes=False):
        """Realiza predicciones combinadas con manejo robusto"""
        if self.modelo_lineal is None or self.modelo_arima is None:
            raise ValueError("Modelo no entrenado. Ejecuta entrenar() primero.")
        
        try:
            # Validar datos futuros
            if 'capital' not in df_futuro.columns or 'gastos_fijos' not in df_futuro.columns:
                raise ValueError("Datos futuros deben contener 'capital' y 'gastos_fijos'")
            
            # Preparar datos para predicción
            X = df_futuro[['capital', 'gastos_fijos']].values
            X_scaled = self.scaler.transform(X)
            
            # Predicción componente lineal
            pred_lineal = self.modelo_lineal.predict(X_scaled)
            
            # Si estamos prediciendo datos históricos, usar la longitud real
            n_pred = len(df_futuro) if len(df_futuro) > 0 else n_predicciones
            
            # Predicción residuos
            pred_arima = self.modelo_arima.get_forecast(n_pred)
            pred_residuos = pred_arima.predicted_mean
            ic_residuos = pred_arima.conf_int()
            
            # Combinar predicciones
            pred_final = pred_lineal + pred_residuos
            pred_inferior = pred_lineal + ic_residuos.iloc[:, 0]
            pred_superior = pred_lineal + ic_residuos.iloc[:, 1]
            
            # Crear índice temporal adecuado
            if len(df_futuro) > 0:
                # Usar el índice existente para datos históricos
                fechas = df_futuro.index
            else:
                # Crear nuevo índice para predicciones futuras
                ultima_fecha = pd.Timestamp.now()
                fechas = pd.date_range(
                    start=ultima_fecha,
                    periods=n_predicciones,
                    freq='M'
                )
            
            # Crear DataFrame de resultados
            resultados = pd.DataFrame({
                'prediccion_hibrida': pred_final,
                'ic_inferior': pred_inferior,
                'ic_superior': pred_superior,
                'componente_lineal': pred_lineal,
                'componente_arima': pred_residuos
            }, index=fechas)
            
            # Verificar que no hay NaN
            if resultados['prediccion_hibrida'].isna().any():
                print("⚠️ Advertencia: Algunas predicciones contienen NaN")
            
            if retornar_componentes:
                return resultados
            else:
                return resultados['prediccion_hibrida']
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            raise
    
    def evaluar(self, df_test, retornar_detalles=False):
        """Evalúa el modelo híbrido con métricas completas"""
        if self.modelo_lineal is None or self.modelo_arima is None:
            raise ValueError("Modelo no entrenado")
        
        try:
            # Verificar que tenemos suficientes datos para evaluación
            if len(df_test) == 0:
                raise ValueError("DataFrame de prueba está vacío")
                
            y_true = df_test['total_mensual'].values
            
            # Obtener predicciones para el período de prueba
            predicciones = self.predecir(df_test, n_predicciones=len(df_test), retornar_componentes=True)
            y_pred = predicciones['prediccion_hibrida'].values
            
            # Verificar que las predicciones tengan la misma longitud
            if len(y_pred) != len(y_true):
                min_len = min(len(y_pred), len(y_true))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                print(f"⚠️  Ajustando longitud: {min_len} puntos para evaluación")
            
            # Calcular métricas completas
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # MAPE con protección contra división por cero
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
            raise
    
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
        
        # ACF de residuos (opcional)
        axes[1,1].acorr(self.componentes['residuos'], maxlags=10)
        axes[1,1].set_title('Autocorrelación de Residuos')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def guardar_modelo(self, ruta):
        """Guarda el modelo entrenado"""
        if self.modelo_lineal is None or self.modelo_arima is None:
            raise ValueError("No hay modelo entrenado para guardar")
            
        modelo_data = {
            'modelo_lineal': self.modelo_lineal,
            'modelo_arima': self.modelo_arima,
            'scaler': self.scaler,
            'orden_arima': self.orden_arima,
            'metricas': self.metricas,
            'componentes': self.componentes
        }
        
        try:
            joblib.dump(modelo_data, ruta)
            print(f"✅ Modelo guardado en: {ruta}")
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            raise

    def cargar_modelo(self, ruta):
        """Carga un modelo previamente guardado"""
        try:
            modelo_data = joblib.load(ruta)
            self.modelo_lineal = modelo_data['modelo_lineal']
            self.modelo_arima = modelo_data['modelo_arima']
            self.scaler = modelo_data['scaler']
            self.orden_arima = modelo_data['orden_arima']
            self.metricas = modelo_data['metricas']
            self.componentes = modelo_data['componentes']
            print(f"✅ Modelo cargado desde: {ruta}")
            self._mostrar_resumen()
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

# Ejemplo de uso mejorado
def ejemplo_uso():
    # Datos de ejemplo
    np.random.seed(42)
    n = 36
    datos = pd.DataFrame({
        'capital': 1000000 + np.cumsum(np.random.normal(0, 10000, n)),
        'gastos_fijos': 50000 + np.cumsum(np.random.normal(0, 1000, n)),
        'total_mensual': 0  # Se calculará
    })
    
    # Crear relación + ruido
    datos['total_mensual'] = (datos['capital'] * 0.01 + 
                             datos['gastos_fijos'] * 0.8 + 
                             np.random.normal(0, 1000, n))
    
    datos.index = pd.date_range(start='2022-01-01', periods=n, freq='M')
    
    # Crear y entrenar modelo
    modelo = ModeloHibrido(orden_arima_auto=True)
    modelo.entrenar(datos)
    
    # Visualizar componentes
    modelo.visualizar_componentes(datos)
    
    # Evaluar (usando mismos datos como ejemplo)
    metricas = modelo.evaluar(datos)
    
    # Predecir
    predicciones = modelo.predecir(datos, n_predicciones=6, retornar_componentes=True)
    print("\n🔮 PREDICCIONES:")
    print(predicciones.round(2))

if __name__ == "__main__":
    ejemplo_uso()