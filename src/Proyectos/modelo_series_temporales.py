import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

class ModeloSeriesTiempo:
    def __init__(self):
        self.modelo_arima = None
        self.modelo_prophet = None
        self.mejor_orden_arima = None
        self.metricas = {}
        self.usar_prophet = False  # Nueva bandera para control de Prophet
    
    def preparar_datos(self, df, fecha_inicio='2025-01-01', frecuencia='M'):
        """Prepara y valida datos para series temporales"""
        df = df.copy()
        
        # Crear índice temporal
        if 'fecha' not in df.columns:
            df['fecha'] = pd.date_range(start=fecha_inicio, periods=len(df), freq=frecuencia)
        
        df['fecha'] = pd.to_datetime(df['fecha'])
        df.set_index('fecha', inplace=True)
        
        # Calcular diferencias para estacionarizar
        df['total_mensual_diff'] = df['total_mensual'].diff()
        
        # Decidir si usar Prophet basado en datos disponibles
        self.usar_prophet = len(df) >= 12 and frecuencia in ['D', 'W']
        if not self.usar_prophet:
            print("ℹ️ Prophet deshabilitado - insuficientes datos o frecuencia inadecuada")
        
        # Verificar estacionalidad y estacionariedad
        self._analizar_serie(df['total_mensual_diff'].dropna())
        
        return df
    
    def _analizar_serie(self, serie):
        """Análisis exploratorio de la serie temporal"""
        print("\n🔍 ANÁLISIS DE LA SERIE TEMPORAL:")
        print(f"   Períodos: {len(serie)}")
        print(f"   Rango fechas: {serie.index[0]} a {serie.index[-1]}")
        print(f"   Media: {serie.mean():.2f}")
        print(f"   Desviación: {serie.std():.2f}")
        
        # Test de estacionariedad
        resultado_adf = adfuller(serie.dropna())
        print(f"   Test ADF (estacionariedad): p-value={resultado_adf[1]:.4f}")
        if resultado_adf[1] > 0.05:
            print("   ⚠️  Serie no estacionaria - puede necesitar diferenciación")
        else:
            print("   ✅ Serie estacionaria")
    
    def _encontrar_mejor_arima(self, serie, max_p=3, max_d=2, max_q=3, exog=None):
        """Encuentra los mejores parámetros ARIMA usando AIC"""
        mejor_aic = np.inf
        mejor_orden = (1,1,1)
        
        print("🔎 Buscando mejor modelo ARIMA...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        modelo = ARIMA(serie, order=(p, d, q), exog=exog)
                        resultado = modelo.fit()
                        if resultado.aic < mejor_aic:
                            mejor_aic = resultado.aic
                            mejor_orden = (p, d, q)
                    except:
                        continue
        
        print(f"✅ Mejor orden ARIMA: {mejor_orden} (AIC: {mejor_aic:.2f})")
        return mejor_orden
    
    def entrenar_arima(self, df, orden_manual=None):
        """Entrena modelo ARIMA con variables exógenas"""
        try:
            # Preparar variables exógenas
            exog = df[['capital', 'gastos_fijos']].values if 'capital' in df.columns else None
            
            # Usar serie diferenciada
            serie = df['total_mensual_diff'].dropna()
            
            if orden_manual is None:
                self.mejor_orden_arima = self._encontrar_mejor_arima(serie, exog=exog)
            else:
                self.mejor_orden_arima = orden_manual
            
            # Entrenar SARIMAX con exógenas
            self.modelo_arima = SARIMAX(
                serie,
                order=self.mejor_orden_arima,
                exog=exog[1:] if exog is not None else None
            )
            self.modelo_arima = self.modelo_arima.fit()
            
            print("✅ Modelo ARIMA entrenado con variables exógenas")
            return True
            
        except Exception as e:
            print(f"❌ Error entrenando ARIMA: {e}")
            return False
    
    def entrenar_prophet(self, df, componentes_estacionalidad=True):
        """Entrena modelo Prophet con configuración flexible"""
        try:
            # Preparar datos para Prophet
            df_prophet = pd.DataFrame({
                'ds': df.index,
                'y': df['total_mensual']
            })
            
            # Configurar modelo
            self.modelo_prophet = Prophet(
                yearly_seasonality=componentes_estacionalidad,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Agregar estacionalidad mensual si hay suficientes datos
            if len(df) > 12 and componentes_estacionalidad:
                self.modelo_prophet.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=3
                )
            
            self.modelo_prophet.fit(df_prophet)
            
            # Validación interna (backtesting simple)
            self._validar_prophet(df_prophet)
            
            print("✅ Modelo Prophet entrenado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error entrenando Prophet: {e}")
            return False
    
    def _validar_prophet(self, df_prophet, split_ratio=0.8):
        """Validación simple del modelo Prophet"""
        split_point = int(len(df_prophet) * split_ratio)
        train = df_prophet[:split_point]
        test = df_prophet[split_point:]
        
        if len(test) > 0:
            modelo_val = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            modelo_val.fit(train)
            
            future = modelo_val.make_future_dataframe(periods=len(test), freq='M')
            forecast_val = modelo_val.predict(future)
            
            pred_test = forecast_val['yhat'].iloc[split_point:]
            mae = mean_absolute_error(test['y'], pred_test)
            self.metricas['Prophet'] = {'MAE_validacion': mae}
            print(f"   📈 MAE validación Prophet: {mae:.2f}")
    
    def evaluar_modelos(self, df, n_backtest=6):
        """Evalúa ambos modelos usando backtesting"""
        if len(df) <= n_backtest:
            print("⚠️  Insuficientes datos para backtesting")
            return
        
        print(f"\n🧪 EVALUACIÓN CON BACKTESTING ({n_backtest} períodos):")
        
        # Datos de entrenamiento y prueba
        train = df.iloc[:-n_backtest]
        test = df.iloc[-n_backtest:]
        
        metricas_comparativas = {}
        
        # Evaluar ARIMA
        if self._entrenar_y_evaluar_arima(train, test, metricas_comparativas):
            mae_arima = metricas_comparativas['ARIMA']['MAE']
            print(f"   ARIMA - MAE: {mae_arima:.2f}")
        
        # Evaluar Prophet
        if self._entrenar_y_evaluar_prophet(train, test, metricas_comparativas):
            mae_prophet = metricas_comparativas['Prophet']['MAE']
            print(f"   Prophet - MAE: {mae_prophet:.2f}")
        
        return metricas_comparativas
    
    def _entrenar_y_evaluar_arima(self, train, test, metricas):
        """Entrena y evalúa ARIMA en conjunto de prueba"""
        try:
            modelo_temp = ARIMA(train['total_mensual'], order=self.mejor_orden_arima or (1,1,1))
            modelo_temp = modelo_temp.fit()
            
            predicciones = modelo_temp.forecast(len(test))
            mae = mean_absolute_error(test['total_mensual'], predicciones)
            
            metricas['ARIMA'] = {
                'MAE': mae,
                'RMSE': np.sqrt(mean_squared_error(test['total_mensual'], predicciones))
            }
            return True
        except:
            return False
    
    def _entrenar_y_evaluar_prophet(self, train, test, metricas):
        """Entrena y evalúa Prophet en conjunto de prueba"""
        try:
            df_train = pd.DataFrame({'ds': train.index, 'y': train['total_mensual']})
            
            modelo_temp = Prophet(yearly_seasonality=True)
            modelo_temp.fit(df_train)
            
            future = modelo_temp.make_future_dataframe(periods=len(test), freq='M')
            forecast = modelo_temp.predict(future)
            
            pred_test = forecast['yhat'].iloc[-len(test):].values
            mae = mean_absolute_error(test['total_mensual'], pred_test)
            
            metricas['Prophet'] = {
                'MAE': mae,
                'RMSE': np.sqrt(mean_squared_error(test['total_mensual'], pred_test))
            }
            return True
        except:
            return False
    
    def predecir_proximas_cuotas(self, df, n_predicciones=6, intervalo_confianza=True):
        """Realiza predicciones considerando la diferenciación"""
        ultima_fecha = df.index[-1]
        fechas_futuras = pd.date_range(
            start=ultima_fecha + timedelta(days=1),
            periods=n_predicciones,
            freq='M'
        )
        
        resultados = pd.DataFrame(index=fechas_futuras)
        
        # Predicciones ARIMA
        if self.modelo_arima is not None:
            pred_diff = self.modelo_arima.get_forecast(
                n_predicciones,
                exog=self._preparar_exog_futuras(df, n_predicciones)
            )
            
            # Revertir diferenciación
            ultimo_valor = df['total_mensual'].iloc[-1]
            pred_acumulada = [ultimo_valor]
            for diff_valor in pred_diff.predicted_mean:
                pred_acumulada.append(pred_acumulada[-1] + diff_valor)
            
            resultados['ARIMA'] = pred_acumulada[1:]
            resultados['ARIMA_Pesos'] = resultados['ARIMA'].apply(lambda x: f"${x:,.2f}")
            
            if intervalo_confianza:
                intervalo = pred_diff.conf_int()
                resultados['ARIMA_inferior'] = intervalo.iloc[:, 0] + ultimo_valor
                resultados['ARIMA_superior'] = intervalo.iloc[:, 1] + ultimo_valor
                resultados['Rango_Pesos'] = resultados.apply(
                    lambda x: f"${x['ARIMA_inferior']:,.2f} - ${x['ARIMA_superior']:,.2f}", 
                    axis=1
                )
        
        # Predicciones Prophet (si está habilitado)
        if self.modelo_prophet is not None and self.usar_prophet:
            future_dates = pd.DataFrame({'ds': fechas_futuras})
            pred_prophet = self.modelo_prophet.predict(future_dates)
            resultados['Prophet'] = pred_prophet['yhat'].values
            resultados['Prophet_Pesos'] = resultados['Prophet'].apply(lambda x: f"${x:,.2f}")
            
            if intervalo_confianza:
                resultados['Prophet_inferior'] = pred_prophet['yhat_lower'].values
                resultados['Prophet_superior'] = pred_prophet['yhat_upper'].values
        
        # Reordenar columnas para mejor visualización
        columnas_ordenadas = [
            'ARIMA', 'ARIMA_Pesos', 'ARIMA_inferior', 'ARIMA_superior', 'Rango_Pesos',
            'Prophet', 'Prophet_Pesos'
        ]
        columnas_existentes = [col for col in columnas_ordenadas if col in resultados.columns]
        resultados = resultados[columnas_existentes]
        
        return resultados
    
    def _preparar_exog_futuras(self, df, n_predicciones):
        """Prepara variables exógenas para predicción"""
        if 'capital' not in df.columns:
            return None
            
        # Proyectar últimos valores conocidos
        ultima_fila = df[['capital', 'gastos_fijos']].iloc[-1]
        return np.tile(ultima_fila.values, (n_predicciones, 1))
    
    def visualizar_predicciones(self, df_historico, predicciones, titulo="Proyección de Cuotas Hipotecarias"):
        """Visualización mejorada con intervalos de confianza"""
        plt.figure(figsize=(14, 8))
        
        # Datos históricos
        plt.plot(df_historico.index, df_historico['total_mensual'],
                'bo-', label='Histórico', linewidth=2, markersize=4)
        
        # Predicciones ARIMA
        if 'ARIMA' in predicciones.columns:
            plt.plot(predicciones.index, predicciones['ARIMA'],
                    'r.--', label='ARIMA', linewidth=2, markersize=6)
            
            # Intervalo de confianza ARIMA
            if 'ARIMA_inferior' in predicciones.columns:
                plt.fill_between(predicciones.index,
                               predicciones['ARIMA_inferior'],
                               predicciones['ARIMA_superior'],
                               color='red', alpha=0.2, label='IC ARIMA')
        
        # Predicciones Prophet
        if 'Prophet' in predicciones.columns:
            plt.plot(predicciones.index, predicciones['Prophet'],
                    'g.--', label='Prophet', linewidth=2, markersize=6)
            
            # Intervalo de confianza Prophet
            if 'Prophet_inferior' in predicciones.columns:
                plt.fill_between(predicciones.index,
                               predicciones['Prophet_inferior'],
                               predicciones['Prophet_superior'],
                               color='green', alpha=0.2, label='IC Prophet')
        
        # Promedio si existe
        if 'Promedio' in predicciones.columns:
            plt.plot(predicciones.index, predicciones['Promedio'],
                    'k*-', label='Promedio', linewidth=2, markersize=8)
        
        plt.title(titulo, fontsize=14, fontweight='bold')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Total Mensual', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generar_reporte(self, df, predicciones):
        """Genera reporte ejecutivo de las predicciones"""
        print("\n" + "="*60)
        print("📊 REPORTE EJECUTIVO DE PROYECCIONES")
        print("="*60)
        
        print(f"\n📈 RESUMEN HISTÓRICO:")
        print(f"   Período analizado: {len(df)} meses")
        print(f"   Promedio histórico: {df['total_mensual'].mean():.2f}")
        print(f"   Tendencia: {'↗️ Alza' if df['total_mensual'].iloc[-1] > df['total_mensual'].iloc[0] else '↘️ Baja'}")
        
        print(f"\n🔮 PREDICCIONES ({len(predicciones)} meses):")
        for modelo in ['ARIMA', 'Prophet', 'Promedio']:
            if modelo in predicciones.columns:
                promedio_pred = predicciones[modelo].mean()
                cambio_porcentual = ((promedio_pred - df['total_mensual'].mean()) / df['total_mensual'].mean()) * 100
                print(f"   {modelo:<8}: {promedio_pred:>8.2f} ({cambio_porcentual:+.1f}%)")
        
        print(f"\n💡 RECOMENDACIONES:")
        if 'Promedio' in predicciones.columns:
            tendencia = "estable" if abs(predicciones['Promedio'].pct_change().mean()) < 0.01 else "cambiante"
            print(f"   • Tendencia proyectada: {tendencia}")
            print(f"   • Prepare flujo de caja para promedios de {predicciones['Promedio'].mean():.2f}")
        
        print("="*60)

# Ejemplo de uso mejorado
def main():
    # Simular datos de ejemplo (reemplazar con tus datos reales)
    np.random.seed(42)
    n_meses = 36
    datos_ejemplo = pd.DataFrame({
        'total_mensual': 1000 + np.cumsum(np.random.normal(10, 5, n_meses)) + 
                        50 * np.sin(np.arange(n_meses) * 2 * np.pi / 12)
    })
    
    modelo = ModeloSeriesTiempo()
    
    # Preparar datos
    df_preparado = modelo.preparar_datos(datos_ejemplo)
    
    # Entrenar modelos
    modelo.entrenar_arima(df_preparado)
    modelo.entrenar_prophet(df_preparado)
    
    # Evaluar modelos
    modelo.evaluar_modelos(df_preparado)
    
    # Predecir
    predicciones = modelo.predecir_proximas_cuotas(df_preparado, n_predicciones=12)
    
    # Visualizar y generar reporte
    modelo.visualizar_predicciones(df_preparado, predicciones)
    modelo.generar_reporte(df_preparado, predicciones)

if __name__ == "__main__":
    main()