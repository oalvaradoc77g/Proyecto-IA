import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

class ModeloSeriesTiempo:
    def __init__(self):
        self.modelo_arima = None
        self.modelo_prophet = None
        self.mejor_orden_arima = None
        self.metricas = {}
        self.usar_prophet = False
    
    def preparar_datos(self, df, fecha_inicio='2025-01-01', frecuencia='M'):
        """Prepara y valida datos para series temporales"""
        df = df.copy()
        
        if 'fecha' not in df.columns:
            df['fecha'] = pd.date_range(start=fecha_inicio, periods=len(df), freq=frecuencia)
        
        df['fecha'] = pd.to_datetime(df['fecha'])
        df.set_index('fecha', inplace=True)
        df['total_mensual_diff'] = df['total_mensual'].diff()
        
        self.usar_prophet = len(df) >= 12 and frecuencia in ['D', 'W']
        
        self._analizar_serie(df['total_mensual_diff'].dropna())
        
        return df
    
    def _analizar_serie(self, serie):
        """Análisis exploratorio"""
        print("\n🔍 ANÁLISIS DE LA SERIE TEMPORAL:")
        print(f"   Períodos: {len(serie)}")
        print(f"   Rango: {serie.index[0]} a {serie.index[-1]}")
        print(f"   Media: {serie.mean():.2f}")
        print(f"   Desviación: {serie.std():.2f}")
        
        resultado_adf = adfuller(serie.dropna())
        print(f"   Test ADF: p-value={resultado_adf[1]:.4f}")
        print("   ✅ Serie estacionaria" if resultado_adf[1] <= 0.05 else "   ⚠️  Serie no estacionaria")
    
    def _encontrar_mejor_arima(self, serie, max_p=3, max_d=2, max_q=3, exog=None):
        """Encuentra mejores parámetros ARIMA"""
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
        """Entrena modelo ARIMA"""
        try:
            exog = df[['capital', 'gastos_fijos']].values if 'capital' in df.columns else None
            serie = df['total_mensual_diff'].dropna()
            
            if orden_manual is None:
                self.mejor_orden_arima = self._encontrar_mejor_arima(serie, exog=exog)
            else:
                self.mejor_orden_arima = orden_manual
            
            self.modelo_arima = ARIMA(
                serie,
                order=self.mejor_orden_arima,
                exog=exog[1:] if exog is not None else None
            ).fit()
            
            print("✅ Modelo ARIMA entrenado")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def entrenar_prophet(self, df, componentes_estacionalidad=True):
        """Entrena modelo Prophet"""
        try:
            df_prophet = pd.DataFrame({
                'ds': df.index,
                'y': df['total_mensual']
            })
            
            self.modelo_prophet = Prophet(
                yearly_seasonality=componentes_estacionalidad,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            if len(df) > 12 and componentes_estacionalidad:
                self.modelo_prophet.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=3
                )
            
            self.modelo_prophet.fit(df_prophet)
            print("✅ Modelo Prophet entrenado")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def predecir_proximas_cuotas(self, df, n_predicciones=6, intervalo_confianza=True):
        """Realiza predicciones"""
        ultima_fecha = df.index[-1]
        fechas_futuras = pd.date_range(
            start=ultima_fecha + timedelta(days=1),
            periods=n_predicciones,
            freq='M'
        )
        
        resultados = pd.DataFrame(index=fechas_futuras)
        
        # ARIMA
        if self.modelo_arima is not None:
            pred_diff = self.modelo_arima.get_forecast(n_predicciones)
            ultimo_valor = df['total_mensual'].iloc[-1]
            pred_acumulada = [ultimo_valor]
            
            for diff_valor in pred_diff.predicted_mean:
                pred_acumulada.append(pred_acumulada[-1] + diff_valor)
            
            resultados['ARIMA'] = pred_acumulada[1:]
            
            if intervalo_confianza:
                intervalo = pred_diff.conf_int()
                resultados['ARIMA_inferior'] = intervalo.iloc[:, 0] + ultimo_valor
                resultados['ARIMA_superior'] = intervalo.iloc[:, 1] + ultimo_valor
        
        # Prophet
        if self.modelo_prophet is not None and self.usar_prophet:
            future_dates = pd.DataFrame({'ds': fechas_futuras})
            pred_prophet = self.modelo_prophet.predict(future_dates)
            resultados['Prophet'] = pred_prophet['yhat'].values
            
            if intervalo_confianza:
                resultados['Prophet_inferior'] = pred_prophet['yhat_lower'].values
                resultados['Prophet_superior'] = pred_prophet['yhat_upper'].values
        
        return resultados
    
    def visualizar_predicciones(self, df_historico, predicciones, titulo="Proyección de Cuotas"):
        """Visualización"""
        plt.figure(figsize=(14, 8))
        
        plt.plot(df_historico.index, df_historico['total_mensual'],
                'bo-', label='Histórico', linewidth=2, markersize=4)
        
        if 'ARIMA' in predicciones.columns:
            plt.plot(predicciones.index, predicciones['ARIMA'],
                    'r.--', label='ARIMA', linewidth=2, markersize=6)
            
            if 'ARIMA_inferior' in predicciones.columns:
                plt.fill_between(predicciones.index,
                               predicciones['ARIMA_inferior'],
                               predicciones['ARIMA_superior'],
                               color='red', alpha=0.2)
        
        if 'Prophet' in predicciones.columns:
            plt.plot(predicciones.index, predicciones['Prophet'],
                    'g.--', label='Prophet', linewidth=2, markersize=6)
            
            if 'Prophet_inferior' in predicciones.columns:
                plt.fill_between(predicciones.index,
                               predicciones['Prophet_inferior'],
                               predicciones['Prophet_superior'],
                               color='green', alpha=0.2)
        
        plt.title(titulo, fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Total Mensual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()