import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class ModeloSeriesTiempo:
    def __init__(self):
        self.modelo_arima = None
        self.modelo_prophet = None
        
    def preparar_datos(self, df):
        """Prepara datos para series temporales"""
        # Crear índice temporal mensual
        df = df.copy()
        df['fecha'] = pd.date_range(start='2025-01-01', periods=len(df), freq='M')
        df.set_index('fecha', inplace=True)
        return df
    
    def entrenar_arima(self, df):
        """Entrena modelo ARIMA"""
        try:
            # Ajustar ARIMA(1,1,1) como punto de partida
            self.modelo_arima = ARIMA(df['total_mensual'], order=(1,1,1))
            self.modelo_arima = self.modelo_arima.fit()
            print("✅ Modelo ARIMA entrenado")
            return True
        except Exception as e:
            print(f"❌ Error entrenando ARIMA: {e}")
            return False
    
    def entrenar_prophet(self, df):
        """Entrena modelo Prophet"""
        try:
            # Preparar datos para Prophet
            df_prophet = pd.DataFrame({
                'ds': df.index,
                'y': df['total_mensual']
            })
            
            self.modelo_prophet = Prophet(yearly_seasonality=False,
                                       weekly_seasonality=False,
                                       daily_seasonality=False)
            self.modelo_prophet.fit(df_prophet)
            print("✅ Modelo Prophet entrenado")
            return True
        except Exception as e:
            print(f"❌ Error entrenando Prophet: {e}")
            return False
    
    def predecir_proximas_cuotas(self, df, n_predicciones=6):
        """Realiza predicciones con ambos modelos"""
        ultima_fecha = df.index[-1]
        fechas_futuras = pd.date_range(start=ultima_fecha + timedelta(days=1),
                                     periods=n_predicciones,
                                     freq='M')
        
        resultados = pd.DataFrame(index=fechas_futuras)
        
        # Predicciones ARIMA
        if self.modelo_arima is not None:
            pred_arima = self.modelo_arima.forecast(n_predicciones)
            resultados['ARIMA'] = pred_arima
        
        # Predicciones Prophet
        if self.modelo_prophet is not None:
            future_dates = pd.DataFrame({'ds': fechas_futuras})
            pred_prophet = self.modelo_prophet.predict(future_dates)
            resultados['Prophet'] = pred_prophet['yhat']
        
        return resultados
    
    def visualizar_predicciones(self, df_historico, predicciones):
        """Visualiza datos históricos y predicciones"""
        plt.figure(figsize=(12, 6))
        
        # Datos históricos
        plt.plot(df_historico.index, df_historico['total_mensual'],
                'b.-', label='Histórico', alpha=0.8)
        
        # Predicciones
        if 'ARIMA' in predicciones.columns:
            plt.plot(predicciones.index, predicciones['ARIMA'],
                    'r.--', label='ARIMA', alpha=0.8)
        if 'Prophet' in predicciones.columns:
            plt.plot(predicciones.index, predicciones['Prophet'],
                    'g.--', label='Prophet', alpha=0.8)
        
        plt.title('Proyección de Cuotas Hipotecarias')
        plt.xlabel('Fecha')
        plt.ylabel('Total Mensual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
