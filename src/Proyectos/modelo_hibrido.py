import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ModeloHibrido:
    def __init__(self):
        self.modelo_lineal = None
        self.modelo_arima = None
        self.scaler = StandardScaler()
        self.orden_arima = (1,1,1)
    
    def separar_componentes(self, df):
        """Separa tendencia lineal y componente residual"""
        # Preparar features para modelo lineal
        X = df[['capital', 'gastos_fijos']].values
        y = df['total_mensual'].values
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Ajustar modelo lineal
        self.modelo_lineal = RidgeCV(alphas=np.logspace(-3, 3, 50))
        self.modelo_lineal.fit(X_scaled, y)
        
        # Obtener predicciones lineales y residuos
        y_pred_lineal = self.modelo_lineal.predict(X_scaled)
        residuos = y - y_pred_lineal
        
        return y_pred_lineal, residuos
    
    def entrenar(self, df):
        """Entrena el modelo híbrido"""
        print("🔄 Entrenando modelo híbrido...")
        
        # Obtener componentes
        _, residuos = self.separar_componentes(df)
        
        # Entrenar ARIMA en residuos
        self.modelo_arima = ARIMA(residuos, order=self.orden_arima)
        self.modelo_arima = self.modelo_arima.fit()
        
        print("✅ Modelo híbrido entrenado")
    
    def predecir(self, df, n_predicciones=6):
        """Realiza predicciones combinadas"""
        # Predicción componente lineal
        X_futuro = df[['capital', 'gastos_fijos']].iloc[-1:].values
        X_futuro = np.tile(X_futuro, (n_predicciones, 1))
        X_futuro_scaled = self.scaler.transform(X_futuro)
        pred_lineal = self.modelo_lineal.predict(X_futuro_scaled)
        
        # Predicción residuos
        pred_residuos = self.modelo_arima.forecast(n_predicciones)
        
        # Combinar predicciones
        pred_final = pred_lineal + pred_residuos
        
        fechas_futuras = pd.date_range(
            start=df.index[-1] + pd.DateOffset(months=1),
            periods=n_predicciones,
            freq='M'
        )
        
        return pd.Series(pred_final, index=fechas_futuras, name='prediccion_hibrida')
    
    def evaluar(self, df_test):
        """Evalúa el modelo híbrido"""
        y_true = df_test['total_mensual']
        
        # Predicción componentes
        X_test = df_test[['capital', 'gastos_fijos']].values
        X_test_scaled = self.scaler.transform(X_test)
        pred_lineal = self.modelo_lineal.predict(X_test_scaled)
        pred_residuos = self.modelo_arima.predict(start=0, end=len(df_test)-1)
        
        # Predicción final
        y_pred = pred_lineal + pred_residuos
        
        # Calcular métricas
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        print("\n📊 EVALUACIÓN MODELO HÍBRIDO:")
        print(f"MAE: {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
        
        return {'mae': mae, 'rmse': rmse}
