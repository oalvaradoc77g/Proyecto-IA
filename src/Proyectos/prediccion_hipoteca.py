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

warnings.filterwarnings('ignore')

class ModeloHipoteca:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.columnas = ['capital', 'gastos_fijos']  # Modificado para usar gastos_fijos
        # Agregar nuevas columnas
        self.variables_macro = ['tasa_uvr', 'tasa_dtf', 'inflacion_ipc']
        self.columnas_categoricas = ['tipo_pago']
        self.ultimas_predicciones = {}  # Para tracking de predicciones
        self.historial_errores = []     # Para monitoreo de errores
        self.data_loader = DataLoader()  # Add DataLoader instance
    
    def cargar_datos(self, path):
        """Carga y valida los datos con mejor manejo de errores"""
        try:
            df = pd.read_excel(path)
            print(f"📊 Datos cargados: {len(df)} registros, {df.shape[1]} columnas")
            
            # Crear nueva variable gastos_fijos
            df['gastos_fijos'] = df['intereses'] + df['seguros']
            
            # Agregar tipo_pago si no existe
            if 'tipo_pago' not in df.columns:
                print("ℹ️ Agregando columna tipo_pago")
                # Calcular umbral usando Series en lugar de acceder directamente
                capital_medio = df['capital'].mean()
                df['tipo_pago'] = df['capital'].apply(
                    lambda x: 'Abono extra' if x > capital_medio * 1.2 else 'Ordinario'
                )
            
            # Verificar y convertir índice temporal
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                df.set_index('fecha', inplace=True)
            else:
                print("⚠️ Creando índice temporal automático")
                df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='M')
            
            # Enriquecer con datos macroeconómicos simulados
            print("🔄 Agregando variables macroeconómicas simuladas...")
            df_enriquecido = self.data_loader.enriquecer_datos(df)
            
            # Asegurar que todas las columnas necesarias existen
            columnas_requeridas = ['capital', 'gastos_fijos', 'total_mensual', 'tipo_pago'] + \
                                ['tasa_uvr', 'tasa_dtf', 'inflacion_ipc']
                                
            for col in columnas_requeridas:
                if col not in df_enriquecido.columns:
                    print(f"⚠️ Agregando columna faltante: {col}")
                    if col in self.data_loader.valores_actuales:
                        df_enriquecido[col] = self.data_loader.valores_actuales[col]
                    elif col == 'tipo_pago':
                        df_enriquecido[col] = 'Ordinario'  # valor por defecto
            
            return df_enriquecido
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def analizar_multicolinealidad(self, df):
        """Análisis completo de multicolinealidad"""
        X = df[self.columnas]
        
        # Matriz de correlación
        print("\n🔗 Matriz de Correlación:")
        corr_matrix = X.corr()
        print(corr_matrix.round(3))
        
        # VIF
        vif = self.calcular_vif(X)
        print("\n📊 Factor de Inflación de Varianza (VIF):")
        for var, vif_val in vif.items():
            status = "⚠️ ALTO" if vif_val > 10 else "✅ OK"
            print(f"  {var}: {vif_val:.2f} {status}")
            
        return vif
    
    def calcular_vif(self, X):
        """Calcula VIF con validación"""
        X_const = sm.add_constant(X)
        vif_data = pd.Series([variance_inflation_factor(X_const.values, i) 
                            for i in range(1, X_const.shape[1])], 
                           index=X.columns)
        return vif_data
    
    def preparar_datos(self, df, temporal=False, test_size=0.2):
        """Prepara datos con validación de splits"""
        X = df[self.columnas]
        y = df['total_mensual']
        
        if temporal and len(df) > 10:  # Solo si hay suficientes datos
            n_test = max(1, int(test_size * len(df)))
            X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
            y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
            print(f"⏰ Split temporal: Train={len(X_train)}, Test={len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=not temporal
            )
            print(f"📊 Split aleatorio: Train={len(X_train)}, Test={len(X_test)}")
            
        return X_train, X_test, y_train, y_test
    
    def crear_modelo(self, X_train, y_train):
        """Crea y entrena el modelo con más opciones"""
        # Buscar mejores alphas
        alphas = np.logspace(-3, 3, 50)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(
                alphas=alphas, 
                scoring='neg_mean_squared_error',
                cv=5  # Simplificamos el CV a un número fijo
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        self.modelo = pipeline
        
        # Información del modelo
        best_alpha = pipeline.named_steps['ridge'].alpha_
        best_score = pipeline.named_steps['ridge'].best_score_  # Removido el signo negativo
        print(f"✅ Modelo entrenado - Mejor alpha: {best_alpha:.4f}, Score: {best_score:.2f}")
        
        return pipeline
    
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
        
        return y_pred, {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def crear_graficos(self, df, y_test, y_pred, X_test):
        """Gráficos más informativos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Predicho vs Actual
        axes[0,0].scatter(y_test, y_pred, alpha=0.6)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0,0].set_xlabel('Valor Real')
        axes[0,0].set_ylabel('Valor Predicho')
        axes[0,0].set_title('Predicciones vs Valores Reales')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Residuales
        residuos = y_test - y_pred
        axes[0,1].plot(residuos, 'o-', alpha=0.7)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Índice')
        axes[0,1].set_ylabel('Residuales')
        axes[0,1].set_title('Análisis de Residuales')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribución de residuales
        axes[1,0].hist(residuos, bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(x=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Residuales')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].set_title('Distribución de Residuales')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Serie temporal (si aplica)
        if len(y_test) > 1:
            axes[1,1].plot(y_test.values, label='Real', marker='o')
            axes[1,1].plot(y_pred, label='Predicho', marker='s')
            axes[1,1].set_xlabel('Tiempo')
            axes[1,1].set_ylabel('Total Mensual')
            axes[1,1].set_title('Evolución Temporal')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
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
    
    def guardar_predicciones(self, predicciones, ruta='predicciones.json'):
        """Guarda predicciones con metadata"""
        try:
            datos = {
                'fecha_generacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predicciones': predicciones.to_dict() if hasattr(predicciones, 'to_dict') else predicciones,
                'metricas': self.metricas if hasattr(self, 'metricas') else {},
                'error_promedio': np.mean(self.historial_errores) if self.historial_errores else None
            }
            
            with open(ruta, 'w') as f:
                json.dump(datos, f, indent=4)
            print(f"✅ Predicciones guardadas en {ruta}")
            
        except Exception as e:
            print(f"❌ Error guardando predicciones: {e}")

def main():
    """Función principal para análisis de movimientos financieros"""
    # Cargar datos
    path = 'data/processed/Datos Movimientos Financieros Ajustados.csv'
    try:
        df = pd.read_csv(path)
        print(f"📊 Datos cargados: {len(df)} registros")
        
        # Preparar datos
        def convert_date(date_str):
            month_dict = {
                'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
                'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
                'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
            }
            month, day = date_str.split()
            return f"2023-{month_dict[month]}-{day.zfill(2)}"

        # Convertir fechas y valores numéricos
        df['Fecha'] = df['Fecha'].apply(convert_date)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        def clean_numeric(x):
            if isinstance(x, str):
                return float(x.replace(',', '').replace('"', ''))
            return float(x) if pd.notnull(x) else 0.0

        df['Débitos'] = df['Débitos'].apply(clean_numeric)
        df['Créditos'] = df['Créditos'].apply(clean_numeric)
        df['Saldo'] = df['Saldo'].apply(clean_numeric)

        # Agrupar por mes
        df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
            'Débitos': 'sum',
            'Créditos': 'sum',
            'Saldo': 'last',
            'Lugar': lambda x: x.value_counts().index[0]  # Lugar más frecuente
        }).reset_index()

        # Preparar datos para el modelo (versión simplificada)
        datos_modelo = pd.DataFrame({
            'fecha': df_mensual['Fecha'].dt.to_timestamp(),
            'total_mensual': df_mensual['Débitos'],
            'capital': df_mensual['Saldo'],
            'gastos_fijos': df_mensual['Débitos'] - df_mensual['Créditos'],
            'lugar_frecuente': df_mensual['Lugar']
        })

        # Análisis básico
        print("\n📊 ANÁLISIS DE GASTOS MENSUALES:")
        print(f"Promedio de gastos: ${datos_modelo['total_mensual'].mean():,.2f}")
        print(f"Máximo gasto mensual: ${datos_modelo['total_mensual'].max():,.2f}")
        print(f"Mínimo gasto mensual: ${datos_modelo['total_mensual'].min():,.2f}")

        # Análisis por categoría
        print("\n🏷️ TOP 5 LUGARES DE GASTO:")
        top_lugares = df.groupby('Lugar')['Débitos'].sum().sort_values(ascending=False).head()
        for lugar, monto in top_lugares.items():
            print(f"   {lugar}: ${monto:,.2f}")

        # Tendencia de gastos
        plt.figure(figsize=(12, 6))
        plt.plot(datos_modelo['fecha'], datos_modelo['total_mensual'], 'b-', marker='o')
        plt.title('Tendencia de Gastos Mensuales')
        plt.xlabel('Fecha')
        plt.ylabel('Total Gastos ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error en el procesamiento: {e}")

if __name__ == "__main__":
    main()