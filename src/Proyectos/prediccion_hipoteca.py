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
    # Modelo de regresión
    modelo_hipoteca = ModeloHipoteca()
    
    # Cargar datos
    path = r'C:\Users\omaroalvaradoc\Documents\Personal\hipoteca_extractos_ene_sep_2025.xlsx'
    df = modelo_hipoteca.cargar_datos(path)
    
    if df is None or len(df) == 0:
        print("No se pudieron cargar los datos")
        return
    
    # Análisis exploratorio
    vif = modelo_hipoteca.analizar_multicolinealidad(df)
    
    # Preparar y entrenar modelo
    X_train, X_test, y_train, y_test = modelo_hipoteca.preparar_datos(df, temporal=True)
    modelo = modelo_hipoteca.crear_modelo(X_train, y_train)
    
    # Evaluar
    y_pred, metricas = modelo_hipoteca.evaluar_modelo(modelo, X_test, y_test)
    
    # Gráficos
    residuos = modelo_hipoteca.crear_graficos(df, y_test, y_pred, X_test)
    
    # Predicción para octubre
    # Tomamos los últimos valores conocidos del dataset
    ultimo_registro = df.iloc[-1]
    capital_octubre = ultimo_registro['capital']
    gastos_fijos_octubre = ultimo_registro['gastos_fijos']
    
    print("\n🗓️ PREDICCIÓN PARA OCTUBRE:")
    print("Usando los últimos valores conocidos:")
    modelo_hipoteca.predecir(capital_octubre, gastos_fijos_octubre, 0)  # El tercer parámetro ya no se usa

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
        columnas_requeridas = ['capital', 'gastos_fijos', 'total_mensual', 
                             'tasa_uvr', 'tasa_dtf', 'inflacion_ipc', 'tipo_pago']
        
        for col in columnas_requeridas:
            if col not in df.columns:
                print(f"⚠️ Falta columna {col}")
                return
        
        # Entrenamiento inicial
        if modelo_hibrido.entrenar(df):
            # 🔄 Validación con dato real
            try:
                mes_actual = df['total_mensual'].iloc[-1]  # Último valor conocido
                prediccion_anterior = modelo_hibrido.predecir_futuro(n_predicciones=1)[0]
                
                error_relativo = abs((mes_actual - prediccion_anterior) / mes_actual)
                
                print("\n🎯 VALIDACIÓN DEL MODELO:")
                print(f"   Valor Real: ${mes_actual:,.2f}")
                print(f"   Predicción: ${prediccion_anterior:,.2f}")
                print(f"   Error: {error_relativo:.2%}")
                
                if error_relativo > 0.15:  # Error mayor al 15%
                    print("\n⚠️  ERROR ALTO - Reentrenando modelo...")
                    # Agregar nuevo dato y reentrenar
                    df_actualizado = df.copy()
                    df_actualizado.loc[len(df_actualizado)] = {'total_mensual': mes_actual}
                    modelo_hibrido.entrenar(df_actualizado)
            except Exception as e:
                print(f"⚠️  Error en validación: {e}")
            
            # 🚀 DESPLIEGUE A PRODUCCIÓN
            print("\n🚀 MODELO EN PRODUCCIÓN")
            predicciones_futuras = modelo_hibrido.predecir_futuro(
                n_predicciones=6, 
                retornar_componentes=True
            )
            
            if predicciones_futuras is not None:
                print("\n📊 PREDICCIONES OPERACIONALES:")
                print("=" * 50)
                for fecha, row in predicciones_futuras.iterrows():
                    print(f"   {fecha.strftime('%B %Y')}:")
                    print(f"      Predicción: ${row['prediccion_hibrida']:,.2f}")
                    if 'ic_inferior' in row:
                        print(f"      Rango: ${row['ic_inferior']:,.2f} - ${row['ic_superior']:,.2f}")
                print("=" * 50)
                
                # Guardar predicciones
                try:
                    predicciones_futuras.to_excel('predicciones_produccion.xlsx')
                    print("\n✅ Predicciones guardadas en 'predicciones_produccion.xlsx'")
                except Exception as e:
                    print(f"⚠️  Error guardando predicciones: {e}")
                
    except Exception as e:
        print(f"❌ Error en producción: {e}")
    
    # Sistema de validación y monitoreo
    if modelo_hibrido is not None and len(df) > 0:
        ultimo_valor_real = df['total_mensual'].iloc[-1]
        ultima_prediccion = modelo_hibrido.ultimas_predicciones.get(
            df.index[-1].strftime('%Y-%m'), None)
        
        if ultima_prediccion is not None:
            if not modelo_hibrido.validar_prediccion(ultima_prediccion, ultimo_valor_real):
                print("🔄 Reentrenando modelo con nuevos datos...")
                modelo_hibrido.entrenar(df)  # Reentrenar
        
        # Guardar nuevas predicciones
        predicciones_futuras = modelo_hibrido.predecir_futuro(n_predicciones=3)
        if predicciones_futuras is not None:
            modelo_hibrido.guardar_predicciones(predicciones_futuras)
    
    print("\n✅ ANÁLISIS COMPLETADO")

if __name__ == "__main__":
    main()