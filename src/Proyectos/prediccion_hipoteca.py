import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class ModeloFinanciero:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.columnas = ['capital', 'gastos_fijos']
        self.historial_errores = []
    
    def preparar_datos(self, df, test_size=0.2):
        """Prepara datos para el modelo"""
        X = df[self.columnas]
        y = df['total_mensual']
        
        # Usar split temporal si hay suficiente data histórica
        if len(df) > 12:
            n_test = max(3, int(test_size * len(df)))
            X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
            y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
            print(f"⏰ Split temporal: Train={len(X_train)}, Test={len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print(f"📊 Split aleatorio: Train={len(X_train)}, Test={len(X_test)}")
            
        return X_train, X_test, y_train, y_test
    
    def crear_modelo(self, X_train, y_train):
        """Crea y entrena el modelo optimizado"""
        alphas = np.logspace(-3, 3, 20)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=alphas, scoring='r2', cv=min(5, len(X_train))))
        ])
        
        pipeline.fit(X_train, y_train)
        self.modelo = pipeline
        
        best_alpha = pipeline.named_steps['ridge'].alpha_
        best_score = pipeline.named_steps['ridge'].best_score_
        print(f"✅ Modelo entrenado - Mejor alpha: {best_alpha:.4f}, R²: {best_score:.4f}")
        
        return pipeline
    
    def evaluar_modelo(self, modelo, X_test, y_test):
        """Evaluación completa del modelo"""
        y_pred = modelo.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
        
        print("\n📊 EVALUACIÓN DEL MODELO:")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return y_pred, {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def crear_graficos(self, df, y_test, y_pred, X_test):
        """Gráficos informativos optimizados"""
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
        axes[0,1].scatter(range(len(residuos)), residuos, alpha=0.7)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Índice')
        axes[0,1].set_ylabel('Residuales')
        axes[0,1].set_title('Análisis de Residuales')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribución de residuales
        axes[1,0].hist(residuos, bins=15, alpha=0.7, edgecolor='black', density=True)
        axes[1,0].axvline(x=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Residuales')
        axes[1,0].set_ylabel('Densidad')
        axes[1,0].set_title('Distribución de Residuales')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Serie temporal
        if len(y_test) > 1:
            indices = range(len(y_test))
            axes[1,1].plot(indices, y_test.values, label='Real', marker='o', linewidth=2)
            axes[1,1].plot(indices, y_pred, label='Predicho', marker='s', linewidth=2)
            axes[1,1].set_xlabel('Período')
            axes[1,1].set_ylabel('Total Mensual')
            axes[1,1].set_title('Evolución Temporal - Test')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return residuos
    
    def predecir(self, capital, gastos_fijos):
        """Predicción simplificada"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None
            
        try:
            ejemplo = np.array([[capital, gastos_fijos]])
            pred = self.modelo.predict(ejemplo)[0]
            print(f"\n🎯 PREDICCIÓN:")
            print(f"   Capital: ${capital:,.2f}")
            print(f"   Gastos Fijos: ${gastos_fijos:,.2f}")
            print(f"   Total Mensual Predicho: ${pred:,.2f}")
            return pred
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None

def main():
    """Función principal optimizada para análisis financiero"""
    # Cargar datos
    path = 'c:/Users/omaroalvaradoc/Documents/Personal/Proyectos/CURSO IA/data/processed/Datos Movimientos Financieros Ajustados.csv'
    
    try:
        df = pd.read_csv(path)
        print(f"📊 Datos cargados: {len(df)} registros")
        
        # Convertir fechas
        month_dict = {
            'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
        }
        
        def convert_date(date_str):
            try:
                month, day = date_str.split()[:2]
                return f"2023-{month_dict.get(month, '01')}-{day.zfill(2)}"
            except:
                return f"2023-01-01"
        
        df['Fecha'] = df['Fecha'].apply(convert_date)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Limpiar valores numéricos
        def clean_numeric(x):
            if pd.isna(x) or x == '':
                return 0.0
            if isinstance(x, str):
                return float(x.replace(',', '').strip())
            return float(x)
        
        # Corregir nombre de columna de Saldos a Saldo
        df['Débitos'] = df['Débitos'].apply(clean_numeric)
        df['Créditos'] = df['Créditos'].apply(clean_numeric)
        df['Saldo'] = df['Saldo'].apply(clean_numeric)  # Cambiado de 'Saldos' a 'Saldo'
        
        # Agrupar por mes
        df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
            'Débitos': 'sum',
            'Créditos': 'sum',
            'Saldo': 'last'  # Cambiado de 'Saldos' a 'Saldo'
        }).reset_index()
        
        # Preparar datos para el modelo
        datos_modelo = pd.DataFrame({
            'fecha': df_mensual['Fecha'].dt.to_timestamp(),
            'total_mensual': df_mensual['Débitos'],
            'capital': df_mensual['Saldo'],  # Cambiado de 'Saldos' a 'Saldo'
            'gastos_fijos': df_mensual['Débitos'] - df_mensual['Créditos']
        })
        
        # Análisis descriptivo
        print("\n📊 ANÁLISIS FINANCIERO MENSUAL:")
        print(f"Período analizado: {len(datos_modelo)} meses")
        print(f"Gasto mensual promedio: ${datos_modelo['total_mensual'].mean():,.2f}")
        print(f"Saldo promedio: ${datos_modelo['capital'].mean():,.2f}")
        print(f"Gastos fijos promedio: ${datos_modelo['gastos_fijos'].mean():,.2f}")
        
        # Entrenar modelo si hay suficientes datos
        if len(datos_modelo) >= 6:
            print("\n🤖 ENTRENANDO MODELO PREDICTIVO...")
            modelo = ModeloFinanciero()
            
            X_train, X_test, y_train, y_test = modelo.preparar_datos(datos_modelo)
            modelo.crear_modelo(X_train, y_train)
            y_pred, metricas = modelo.evaluar_modelo(modelo.modelo, X_test, y_test)
            modelo.crear_graficos(datos_modelo, y_test, y_pred, X_test)
            
            # Predicción de ejemplo
            if len(datos_modelo) > 0:
                ultimo = datos_modelo.iloc[-1]
                modelo.predecir(ultimo['capital'], ultimo['gastos_fijos'])
        else:
            print("⚠️ Datos insuficientes para entrenar modelo (mínimo 6 meses)")
        
        # Análisis de categorías de gasto
        print("\n🏷️ ANÁLISIS POR CATEGORÍA:")
        # Cambiar 'Transacción' por 'Transacción_Detalle'
        top_transacciones = df['Transacción_Detalle'].value_counts().head(10)
        for trans, count in top_transacciones.items():
            monto_total = df[df['Transacción_Detalle'] == trans]['Débitos'].sum()
            print(f"   {trans[:40]:40} | {count:3d} trans | ${monto_total:>12,.2f}")
        
        # Gráficos de tendencia
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(datos_modelo['fecha'], datos_modelo['total_mensual'], 'b-', marker='o', linewidth=2)
        plt.title('Evolución de Gastos Mensuales')
        plt.xlabel('Fecha')
        plt.ylabel('Total Gastos ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.plot(datos_modelo['fecha'], datos_modelo['capital'], 'g-', marker='s', linewidth=2)
        plt.title('Evolución del Saldo')
        plt.xlabel('Fecha')
        plt.ylabel('Saldo ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        datos_modelo['gastos_fijos'].plot(kind='bar', color='orange', alpha=0.7)
        plt.title('Gastos Fijos por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Gastos Fijos ($)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        df_top = df.groupby('Lugar')['Débitos'].sum().nlargest(8)
        df_top.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Distribución de Gastos por Lugar (Top 8)')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error en el procesamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()