import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio base al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Importar la función de ejemplo_uso
try:
    from examples.ejemplo_uso import main as ejemplo_uso_main
except ImportError:
    print("⚠️ No se pudo importar ejemplo_uso, continuando sin él...")
    ejemplo_uso_main = None

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42

def analizar_tendencias(df):
    """Análisis detallado de tendencias financieras"""
    # Asegurar que la fecha está en formato datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Análisis mensual
    df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
        'Débitos': 'sum',
        'Créditos': 'sum',
        'Saldo': 'last'
    }).reset_index()
    
    # Convertir Period a timestamp para plotting
    df_mensual['Fecha'] = df_mensual['Fecha'].dt.to_timestamp()
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Tendencia de Ingresos vs Gastos
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df_mensual['Fecha'], df_mensual['Débitos'], 'r-', label='Gastos', marker='o', linewidth=2)
    ax1.plot(df_mensual['Fecha'], df_mensual['Créditos'], 'g-', label='Ingresos', marker='s', linewidth=2)
    ax1.set_title('Tendencia de Ingresos vs Gastos', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Monto ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Tendencia del Saldo
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df_mensual['Fecha'], df_mensual['Saldo'], 'b-', label='Saldo', marker='o', linewidth=2)
    ax2.set_title('Tendencia del Saldo', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Saldo ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Distribución de Gastos por Categoría
    ax3 = plt.subplot(2, 2, 3)
    top_categorias = df.groupby('Transacción_Detalle')['Débitos'].sum().nlargest(10)
    ax3.bar(range(len(top_categorias)), top_categorias.values, color='orange', alpha=0.7)
    ax3.set_title('Top 10 Gastos por Categoría', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Categoría')
    ax3.set_ylabel('Total Gastos ($)')
    ax3.set_xticks(range(len(top_categorias)))
    ax3.set_xticklabels([cat[:20] + '...' if len(cat) > 20 else cat 
                         for cat in top_categorias.index], 
                        rotation=45, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Comparativa Mensual de Ingresos y Gastos
    ax4 = plt.subplot(2, 2, 4)
    df_mensual.plot(x='Fecha', y=['Débitos', 'Créditos'], kind='bar', ax=ax4, color=['red', 'green'], alpha=0.7)
    ax4.set_title('Comparativa Mensual: Ingresos vs Gastos', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mes')
    ax4.set_ylabel('Monto ($)')
    ax4.legend(["Gastos", "Ingresos"])
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen estadístico mejorado
    print("\n" + "="*60)
    print("📊 RESUMEN DE TENDENCIAS FINANCIERAS")
    print("="*60)
    
    # Resumen mensual
    for index, row in df_mensual.iterrows():
        mes = row['Fecha'].strftime('%Y-%m')
        debitos = row['Débitos']
        creditos = row['Créditos']
        saldo = row['Saldo']
        print(f"{mes}: Gastos=${debitos:,.2f}, Ingresos=${creditos:,.2f}, Saldo=${saldo:,.2f}")
    
    # Resumen por categoría
    print("\nTop 10 categorías de gasto:")
    top_categorias = df.groupby('Transacción_Detalle')['Débitos'].sum().nlargest(10)
    for categoria, total in top_categorias.items():
        print(f"  {categoria[:50]}: ${total:,.2f}")
    
    print("="*60 + "\n")

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

def optimize_lgbm(trial, X_train, y_train, X_test, y_test):
    """Función objetivo para Optuna"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': RANDOM_STATE
    }
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def analizar_categorias(df):
    """Análisis detallado de gastos por categorías con sugerencias de ahorro"""
    
    # Diccionario de clasificación de categorías
    categorias = {
        'Alimentación': ['COMPRA EN CANAL ELECTRONI', 'TIENDA', 'MARKET', 'FRUVER', 'CARNES', 
                        'PANADERIA', 'SUBWAY', 'KFC', 'RESTAURAN', 'ARA', 'D1', 'EXITO', 'OXXO'],
        'Transporte': ['RETIRO RED', 'GASOLINA', 'EDS', 'BIOMAX', 'TEXACO'],
        'Vivienda': ['FIRENZE', 'CONJUNTO', 'ACUEDUCTO', 'ENEL', 'CLARO', 'ETB', 'VANTI'],
        'Servicios Financieros': ['PAGO PRESTAMO', 'TRANSFEREN', 'TRASLADO', 'BANCO'],
        'Salud': ['DROGUERIA', 'FARMA', 'COMPENSAR'],
        'Entretenimiento': ['NEQUI', 'DAVIPLATA', 'BOLD'],
        'Educación': ['MATRICULA', 'PROFESIONAL'],
        'Otros': []
    }
    
    def clasificar_transaccion(transaccion):
        """Clasifica una transacción en su categoría correspondiente"""
        transaccion_upper = str(transaccion).upper()
        for categoria, palabras_clave in categorias.items():
            if any(palabra in transaccion_upper for palabra in palabras_clave):
                return categoria
        return 'Otros'
    
    # Aplicar clasificación
    df['Categoria'] = df['Transacción_Detalle'].apply(clasificar_transaccion)
    
    # Análisis por categoría
    gastos_por_categoria = df.groupby('Categoria')['Débitos'].agg(['sum', 'count', 'mean']).round(2)
    gastos_por_categoria = gastos_por_categoria.sort_values('sum', ascending=False)
    gastos_por_categoria['porcentaje'] = (gastos_por_categoria['sum'] / df['Débitos'].sum() * 100).round(2)
    
    # Crear visualización
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Gráfico de pastel - Distribución de gastos
    ax1 = plt.subplot(2, 2, 1)
    colors = plt.cm.Set3(range(len(gastos_por_categoria)))
    ax1.pie(gastos_por_categoria['sum'], labels=gastos_por_categoria.index, 
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Distribución de Gastos por Categoría', fontsize=12, fontweight='bold')
    
    # 2. Gráfico de barras - Monto total por categoría
    ax2 = plt.subplot(2, 2, 2)
    gastos_por_categoria['sum'].plot(kind='barh', ax=ax2, color=colors)
    ax2.set_xlabel('Monto Total ($)')
    ax2.set_title('Gastos Totales por Categoría', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Evolución mensual por categoría principal
    ax3 = plt.subplot(2, 2, 3)
    df['Mes'] = df['Fecha'].dt.to_period('M')
    top_categorias = gastos_por_categoria.head(4).index
    
    for categoria in top_categorias:
        df_cat = df[df['Categoria'] == categoria]
        gastos_mensuales = df_cat.groupby('Mes')['Débitos'].sum()
        ax3.plot(gastos_mensuales.index.astype(str), gastos_mensuales.values, 
                marker='o', label=categoria, linewidth=2)
    
    ax3.set_title('Evolución Mensual - Top 4 Categorías', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Mes')
    ax3.set_ylabel('Monto ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 4. Promedio de gasto por transacción
    ax4 = plt.subplot(2, 2, 4)
    gastos_por_categoria['mean'].plot(kind='bar', ax=ax4, color='skyblue', alpha=0.7)
    ax4.set_title('Gasto Promedio por Transacción', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Categoría')
    ax4.set_ylabel('Promedio ($)')
    ax4.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir análisis detallado
    print("\n" + "="*80)
    print("💰 ANÁLISIS DETALLADO DE GASTOS POR CATEGORÍAS")
    print("="*80)
    
    total_gastos = df['Débitos'].sum()
    
    for idx, (categoria, row) in enumerate(gastos_por_categoria.iterrows(), 1):
        print(f"\n{idx}. {categoria.upper()}")
        print(f"   ├─ Total gastado: ${row['sum']:,.2f} ({row['porcentaje']:.1f}%)")
        print(f"   ├─ Número de transacciones: {int(row['count'])}")
        print(f"   └─ Promedio por transacción: ${row['mean']:,.2f}")
    
    # Sugerencias de ahorro
    print("\n" + "="*80)
    print("💡 SUGERENCIAS DE AHORRO")
    print("="*80)
    
    # Identificar categorías con mayor potencial de ahorro
    top_3_gastos = gastos_por_categoria.head(3)
    
    for categoria, row in top_3_gastos.iterrows():
        porcentaje = row['porcentaje']
        monto = row['sum']
        ahorro_potencial_5 = monto * 0.05
        ahorro_potencial_10 = monto * 0.10
        ahorro_potencial_15 = monto * 0.15
        
        print(f"\n📌 {categoria}:")
        print(f"   Gasto actual: ${monto:,.2f} ({porcentaje:.1f}% del total)")
        print(f"   Ahorro potencial:")
        print(f"      • Reduciendo 5%:  ${ahorro_potencial_5:,.2f}/mes → ${ahorro_potencial_5*12:,.2f}/año")
        print(f"      • Reduciendo 10%: ${ahorro_potencial_10:,.2f}/mes → ${ahorro_potencial_10*12:,.2f}/año")
        print(f"      • Reduciendo 15%: ${ahorro_potencial_15:,.2f}/mes → ${ahorro_potencial_15*12:,.2f}/año")
    
    # Análisis de gastos hormiga
    gastos_pequenos = df[df['Débitos'] < 50000]
    if len(gastos_pequenos) > 0:
        print(f"\n🐜 ANÁLISIS DE GASTOS HORMIGA:")
        print(f"   Total de transacciones pequeñas (<$50,000): {len(gastos_pequenos)}")
        print(f"   Monto acumulado: ${gastos_pequenos['Débitos'].sum():,.2f}")
        print(f"   Promedio: ${gastos_pequenos['Débitos'].mean():,.2f}")
        print(f"   💡 Tip: Estos gastos pequeños suman {(gastos_pequenos['Débitos'].sum()/total_gastos*100):.1f}% del total")
    
    print("="*80 + "\n")
    
    return gastos_por_categoria

def main():
    """Función principal - Análisis de tendencias financieras"""
    # Llamar a la función principal de ejemplo_uso.py solo si está disponible
    if ejemplo_uso_main is not None:
        try:
            print("🔄 Ejecutando análisis complementario...")
            ejemplo_uso_main()
        except Exception as e:
            print(f"⚠️ Error en ejemplo_uso: {e}")
    
    # Ruta corregida
    path = 'C:\\Users\\omaroalvaradoc\\Documents\\Personal\\Proyectos\\CURSO IA\\src\\data\\Datos Movimientos Financieros.csv'
    
    if not os.path.exists(path):
        print(f"❌ Error: El archivo no se encuentra en la ruta: {path}")
        return
    
    try:
        # Cargar datos
        df = pd.read_csv(path)
        print(f"📊 Datos cargados: {len(df)} registros\n")
        
        # Convertir fechas - MEJORADO para manejar formato con año incluido
        month_dict = {
            'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
        }
        
        def convert_date_smart(date_str):
            """Conversión inteligente de fechas con soporte para formato con año incluido"""
            try:
                parts = date_str.split()
                
                # Verificar si el primer elemento es un año (4 dígitos)
                if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
                    # Formato "YYYY MES DIA"
                    year = parts[0]
                    month = parts[1]
                    day = parts[2]
                else:
                    # Formato anterior "MES DIA"
                    month = parts[0]
                    day = parts[1]
                    # Usar año actual si no se proporciona
                    year = str(pd.Timestamp.now().year)
                
                return f"{year}-{month_dict.get(month, '01')}-{day.zfill(2)}"
                
            except Exception as e:
                print(f"⚠️ Error procesando fecha '{date_str}': {e}")
                return f"{pd.Timestamp.now().year}-01-01"

        print("🔄 Aplicando conversión inteligente de fechas...")
        df['Fecha'] = df['Fecha'].apply(convert_date_smart)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Verificar y mostrar distribución de años
        print("📅 Distribución de datos por año:")
        años_dist = df['Fecha'].dt.year.value_counts().sort_index()
        for año, count in años_dist.items():
            print(f"   {año}: {count} registros")
        
        # Verificar conversión correcta por período
        print("\n📅 Verificando conversión de fechas por período:")
        fechas_unicas = df['Fecha'].dt.to_period('M').unique()
        for fecha in sorted(fechas_unicas):
            count = len(df[df['Fecha'].dt.to_period('M') == fecha])
            año_mes = str(fecha)
            print(f"   {año_mes}: {count} registros")
        
        # Advertencia si hay fechas futuras
        fecha_actual = pd.Timestamp.now()
        fechas_futuras = df[df['Fecha'] > fecha_actual]
        if len(fechas_futuras) > 0:
            print(f"\n⚠️ Se detectaron {len(fechas_futuras)} registros con fechas futuras")
            print("   Esto es normal si estás proyectando o tienes datos programados")
        
        # Limpiar valores numéricos
        def clean_numeric(x):
            if pd.isna(x) or x == '':
                return 0.0
            if isinstance(x, str):
                return float(x.replace(',', '').strip())
            return float(x)
        
        df['Débitos'] = df['Débitos'].apply(clean_numeric)
        df['Créditos'] = df['Créditos'].apply(clean_numeric)
        df['Saldo'] = df['Saldo'].apply(clean_numeric)
    
        # Realizar análisis de tendencias
        print("🔍 Iniciando análisis de tendencias...\n")
        analizar_tendencias(df)
        
        # Análisis de categorías
        print("\n🏷️ Iniciando análisis de categorías...\n")
        gastos_por_categoria = analizar_categorias(df)
        
        # 🔥 CONSOLIDADO: Una sola agrupación mensual
        df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
            'Débitos': 'sum',
            'Créditos': 'sum',
            'Saldo': 'last'
        }).reset_index()
        
        # Preparar datos para el modelo
        datos_modelo = pd.DataFrame({
            'fecha': df_mensual['Fecha'].dt.to_timestamp(),
            'total_mensual': df_mensual['Débitos'],
            'capital': df_mensual['Saldo'],
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