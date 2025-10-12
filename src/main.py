"""
Script principal para análisis de movimientos financieros
Proyecto: Predicción Hipoteca y Análisis Financiero
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio base al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42

def analizar_tendencias(df):
    """Análisis detallado de tendencias financieras"""
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Análisis mensual
    df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
        'Débitos': 'sum',
        'Créditos': 'sum',
        'Saldo': 'last'
    }).reset_index()
    
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
    
    # 4. Comparativa Mensual
    ax4 = plt.subplot(2, 2, 4)
    df_mensual.plot(x='Fecha', y=['Débitos', 'Créditos'], kind='bar', ax=ax4, color=['red', 'green'], alpha=0.7)
    ax4.set_title('Comparativa Mensual: Ingresos vs Gastos', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mes')
    ax4.set_ylabel('Monto ($)')
    ax4.legend(["Gastos", "Ingresos"])
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen estadístico
    print("\n" + "="*60)
    print("📊 RESUMEN DE TENDENCIAS FINANCIERAS")
    print("="*60)
    
    for index, row in df_mensual.iterrows():
        mes = row['Fecha'].strftime('%Y-%m')
        print(f"{mes}: Gastos=${row['Débitos']:,.2f}, Ingresos=${row['Créditos']:,.2f}, Saldo=${row['Saldo']:,.2f}")
    
    print("\nTop 10 categorías de gasto:")
    for categoria, total in top_categorias.items():
        print(f"  {categoria[:50]}: ${total:,.2f}")
    
    print("="*60 + "\n")


def analizar_categorias(df):
    """Análisis detallado de gastos por categorías con sugerencias de ahorro"""
    
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
        transaccion_upper = str(transaccion).upper()
        for categoria, palabras_clave in categorias.items():
            if any(palabra in transaccion_upper for palabra in palabras_clave):
                return categoria
        return 'Otros'
    
    df['Categoria'] = df['Transacción_Detalle'].apply(clasificar_transaccion)
    
    # Análisis por categoría
    gastos_por_categoria = df.groupby('Categoria')['Débitos'].agg(['sum', 'count', 'mean']).round(2)
    gastos_por_categoria = gastos_por_categoria.sort_values('sum', ascending=False)
    gastos_por_categoria['porcentaje'] = (gastos_por_categoria['sum'] / df['Débitos'].sum() * 100).round(2)
    
    # Visualización
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Gráfico de pastel
    ax1 = plt.subplot(2, 2, 1)
    colors = plt.cm.Set3(range(len(gastos_por_categoria)))
    ax1.pie(gastos_por_categoria['sum'], labels=gastos_por_categoria.index, 
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Distribución de Gastos por Categoría', fontsize=12, fontweight='bold')
    
    # 2. Gráfico de barras
    ax2 = plt.subplot(2, 2, 2)
    gastos_por_categoria['sum'].plot(kind='barh', ax=ax2, color=colors)
    ax2.set_xlabel('Monto Total ($)')
    ax2.set_title('Gastos Totales por Categoría', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Evolución mensual
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
    
    # 4. Promedio por transacción
    ax4 = plt.subplot(2, 2, 4)
    gastos_por_categoria['mean'].plot(kind='bar', ax=ax4, color='skyblue', alpha=0.7)
    ax4.set_title('Gasto Promedio por Transacción', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Categoría')
    ax4.set_ylabel('Promedio ($)')
    ax4.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir análisis
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
    
    top_3_gastos = gastos_por_categoria.head(3)
    
    for categoria, row in top_3_gastos.iterrows():
        monto = row['sum']
        ahorro_5 = monto * 0.05
        ahorro_10 = monto * 0.10
        ahorro_15 = monto * 0.15
        
        print(f"\n📌 {categoria}:")
        print(f"   Gasto actual: ${monto:,.2f} ({row['porcentaje']:.1f}% del total)")
        print(f"   Ahorro potencial:")
        print(f"      • Reduciendo 5%:  ${ahorro_5:,.2f}/mes → ${ahorro_5*12:,.2f}/año")
        print(f"      • Reduciendo 10%: ${ahorro_10:,.2f}/mes → ${ahorro_10*12:,.2f}/año")
        print(f"      • Reduciendo 15%: ${ahorro_15:,.2f}/mes → ${ahorro_15*12:,.2f}/año")
    
    # Análisis de gastos hormiga
    gastos_pequenos = df[df['Débitos'] < 50000]
    if len(gastos_pequenos) > 0:
        print(f"\n🐜 ANÁLISIS DE GASTOS HORMIGA:")
        print(f"   Total transacciones pequeñas (<$50,000): {len(gastos_pequenos)}")
        print(f"   Monto acumulado: ${gastos_pequenos['Débitos'].sum():,.2f}")
        print(f"   Promedio: ${gastos_pequenos['Débitos'].mean():,.2f}")
        print(f"   💡 Representan {(gastos_pequenos['Débitos'].sum()/total_gastos*100):.1f}% del total")
    
    print("="*80 + "\n")
    
    return gastos_por_categoria


def main():
    """Función principal - Análisis de tendencias financieras"""
    # Ruta actualizada
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_path, 'data', 'raw', 'Datos Movimientos Financieros.csv')
    
    if not os.path.exists(path):
        print(f"❌ Error: El archivo no se encuentra en la ruta: {path}")
        return
    
    try:
        df = pd.read_csv(path)
        print(f"📊 Datos cargados: {len(df)} registros\n")
        
        # Conversión de fechas mejorada
        month_dict = {
            'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
        }
        
        def convert_date_smart(date_str):
            try:
                parts = date_str.split()
                if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
                    year, month, day = parts[0], parts[1], parts[2]
                else:
                    month, day = parts[0], parts[1]
                    year = str(pd.Timestamp.now().year)
                
                return f"{year}-{month_dict.get(month, '01')}-{day.zfill(2)}"
            except Exception as e:
                print(f"⚠️ Error procesando fecha '{date_str}': {e}")
                return f"{pd.Timestamp.now().year}-01-01"

        df['Fecha'] = df['Fecha'].apply(convert_date_smart)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
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
    
        # Análisis de tendencias
        print("🔍 Iniciando análisis de tendencias...\n")
        analizar_tendencias(df)
        
        # Análisis de categorías
        print("\n🏷️ Iniciando análisis de categorías...\n")
        analizar_categorias(df)

    except Exception as e:
        print(f"❌ Error en el procesamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
