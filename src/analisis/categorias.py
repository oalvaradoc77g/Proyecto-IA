"""
Análisis de gastos por categorías
"""

import pandas as pd
import matplotlib.pyplot as plt


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
