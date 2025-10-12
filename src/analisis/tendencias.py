"""
Análisis de tendencias financieras
"""

import pandas as pd
import matplotlib.pyplot as plt


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
