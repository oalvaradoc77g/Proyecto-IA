"""
Análisis de patrones estacionales
"""

import pandas as pd
import matplotlib.pyplot as plt


def analizar_patrones_estacionales(df):
    """Análisis de patrones estacionales en gastos e ingresos"""
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Mes'] = df['Fecha'].dt.month
    df['Dia_Semana'] = df['Fecha'].dt.dayofweek
    df['Quincena'] = df['Fecha'].dt.day.apply(lambda x: 1 if x <= 15 else 2)
    
    # Análisis por mes
    por_mes = df.groupby('Mes').agg({
        'Débitos': 'sum',
        'Créditos': 'sum'
    }).reset_index()
    
    # Análisis por día de la semana
    por_dia = df.groupby('Dia_Semana').agg({
        'Débitos': 'sum',
        'Créditos': 'sum'
    }).reset_index()
    
    # Análisis por quincena
    por_quincena = df.groupby('Quincena').agg({
        'Débitos': 'sum',
        'Créditos': 'sum'
    }).reset_index()
    
    # Visualización
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Patrón mensual
    ax1 = plt.subplot(2, 2, 1)
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    x = range(len(por_mes))
    width = 0.35
    ax1.bar([i - width/2 for i in x], por_mes['Débitos'], width, label='Gastos', color='red', alpha=0.7)
    ax1.bar([i + width/2 for i in x], por_mes['Créditos'], width, label='Ingresos', color='green', alpha=0.7)
    ax1.set_xlabel('Mes')
    ax1.set_ylabel('Monto ($)')
    ax1.set_title('Patrón de Gastos e Ingresos por Mes', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([meses[m-1] for m in por_mes['Mes']])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Patrón por día de semana
    ax2 = plt.subplot(2, 2, 2)
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    x = range(len(por_dia))
    ax2.bar([i - width/2 for i in x], por_dia['Débitos'], width, label='Gastos', color='red', alpha=0.7)
    ax2.bar([i + width/2 for i in x], por_dia['Créditos'], width, label='Ingresos', color='green', alpha=0.7)
    ax2.set_xlabel('Día de la Semana')
    ax2.set_ylabel('Monto ($)')
    ax2.set_title('Patrón de Gastos e Ingresos por Día', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([dias[d] for d in por_dia['Dia_Semana']])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Patrón quincenal
    ax3 = plt.subplot(2, 2, 3)
    x = range(len(por_quincena))
    ax3.bar([i - width/2 for i in x], por_quincena['Débitos'], width, label='Gastos', color='red', alpha=0.7)
    ax3.bar([i + width/2 for i in x], por_quincena['Créditos'], width, label='Ingresos', color='green', alpha=0.7)
    ax3.set_xlabel('Quincena')
    ax3.set_ylabel('Monto ($)')
    ax3.set_title('Patrón de Gastos e Ingresos por Quincena', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Primera Quincena', 'Segunda Quincena'])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Heatmap de gastos por mes
    ax4 = plt.subplot(2, 2, 4)
    gastos_mes = por_mes.set_index('Mes')['Débitos']
    colors_gastos = plt.cm.Reds(gastos_mes / gastos_mes.max())
    ax4.barh([meses[m-1] for m in por_mes['Mes']], por_mes['Débitos'], color=colors_gastos)
    ax4.set_xlabel('Monto ($)')
    ax4.set_title('Intensidad de Gastos por Mes', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir insights
    print("\n" + "="*80)
    print("📊 ANÁLISIS DE PATRONES ESTACIONALES")
    print("="*80)
    
    mes_mayor_gasto = por_mes.loc[por_mes['Débitos'].idxmax()]
    mes_menor_gasto = por_mes.loc[por_mes['Débitos'].idxmin()]
    dia_mayor_gasto = por_dia.loc[por_dia['Débitos'].idxmax()]
    
    print(f"\n📅 Patrones Mensuales:")
    print(f"   • Mes con más gastos: {meses[int(mes_mayor_gasto['Mes'])-1]} (${mes_mayor_gasto['Débitos']:,.2f})")
    print(f"   • Mes con menos gastos: {meses[int(mes_menor_gasto['Mes'])-1]} (${mes_menor_gasto['Débitos']:,.2f})")
    
    print(f"\n📆 Patrones Semanales:")
    print(f"   • Día con más gastos: {dias[int(dia_mayor_gasto['Dia_Semana'])]} (${dia_mayor_gasto['Débitos']:,.2f})")
    
    print(f"\n💡 Insights:")
    if por_quincena.iloc[0]['Débitos'] > por_quincena.iloc[1]['Débitos']:
        print("   • Gastas más en la primera quincena del mes")
    else:
        print("   • Gastas más en la segunda quincena del mes")
    
    print("="*80 + "\n")
