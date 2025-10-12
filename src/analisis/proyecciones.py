"""
Proyecciones financieras futuras
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def proyectar_tendencias(df, meses_futuro=6):
    """Proyección de tendencias financieras futuras usando regresión lineal"""
    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Agrupar por mes
    df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
        'Débitos': 'sum',
        'Créditos': 'sum',
        'Saldo': 'last'
    }).reset_index()
    
    df_mensual['Fecha'] = df_mensual['Fecha'].dt.to_timestamp()
    
    # Convertir fechas a números para regresión
    df_mensual['Mes_Num'] = range(len(df_mensual))
    
    # Proyecciones para Débitos
    slope_debitos, intercept_debitos, r_debitos, p_debitos, se_debitos = stats.linregress(
        df_mensual['Mes_Num'], df_mensual['Débitos']
    )
    
    # Proyecciones para Créditos
    slope_creditos, intercept_creditos, r_creditos, p_creditos, se_creditos = stats.linregress(
        df_mensual['Mes_Num'], df_mensual['Créditos']
    )
    
    # Crear proyecciones futuras
    ultimo_mes = df_mensual['Mes_Num'].max()
    meses_proyeccion = range(ultimo_mes + 1, ultimo_mes + meses_futuro + 1)
    
    proyeccion_debitos = [slope_debitos * x + intercept_debitos for x in meses_proyeccion]
    proyeccion_creditos = [slope_creditos * x + intercept_creditos for x in meses_proyeccion]
    
    # Generar fechas futuras
    ultima_fecha = df_mensual['Fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), periods=meses_futuro, freq='MS')
    
    # Calcular saldo proyectado
    saldo_actual = df_mensual['Saldo'].iloc[-1]
    saldos_proyectados = []
    
    for debito, credito in zip(proyeccion_debitos, proyeccion_creditos):
        saldo_actual = saldo_actual + credito - debito
        saldos_proyectados.append(saldo_actual)
    
    # Visualización
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Proyección de Gastos
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df_mensual['Fecha'], df_mensual['Débitos'], 'ro-', label='Gastos Reales', linewidth=2, markersize=8)
    ax1.plot(fechas_futuras, proyeccion_debitos, 'r--', label='Proyección Gastos', linewidth=2, markersize=8, marker='s')
    ax1.axvline(x=ultima_fecha, color='gray', linestyle=':', linewidth=2, label='Hoy')
    ax1.set_title('Proyección de Gastos', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Monto ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Proyección de Ingresos
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df_mensual['Fecha'], df_mensual['Créditos'], 'go-', label='Ingresos Reales', linewidth=2, markersize=8)
    ax2.plot(fechas_futuras, proyeccion_creditos, 'g--', label='Proyección Ingresos', linewidth=2, markersize=8, marker='s')
    ax2.axvline(x=ultima_fecha, color='gray', linestyle=':', linewidth=2, label='Hoy')
    ax2.set_title('Proyección de Ingresos', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Monto ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Proyección de Saldo
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df_mensual['Fecha'], df_mensual['Saldo'], 'bo-', label='Saldo Real', linewidth=2, markersize=8)
    ax3.plot(fechas_futuras, saldos_proyectados, 'b--', label='Proyección Saldo', linewidth=2, markersize=8, marker='s')
    ax3.axvline(x=ultima_fecha, color='gray', linestyle=':', linewidth=2, label='Hoy')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_title('Proyección de Saldo', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Fecha')
    ax3.set_ylabel('Saldo ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Comparativa General
    ax4 = plt.subplot(2, 2, 4)
    
    # Datos históricos
    ax4.plot(df_mensual['Fecha'], df_mensual['Débitos'], 'r-', label='Gastos Históricos', linewidth=2)
    ax4.plot(df_mensual['Fecha'], df_mensual['Créditos'], 'g-', label='Ingresos Históricos', linewidth=2)
    
    # Proyecciones
    ax4.plot(fechas_futuras, proyeccion_debitos, 'r--', label='Proyección Gastos', linewidth=2)
    ax4.plot(fechas_futuras, proyeccion_creditos, 'g--', label='Proyección Ingresos', linewidth=2)
    
    ax4.axvline(x=ultima_fecha, color='gray', linestyle=':', linewidth=2, label='Hoy')
    ax4.set_title('Comparativa: Histórico vs Proyección', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('Monto ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen de proyecciones
    print("\n" + "="*80)
    print("🔮 PROYECCIONES FINANCIERAS FUTURAS")
    print("="*80)
    
    print(f"\n📈 Tendencias detectadas:")
    print(f"   • Gastos: {'↗️ Incrementando' if slope_debitos > 0 else '↘️ Disminuyendo'} ${abs(slope_debitos):,.2f}/mes")
    print(f"   • Ingresos: {'↗️ Incrementando' if slope_creditos > 0 else '↘️ Disminuyendo'} ${abs(slope_creditos):,.2f}/mes")
    print(f"   • Confiabilidad Gastos: {abs(r_debitos)*100:.1f}%")
    print(f"   • Confiabilidad Ingresos: {abs(r_creditos)*100:.1f}%")
    
    print(f"\n📅 Proyección para los próximos {meses_futuro} meses:")
    print("-" * 80)
    
    for i, fecha in enumerate(fechas_futuras):
        mes = fecha.strftime('%B %Y')
        debito = proyeccion_debitos[i]
        credito = proyeccion_creditos[i]
        saldo = saldos_proyectados[i]
        balance = credito - debito
        
        print(f"\n{mes}:")
        print(f"   ├─ Gastos proyectados: ${debito:,.2f}")
        print(f"   ├─ Ingresos proyectados: ${credito:,.2f}")
        print(f"   ├─ Balance mensual: ${balance:,.2f} {'✅' if balance > 0 else '⚠️'}")
        print(f"   └─ Saldo estimado: ${saldo:,.2f} {'✅' if saldo > 0 else '❌'}")
    
    # Alertas y recomendaciones
    print("\n" + "="*80)
    print("⚠️ ALERTAS Y RECOMENDACIONES")
    print("="*80)
    
    # Alerta de saldo negativo
    saldos_negativos = [i for i, s in enumerate(saldos_proyectados) if s < 0]
    if saldos_negativos:
        primer_mes_negativo = fechas_futuras[saldos_negativos[0]].strftime('%B %Y')
        print(f"\n🚨 ALERTA: Saldo negativo proyectado en {primer_mes_negativo}")
        print(f"   Déficit estimado: ${abs(saldos_proyectados[saldos_negativos[0]]):,.2f}")
    
    # Recomendaciones basadas en tendencias
    if slope_debitos > slope_creditos:
        diferencia = slope_debitos - slope_creditos
        print(f"\n💡 Los gastos crecen más rápido que los ingresos (${diferencia:,.2f}/mes)")
        print("   Recomendaciones:")
        print("   1. Revisar y reducir gastos recurrentes")
        print("   2. Buscar fuentes adicionales de ingreso")
        print("   3. Crear un fondo de emergencia")
    
    # Proyección de ahorro
    if slope_creditos > slope_debitos:
        ahorro_mensual = slope_creditos - slope_debitos
        ahorro_semestral = sum(proyeccion_creditos) - sum(proyeccion_debitos)
        print(f"\n💰 Capacidad de ahorro proyectada:")
        print(f"   • Mensual: ${ahorro_mensual:,.2f}")
        print(f"   • Semestral: ${ahorro_semestral:,.2f}")
        print(f"   • Anual estimado: ${ahorro_semestral*2:,.2f}")
    
    print("="*80 + "\n")
    
    return {
        'proyeccion_debitos': proyeccion_debitos,
        'proyeccion_creditos': proyeccion_creditos,
        'saldos_proyectados': saldos_proyectados,
        'fechas_futuras': fechas_futuras
    }
