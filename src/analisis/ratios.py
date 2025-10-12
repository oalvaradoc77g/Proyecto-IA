"""
Análisis de ratios financieros y salud financiera
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calcular_ratios_financieros(df):
    """Calcula y analiza ratios financieros clave"""
    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Agrupar por mes
    df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
        'Débitos': 'sum',
        'Créditos': 'sum',
        'Saldo': 'last'
    }).reset_index()
    
    df_mensual['Fecha'] = df_mensual['Fecha'].dt.to_timestamp()
    
    # Calcular ratios
    df_mensual['Ahorro_Mensual'] = df_mensual['Créditos'] - df_mensual['Débitos']
    df_mensual['Ratio_Ahorro'] = (df_mensual['Ahorro_Mensual'] / df_mensual['Créditos'] * 100).round(2)
    df_mensual['Ratio_Gastos'] = (df_mensual['Débitos'] / df_mensual['Créditos'] * 100).round(2)
    
    # Identificar deudas (préstamos, transferencias, etc.)
    palabras_deuda = ['PAGO PRESTAMO', 'CREDITO', 'CUOTA', 'FINANCIACION']
    df['Es_Deuda'] = df['Transacción_Detalle'].str.upper().str.contains('|'.join(palabras_deuda), na=False)
    
    deudas_mensuales = df[df['Es_Deuda']].groupby(df['Fecha'].dt.to_period('M'))['Débitos'].sum().reset_index()
    deudas_mensuales.columns = ['Fecha', 'Pago_Deudas']
    deudas_mensuales['Fecha'] = deudas_mensuales['Fecha'].dt.to_timestamp()
    
    df_mensual = df_mensual.merge(deudas_mensuales, on='Fecha', how='left')
    df_mensual['Pago_Deudas'] = df_mensual['Pago_Deudas'].fillna(0)
    
    df_mensual['Ratio_Deuda_Ingreso'] = (df_mensual['Pago_Deudas'] / df_mensual['Créditos'] * 100).round(2)
    
    # Liquidez (capacidad de cubrir gastos con saldo actual)
    df_mensual['Ratio_Liquidez'] = (df_mensual['Saldo'] / df_mensual['Débitos']).round(2)
    
    # Capacidad de ahorro (meses que puede vivir con el saldo actual)
    df_mensual['Meses_Emergencia'] = (df_mensual['Saldo'] / df_mensual['Débitos']).round(1)
    
    # Calcular promedios
    promedio_ahorro = df_mensual['Ratio_Ahorro'].mean()
    promedio_gastos = df_mensual['Ratio_Gastos'].mean()
    promedio_deuda = df_mensual['Ratio_Deuda_Ingreso'].mean()
    promedio_liquidez = df_mensual['Ratio_Liquidez'].mean()
    
    # Visualización
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Ratio de Ahorro
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df_mensual['Fecha'], df_mensual['Ratio_Ahorro'], 'g-o', linewidth=2, markersize=8)
    ax1.axhline(y=20, color='orange', linestyle='--', label='Meta recomendada (20%)', linewidth=2)
    ax1.axhline(y=promedio_ahorro, color='blue', linestyle=':', label=f'Promedio ({promedio_ahorro:.1f}%)', linewidth=2)
    ax1.set_title('Ratio de Ahorro (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Porcentaje (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Ratio de Gastos
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(df_mensual['Fecha'], df_mensual['Ratio_Gastos'], 'r-o', linewidth=2, markersize=8)
    ax2.axhline(y=80, color='orange', linestyle='--', label='Límite recomendado (80%)', linewidth=2)
    ax2.axhline(y=promedio_gastos, color='blue', linestyle=':', label=f'Promedio ({promedio_gastos:.1f}%)', linewidth=2)
    ax2.set_title('Ratio de Gastos (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Porcentaje (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Ratio Deuda/Ingreso
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(df_mensual['Fecha'], df_mensual['Ratio_Deuda_Ingreso'], 'purple', marker='o', linewidth=2, markersize=8)
    ax3.axhline(y=35, color='red', linestyle='--', label='Límite saludable (35%)', linewidth=2)
    ax3.axhline(y=promedio_deuda, color='blue', linestyle=':', label=f'Promedio ({promedio_deuda:.1f}%)', linewidth=2)
    ax3.set_title('Ratio Deuda/Ingreso (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Fecha')
    ax3.set_ylabel('Porcentaje (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Ratio de Liquidez
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(df_mensual['Fecha'], df_mensual['Ratio_Liquidez'], 'b-o', linewidth=2, markersize=8)
    ax4.axhline(y=1, color='orange', linestyle='--', label='Mínimo recomendado (1)', linewidth=2)
    ax4.axhline(y=promedio_liquidez, color='blue', linestyle=':', label=f'Promedio ({promedio_liquidez:.1f})', linewidth=2)
    ax4.set_title('Ratio de Liquidez', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Fondo de Emergencia (Meses)
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(df_mensual['Fecha'], df_mensual['Meses_Emergencia'], 'teal', marker='o', linewidth=2, markersize=8)
    ax5.axhline(y=3, color='orange', linestyle='--', label='Mínimo (3 meses)', linewidth=2)
    ax5.axhline(y=6, color='green', linestyle='--', label='Ideal (6 meses)', linewidth=2)
    ax5.set_title('Fondo de Emergencia (Meses)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Fecha')
    ax5.set_ylabel('Meses')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # 6. Comparativa de Ratios
    ax6 = plt.subplot(3, 3, 6)
    ratios_promedio = {
        'Ahorro': promedio_ahorro,
        'Gastos': promedio_gastos,
        'Deuda/Ing': promedio_deuda,
        'Liquidez': promedio_liquidez * 10  # Escalar para visualizar
    }
    colors_bars = ['green', 'red', 'purple', 'blue']
    ax6.bar(ratios_promedio.keys(), ratios_promedio.values(), color=colors_bars, alpha=0.7)
    ax6.set_title('Resumen de Ratios Promedio', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Valor (%)')
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Evolución del Saldo
    ax7 = plt.subplot(3, 3, 7)
    ax7.fill_between(df_mensual['Fecha'], df_mensual['Saldo'], alpha=0.3, color='blue')
    ax7.plot(df_mensual['Fecha'], df_mensual['Saldo'], 'b-o', linewidth=2, markersize=8)
    ax7.set_title('Evolución del Saldo', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Fecha')
    ax7.set_ylabel('Saldo ($)')
    ax7.grid(True, alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
    
    # 8. Ingresos vs Gastos vs Ahorro
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(df_mensual['Fecha'], df_mensual['Créditos'], 'g-', label='Ingresos', linewidth=2)
    ax8.plot(df_mensual['Fecha'], df_mensual['Débitos'], 'r-', label='Gastos', linewidth=2)
    ax8.plot(df_mensual['Fecha'], df_mensual['Ahorro_Mensual'], 'b-', label='Ahorro', linewidth=2)
    ax8.set_title('Flujo de Efectivo Mensual', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Fecha')
    ax8.set_ylabel('Monto ($)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)
    
    # 9. Semáforo de Salud Financiera
    ax9 = plt.subplot(3, 3, 9)
    
    # Calcular score de salud financiera
    score_ahorro = min(100, (promedio_ahorro / 20) * 100) if promedio_ahorro > 0 else 0
    score_gastos = max(0, 100 - promedio_gastos) if promedio_gastos <= 100 else 0
    score_deuda = max(0, 100 - (promedio_deuda / 35) * 100) if promedio_deuda >= 0 else 100
    score_liquidez = min(100, promedio_liquidez * 50) if promedio_liquidez > 0 else 0
    
    score_total = (score_ahorro + score_gastos + score_deuda + score_liquidez) / 4
    
    categorias = ['Ahorro', 'Gastos', 'Deuda', 'Liquidez']
    scores = [score_ahorro, score_gastos, score_deuda, score_liquidez]
    
    colors_sem = ['green' if s >= 70 else 'orange' if s >= 40 else 'red' for s in scores]
    ax9.barh(categorias, scores, color=colors_sem, alpha=0.7)
    ax9.set_xlim(0, 100)
    ax9.set_title(f'Salud Financiera: {score_total:.1f}/100', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Score')
    ax9.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir análisis detallado
    print("\n" + "="*90)
    print("📊 ANÁLISIS DE RATIOS FINANCIEROS Y SALUD FINANCIERA")
    print("="*90)
    
    print("\n💰 RATIOS PRINCIPALES:")
    print("-" * 90)
    
    # Ratio de Ahorro
    estado_ahorro = "✅ Excelente" if promedio_ahorro >= 20 else "⚠️ Mejorable" if promedio_ahorro >= 10 else "❌ Crítico"
    print(f"\n1. RATIO DE AHORRO: {promedio_ahorro:.2f}% {estado_ahorro}")
    print(f"   └─ Ahorro mensual promedio: ${df_mensual['Ahorro_Mensual'].mean():,.2f}")
    print(f"   └─ Meta recomendada: 20% de los ingresos")
    
    # Ratio de Gastos
    estado_gastos = "✅ Saludable" if promedio_gastos <= 80 else "⚠️ Elevado" if promedio_gastos <= 95 else "❌ Crítico"
    print(f"\n2. RATIO DE GASTOS: {promedio_gastos:.2f}% {estado_gastos}")
    print(f"   └─ Gastos promedio: ${df_mensual['Débitos'].mean():,.2f}")
    print(f"   └─ Límite recomendado: 80% de los ingresos")
    
    # Ratio Deuda/Ingreso
    estado_deuda = "✅ Saludable" if promedio_deuda <= 35 else "⚠️ Moderado" if promedio_deuda <= 50 else "❌ Alto riesgo"
    print(f"\n3. RATIO DEUDA/INGRESO: {promedio_deuda:.2f}% {estado_deuda}")
    print(f"   └─ Pago de deudas promedio: ${df_mensual['Pago_Deudas'].mean():,.2f}")
    print(f"   └─ Límite saludable: 35% de los ingresos")
    
    # Ratio de Liquidez
    estado_liquidez = "✅ Bueno" if promedio_liquidez >= 1 else "⚠️ Ajustado" if promedio_liquidez >= 0.5 else "❌ Crítico"
    print(f"\n4. RATIO DE LIQUIDEZ: {promedio_liquidez:.2f} {estado_liquidez}")
    print(f"   └─ Capacidad para cubrir gastos mensuales")
    print(f"   └─ Mínimo recomendado: 1.0")
    
    # Fondo de Emergencia
    meses_prom = df_mensual['Meses_Emergencia'].mean()
    estado_emergencia = "✅ Excelente" if meses_prom >= 6 else "⚠️ Aceptable" if meses_prom >= 3 else "❌ Insuficiente"
    print(f"\n5. FONDO DE EMERGENCIA: {meses_prom:.1f} meses {estado_emergencia}")
    print(f"   └─ Meses que puedes cubrir con el saldo actual")
    print(f"   └─ Recomendado: 3-6 meses de gastos")
    
    # Score de Salud Financiera
    print("\n" + "="*90)
    print("🏥 EVALUACIÓN DE SALUD FINANCIERA")
    print("="*90)
    
    print(f"\nScore Total: {score_total:.1f}/100")
    
    if score_total >= 80:
        salud = "🟢 EXCELENTE - Finanzas muy saludables"
    elif score_total >= 60:
        salud = "🟡 BUENA - Finanzas saludables con áreas de mejora"
    elif score_total >= 40:
        salud = "🟠 REGULAR - Requiere atención y mejoras"
    else:
        salud = "🔴 CRÍTICA - Acción inmediata necesaria"
    
    print(f"Estado: {salud}")
    
    print("\nDetalle por categoría:")
    for cat, score in zip(categorias, scores):
        emoji = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
        print(f"  {emoji} {cat}: {score:.1f}/100")
    
    # Recomendaciones
    print("\n" + "="*90)
    print("💡 RECOMENDACIONES PERSONALIZADAS")
    print("="*90)
    
    if promedio_ahorro < 20:
        print("\n📌 AHORRO:")
        print("   • Aumenta tu tasa de ahorro al 20% reduciendo gastos no esenciales")
        print("   • Automatiza el ahorro: transfiere el 20% de ingresos apenas los recibas")
        print(f"   • Meta mensual: ${df_mensual['Créditos'].mean() * 0.20:,.2f}")
    
    if promedio_gastos > 80:
        print("\n📌 GASTOS:")
        print("   • Reduce gastos al 80% de tus ingresos máximo")
        print("   • Revisa gastos recurrentes y elimina suscripciones innecesarias")
        print("   • Aplica la regla 50/30/20 (necesidades/deseos/ahorro)")
    
    if promedio_deuda > 35:
        print("\n📌 DEUDAS:")
        print("   • Ratio de deuda muy alto - Prioriza pagar deudas")
        print("   • Método bola de nieve: paga primero la deuda más pequeña")
        print("   • Evita nuevas deudas hasta bajar el ratio al 35%")
    
    if meses_prom < 3:
        print("\n📌 FONDO DE EMERGENCIA:")
        objetivo = df_mensual['Débitos'].mean() * 3
        print(f"   • Construye un fondo de emergencia de ${objetivo:,.2f} (3 meses)")
        print("   • Destina al menos 10% de ingresos mensualmente")
        print("   • Mantén el fondo en cuenta de ahorros de fácil acceso")
    
    print("\n" + "="*90 + "\n")
    
    return {
        'ratios': df_mensual,
        'score_total': score_total,
        'promedio_ahorro': promedio_ahorro,
        'promedio_gastos': promedio_gastos,
        'promedio_deuda': promedio_deuda,
        'promedio_liquidez': promedio_liquidez
    }
