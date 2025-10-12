"""
Análisis de sensibilidad financiera - Escenarios What-If
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analizar_sensibilidad(df, meses_proyeccion=12):
    """Analiza cómo cambios en ingresos/gastos afectan la situación financiera"""
    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Calcular promedios base
    ingresos_promedio = df.groupby(df['Fecha'].dt.to_period('M'))['Créditos'].sum().mean()
    gastos_promedio = df.groupby(df['Fecha'].dt.to_period('M'))['Débitos'].sum().mean()
    saldo_actual = df['Saldo'].iloc[-1]
    
    # Definir escenarios
    escenarios = {
        'Pesimista': {'ingresos': -20, 'gastos': 10},
        'Moderado_Negativo': {'ingresos': -10, 'gastos': 5},
        'Base': {'ingresos': 0, 'gastos': 0},
        'Moderado_Positivo': {'ingresos': 10, 'gastos': -5},
        'Optimista': {'ingresos': 20, 'gastos': -10}
    }
    
    # Calcular proyecciones para cada escenario
    resultados = {}
    
    for nombre, cambios in escenarios.items():
        ingresos_ajustados = ingresos_promedio * (1 + cambios['ingresos'] / 100)
        gastos_ajustados = gastos_promedio * (1 + cambios['gastos'] / 100)
        
        saldos = [saldo_actual]
        ahorros = []
        
        for mes in range(meses_proyeccion):
            ahorro_mensual = ingresos_ajustados - gastos_ajustados
            nuevo_saldo = saldos[-1] + ahorro_mensual
            saldos.append(nuevo_saldo)
            ahorros.append(ahorro_mensual)
        
        resultados[nombre] = {
            'ingresos': ingresos_ajustados,
            'gastos': gastos_ajustados,
            'ahorro_mensual': ahorro_mensual,
            'saldos': saldos[1:],
            'saldo_final': saldos[-1],
            'ahorro_total': sum(ahorros)
        }
    
    # Análisis de puntos críticos
    cambios_ingresos = range(-30, 31, 5)
    cambios_gastos = range(-30, 31, 5)
    
    # Crear matriz de sensibilidad
    matriz_saldo_final = np.zeros((len(cambios_gastos), len(cambios_ingresos)))
    
    for i, cambio_gasto in enumerate(cambios_gastos):
        for j, cambio_ingreso in enumerate(cambios_ingresos):
            ing = ingresos_promedio * (1 + cambio_ingreso / 100)
            gast = gastos_promedio * (1 + cambio_gasto / 100)
            ahorro_mes = ing - gast
            saldo_final = saldo_actual + (ahorro_mes * meses_proyeccion)
            matriz_saldo_final[i, j] = saldo_final
    
    # Visualización
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Proyección de Saldos por Escenario
    ax1 = plt.subplot(3, 3, 1)
    meses = range(1, meses_proyeccion + 1)
    colors = {'Pesimista': 'red', 'Moderado_Negativo': 'orange', 'Base': 'blue', 
              'Moderado_Positivo': 'lightgreen', 'Optimista': 'green'}
    
    for nombre, datos in resultados.items():
        ax1.plot(meses, datos['saldos'], label=nombre, color=colors[nombre], 
                linewidth=2, marker='o', markersize=4)
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title(f'Proyección de Saldos ({meses_proyeccion} meses)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Mes')
    ax1.set_ylabel('Saldo ($)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Ahorro Mensual por Escenario
    ax2 = plt.subplot(3, 3, 2)
    nombres_esc = list(resultados.keys())
    ahorros_mensuales = [resultados[n]['ahorro_mensual'] for n in nombres_esc]
    colores_barras = [colors[n] for n in nombres_esc]
    
    bars = ax2.bar(nombres_esc, ahorros_mensuales, color=colores_barras, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_title('Ahorro Mensual por Escenario', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Ahorro ($)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 3. Saldo Final por Escenario
    ax3 = plt.subplot(3, 3, 3)
    saldos_finales = [resultados[n]['saldo_final'] for n in nombres_esc]
    bars = ax3.bar(nombres_esc, saldos_finales, color=colores_barras, alpha=0.7)
    ax3.axhline(y=saldo_actual, color='blue', linestyle='--', 
                label=f'Saldo Actual: ${saldo_actual:,.0f}', linewidth=2)
    ax3.set_title(f'Saldo Final Proyectado ({meses_proyeccion} meses)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Saldo ($)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Mapa de Calor - Sensibilidad
    ax4 = plt.subplot(3, 3, 4)
    im = ax4.imshow(matriz_saldo_final, cmap='RdYlGn', aspect='auto')
    ax4.set_xticks(range(len(cambios_ingresos)))
    ax4.set_yticks(range(len(cambios_gastos)))
    ax4.set_xticklabels([f'{x:+d}%' for x in cambios_ingresos], fontsize=8)
    ax4.set_yticklabels([f'{x:+d}%' for x in cambios_gastos], fontsize=8)
    ax4.set_xlabel('Cambio en Ingresos')
    ax4.set_ylabel('Cambio en Gastos')
    ax4.set_title('Mapa de Sensibilidad - Saldo Final', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Saldo Final ($)')
    
    # 5. Impacto de Cambio en Ingresos
    ax5 = plt.subplot(3, 3, 5)
    saldos_var_ingresos = []
    for cambio in range(-30, 31, 5):
        ing = ingresos_promedio * (1 + cambio / 100)
        ahorro = ing - gastos_promedio
        saldo = saldo_actual + (ahorro * meses_proyeccion)
        saldos_var_ingresos.append(saldo)
    
    ax5.plot(range(-30, 31, 5), saldos_var_ingresos, 'b-o', linewidth=2, markersize=6)
    ax5.axhline(y=saldo_actual, color='orange', linestyle='--', label='Saldo Actual', linewidth=2)
    ax5.axvline(x=0, color='gray', linestyle=':', linewidth=1)
    ax5.set_xlabel('Cambio en Ingresos (%)')
    ax5.set_ylabel('Saldo Final ($)')
    ax5.set_title('Sensibilidad a Cambios en Ingresos', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Impacto de Cambio en Gastos
    ax6 = plt.subplot(3, 3, 6)
    saldos_var_gastos = []
    for cambio in range(-30, 31, 5):
        gast = gastos_promedio * (1 + cambio / 100)
        ahorro = ingresos_promedio - gast
        saldo = saldo_actual + (ahorro * meses_proyeccion)
        saldos_var_gastos.append(saldo)
    
    ax6.plot(range(-30, 31, 5), saldos_var_gastos, 'r-o', linewidth=2, markersize=6)
    ax6.axhline(y=saldo_actual, color='orange', linestyle='--', label='Saldo Actual', linewidth=2)
    ax6.axvline(x=0, color='gray', linestyle=':', linewidth=1)
    ax6.set_xlabel('Cambio en Gastos (%)')
    ax6.set_ylabel('Saldo Final ($)')
    ax6.set_title('Sensibilidad a Cambios en Gastos', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Punto de Equilibrio
    ax7 = plt.subplot(3, 3, 7)
    meses_hasta_cero = []
    escenarios_punto = []
    
    for nombre, datos in resultados.items():
        if datos['ahorro_mensual'] < 0:
            meses = abs(saldo_actual / datos['ahorro_mensual']) if datos['ahorro_mensual'] != 0 else 0
            meses_hasta_cero.append(min(meses, meses_proyeccion))
            escenarios_punto.append(nombre)
    
    if meses_hasta_cero:
        ax7.barh(escenarios_punto, meses_hasta_cero, color='red', alpha=0.7)
        ax7.set_xlabel('Meses hasta saldo $0')
        ax7.set_title('Punto de Quiebre por Escenario', fontsize=12, fontweight='bold')
        ax7.grid(axis='x', alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Todos los escenarios\nson positivos', 
                ha='center', va='center', fontsize=12, transform=ax7.transAxes)
        ax7.set_title('Punto de Quiebre por Escenario', fontsize=12, fontweight='bold')
    
    # 8. Comparativa Ingresos vs Gastos por Escenario
    ax8 = plt.subplot(3, 3, 8)
    x_pos = np.arange(len(nombres_esc))
    width = 0.35
    
    ingresos_esc = [resultados[n]['ingresos'] for n in nombres_esc]
    gastos_esc = [resultados[n]['gastos'] for n in nombres_esc]
    
    ax8.bar(x_pos - width/2, ingresos_esc, width, label='Ingresos', color='green', alpha=0.7)
    ax8.bar(x_pos + width/2, gastos_esc, width, label='Gastos', color='red', alpha=0.7)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(nombres_esc, rotation=45, ha='right', fontsize=8)
    ax8.set_ylabel('Monto ($)')
    ax8.set_title('Ingresos vs Gastos por Escenario', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)
    
    # 9. Resumen de Riesgos
    ax9 = plt.subplot(3, 3, 9)
    
    # Calcular probabilidad de riesgo
    riesgos = {
        'Saldo Negativo': sum(1 for r in resultados.values() if r['saldo_final'] < 0) / len(resultados) * 100,
        'Ahorro Bajo': sum(1 for r in resultados.values() if r['ahorro_mensual'] < ingresos_promedio * 0.1) / len(resultados) * 100,
        'Alta Vulnerabilidad': sum(1 for r in resultados.values() if r['saldo_final'] < gastos_promedio * 3) / len(resultados) * 100
    }
    
    colors_riesgo = ['red' if v > 50 else 'orange' if v > 20 else 'green' for v in riesgos.values()]
    ax9.barh(list(riesgos.keys()), list(riesgos.values()), color=colors_riesgo, alpha=0.7)
    ax9.set_xlabel('Probabilidad (%)')
    ax9.set_xlim(0, 100)
    ax9.set_title('Análisis de Riesgos', fontsize=12, fontweight='bold')
    ax9.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir análisis detallado
    print("\n" + "="*100)
    print("🔍 ANÁLISIS DE SENSIBILIDAD FINANCIERA - ESCENARIOS WHAT-IF")
    print("="*100)
    
    print(f"\n📊 SITUACIÓN BASE:")
    print(f"   • Ingresos promedio mensuales: ${ingresos_promedio:,.2f}")
    print(f"   • Gastos promedio mensuales: ${gastos_promedio:,.2f}")
    print(f"   • Ahorro mensual actual: ${ingresos_promedio - gastos_promedio:,.2f}")
    print(f"   • Saldo actual: ${saldo_actual:,.2f}")
    
    print("\n" + "="*100)
    print(f"📈 PROYECCIONES POR ESCENARIO ({meses_proyeccion} meses):")
    print("="*100)
    
    for nombre, datos in resultados.items():
        cambio_ing = escenarios[nombre]['ingresos']
        cambio_gast = escenarios[nombre]['gastos']
        
        print(f"\n{'🔴' if 'Pesimista' in nombre else '🟠' if 'Negativo' in nombre else '🔵' if 'Base' in nombre else '🟢'} {nombre.upper()}")
        print(f"   Cambios: Ingresos {cambio_ing:+d}%, Gastos {cambio_gast:+d}%")
        print("-" * 100)
        print(f"   ├─ Ingresos ajustados: ${datos['ingresos']:,.2f}")
        print(f"   ├─ Gastos ajustados: ${datos['gastos']:,.2f}")
        print(f"   ├─ Ahorro mensual: ${datos['ahorro_mensual']:,.2f} {'✅' if datos['ahorro_mensual'] > 0 else '❌'}")
        print(f"   ├─ Ahorro total {meses_proyeccion} meses: ${datos['ahorro_total']:,.2f}")
        print(f"   └─ Saldo final: ${datos['saldo_final']:,.2f} {'✅' if datos['saldo_final'] > 0 else '❌'}")
    
    # Análisis crítico
    print("\n" + "="*100)
    print("⚠️ ANÁLISIS CRÍTICO DE SENSIBILIDAD")
    print("="*100)
    
    # Punto de equilibrio en ingresos
    punto_eq_ingresos = (gastos_promedio - ingresos_promedio) / ingresos_promedio * 100
    print(f"\n💡 INGRESOS:")
    if punto_eq_ingresos < 0:
        print(f"   • Actualmente tienes un margen de {abs(punto_eq_ingresos):.1f}% sobre el punto de equilibrio")
        print(f"   • Puedes soportar una reducción de hasta {abs(punto_eq_ingresos):.1f}% en ingresos")
    else:
        print(f"   • ⚠️ Necesitas aumentar ingresos en {punto_eq_ingresos:.1f}% para equilibrar")
    
    print(f"   • Una reducción del 10% en ingresos = ${ingresos_promedio * -0.10:,.2f}/mes")
    print(f"   • Un aumento del 10% en ingresos = ${ingresos_promedio * 0.10:,.2f}/mes")
    
    # Punto de equilibrio en gastos
    print(f"\n💡 GASTOS:")
    margen_gastos = (ingresos_promedio - gastos_promedio) / gastos_promedio * 100
    print(f"   • Margen actual antes de déficit: {margen_gastos:.1f}%")
    print(f"   • Puedes aumentar gastos hasta {margen_gastos:.1f}% sin entrar en déficit")
    print(f"   • Una reducción del 10% en gastos = ${gastos_promedio * 0.10:,.2f}/mes ahorrados")
    print(f"   • Un aumento del 10% en gastos = ${gastos_promedio * -0.10:,.2f}/mes en déficit")
    
    # Escenarios de riesgo
    print("\n" + "="*100)
    print("🚨 ESCENARIOS DE RIESGO")
    print("="*100)
    
    escenarios_negativos = {k: v for k, v in resultados.items() if v['saldo_final'] < 0}
    
    if escenarios_negativos:
        print(f"\n⚠️ {len(escenarios_negativos)} escenario(s) resultan en saldo negativo:")
        for nombre, datos in escenarios_negativos.items():
            print(f"   • {nombre}: Déficit de ${abs(datos['saldo_final']):,.2f}")
    else:
        print("\n✅ Ningún escenario analizado resulta en saldo negativo")
    
    # Recomendaciones
    print("\n" + "="*100)
    print("💡 RECOMENDACIONES ESTRATÉGICAS")
    print("="*100)
    
    if margen_gastos < 20:
        print("\n🔴 PRIORIDAD ALTA - Margen muy ajustado:")
        print("   1. Crear fondo de emergencia de 3-6 meses")
        print("   2. Reducir gastos variables al menos 15%")
        print("   3. Buscar fuentes adicionales de ingreso")
    
    print(f"\n📌 Para aumentar tu resiliencia financiera:")
    print(f"   • Aumentar ingresos 10% = ${ingresos_promedio * 0.10:,.2f}/mes más de ahorro")
    print(f"   • Reducir gastos 10% = ${gastos_promedio * 0.10:,.2f}/mes más de ahorro")
    print(f"   • Combinado (10% ambos) = ${(ingresos_promedio * 0.10) + (gastos_promedio * 0.10):,.2f}/mes más de ahorro")
    
    print(f"\n📌 Escenarios de mejora rápida:")
    mejora_5_gastos = gastos_promedio * 0.05
    mejora_10_gastos = gastos_promedio * 0.10
    print(f"   • Reduciendo gastos 5%: Ahorras ${mejora_5_gastos:,.2f}/mes = ${mejora_5_gastos * 12:,.2f}/año")
    print(f"   • Reduciendo gastos 10%: Ahorras ${mejora_10_gastos:,.2f}/mes = ${mejora_10_gastos * 12:,.2f}/año")
    
    print("\n" + "="*100 + "\n")
    
    return {
        'escenarios': resultados,
        'situacion_base': {
            'ingresos': ingresos_promedio,
            'gastos': gastos_promedio,
            'saldo_actual': saldo_actual
        },
        'matriz_sensibilidad': matriz_saldo_final
    }
