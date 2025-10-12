"""
Dashboard interactivo de visualización financiera
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime


def crear_dashboard_interactivo(df):
    """Crea un dashboard interactivo con visualizaciones financieras"""
    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Preparar datos mensuales
    df_mensual = df.groupby(df['Fecha'].dt.to_period('M')).agg({
        'Débitos': 'sum',
        'Créditos': 'sum',
        'Saldo': 'last'
    }).reset_index()
    df_mensual['Fecha'] = df_mensual['Fecha'].dt.to_timestamp()
    df_mensual['Ahorro'] = df_mensual['Créditos'] - df_mensual['Débitos']
    df_mensual['Mes_Nombre'] = df_mensual['Fecha'].dt.strftime('%b %Y')
    
    # Clasificar transacciones
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
    
    # Dashboard 1: Visión General
    print("📊 Generando Dashboard 1: Visión General...")
    crear_dashboard_vision_general(df, df_mensual)
    
    # Dashboard 2: Análisis de Categorías
    print("📊 Generando Dashboard 2: Análisis de Categorías...")
    crear_dashboard_categorias(df)
    
    # Dashboard 3: Tendencias Temporales
    print("📊 Generando Dashboard 3: Tendencias Temporales...")
    crear_dashboard_tendencias(df_mensual)
    
    # Dashboard 4: Análisis Comparativo
    print("📊 Generando Dashboard 4: Análisis Comparativo...")
    crear_dashboard_comparativo(df, df_mensual)
    
    print("\n✅ Dashboards interactivos generados exitosamente!")


def crear_dashboard_vision_general(df, df_mensual):
    """Dashboard principal con métricas clave"""
    
    # Calcular métricas
    ingresos_total = df['Créditos'].sum()
    gastos_total = df['Débitos'].sum()
    saldo_actual = df['Saldo'].iloc[-1]
    ahorro_total = ingresos_total - gastos_total
    
    # Crear subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Evolución de Ingresos y Gastos', 'Saldo Actual',
                       'Distribución Mensual', 'Ahorro Mensual',
                       'Flujo de Efectivo Acumulado', 'Indicadores Clave'),
        specs=[[{'type': 'scatter'}, {'type': 'indicator'}],
               [{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'table'}]],
        row_heights=[0.35, 0.35, 0.30]
    )
    
    # 1. Evolución de Ingresos y Gastos
    fig.add_trace(
        go.Scatter(x=df_mensual['Fecha'], y=df_mensual['Créditos'],
                  name='Ingresos', mode='lines+markers',
                  line=dict(color='green', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_mensual['Fecha'], y=df_mensual['Débitos'],
                  name='Gastos', mode='lines+markers',
                  line=dict(color='red', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # 2. Indicador de Saldo
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=saldo_actual,
            title={'text': "Saldo Actual"},
            delta={'reference': df_mensual['Saldo'].iloc[-2] if len(df_mensual) > 1 else saldo_actual},
            gauge={
                'axis': {'range': [None, saldo_actual * 1.5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, saldo_actual * 0.5], 'color': "lightgray"},
                    {'range': [saldo_actual * 0.5, saldo_actual], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': saldo_actual * 0.8
                }
            }
        ),
        row=1, col=2
    )
    
    # 3. Distribución Mensual
    fig.add_trace(
        go.Bar(x=df_mensual['Mes_Nombre'], y=df_mensual['Créditos'],
              name='Ingresos', marker_color='green', opacity=0.7),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df_mensual['Mes_Nombre'], y=df_mensual['Débitos'],
              name='Gastos', marker_color='red', opacity=0.7),
        row=2, col=1
    )
    
    # 4. Ahorro Mensual
    colors = ['green' if x >= 0 else 'red' for x in df_mensual['Ahorro']]
    fig.add_trace(
        go.Bar(x=df_mensual['Mes_Nombre'], y=df_mensual['Ahorro'],
              name='Ahorro', marker_color=colors),
        row=2, col=2
    )
    
    # 5. Flujo de Efectivo Acumulado
    df_mensual['Ahorro_Acumulado'] = df_mensual['Ahorro'].cumsum()
    fig.add_trace(
        go.Scatter(x=df_mensual['Fecha'], y=df_mensual['Ahorro_Acumulado'],
                  name='Ahorro Acumulado', mode='lines+markers',
                  fill='tozeroy', line=dict(color='blue', width=3)),
        row=3, col=1
    )
    
    # 6. Tabla de Indicadores
    fig.add_trace(
        go.Table(
            header=dict(values=['Métrica', 'Valor'],
                       fill_color='paleturquoise',
                       align='left',
                       font=dict(size=12, color='black')),
            cells=dict(values=[
                ['Ingresos Totales', 'Gastos Totales', 'Ahorro Total', 
                 'Tasa de Ahorro', 'Promedio Mensual Ingresos', 'Promedio Mensual Gastos'],
                [f'${ingresos_total:,.2f}', f'${gastos_total:,.2f}', f'${ahorro_total:,.2f}',
                 f'{(ahorro_total/ingresos_total*100):.2f}%',
                 f'${df_mensual["Créditos"].mean():,.2f}',
                 f'${df_mensual["Débitos"].mean():,.2f}']
            ],
            fill_color='lavender',
            align='left',
            font=dict(size=11))
        ),
        row=3, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        title_text="📊 Dashboard Financiero - Visión General",
        title_font_size=24,
        showlegend=True,
        height=1200,
        hovermode='x unified'
    )
    
    fig.show()


def crear_dashboard_categorias(df):
    """Dashboard de análisis por categorías"""
    
    gastos_categoria = df.groupby('Categoria')['Débitos'].sum().sort_values(ascending=False)
    transacciones_categoria = df.groupby('Categoria').size()
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Gastos por Categoría', 'Número de Transacciones',
                       'Evolución Temporal Top 5', 'Gasto Promedio por Transacción'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 1. Pie Chart de Gastos
    fig.add_trace(
        go.Pie(labels=gastos_categoria.index, values=gastos_categoria.values,
              hole=0.4, textposition='inside', textinfo='label+percent'),
        row=1, col=1
    )
    
    # 2. Número de Transacciones
    fig.add_trace(
        go.Bar(x=transacciones_categoria.index, y=transacciones_categoria.values,
              marker_color='lightblue', text=transacciones_categoria.values,
              textposition='auto'),
        row=1, col=2
    )
    
    # 3. Evolución Top 5 Categorías
    df['Mes'] = df['Fecha'].dt.to_period('M')
    top5_categorias = gastos_categoria.head(5).index
    
    for categoria in top5_categorias:
        df_cat = df[df['Categoria'] == categoria]
        gastos_mes = df_cat.groupby('Mes')['Débitos'].sum()
        fig.add_trace(
            go.Scatter(x=gastos_mes.index.astype(str), y=gastos_mes.values,
                      name=categoria, mode='lines+markers', line=dict(width=2)),
            row=2, col=1
        )
    
    # 4. Gasto Promedio por Transacción
    gasto_promedio = df.groupby('Categoria')['Débitos'].mean().sort_values(ascending=False)
    fig.add_trace(
        go.Bar(x=gasto_promedio.index, y=gasto_promedio.values,
              marker_color='orange', text=[f'${x:,.0f}' for x in gasto_promedio.values],
              textposition='auto'),
        row=2, col=2
    )
    
    # Layout
    fig.update_layout(
        title_text="🏷️ Dashboard de Análisis por Categorías",
        title_font_size=24,
        showlegend=True,
        height=900
    )
    
    fig.show()


def crear_dashboard_tendencias(df_mensual):
    """Dashboard de tendencias temporales"""
    
    # Crear figura
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tendencia de Ahorro', 'Ratio Gastos/Ingresos (%)',
                       'Velocidad de Gasto', 'Proyección Lineal'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # 1. Tendencia de Ahorro con Media Móvil
    df_mensual['Ahorro_MA3'] = df_mensual['Ahorro'].rolling(window=3).mean()
    
    fig.add_trace(
        go.Scatter(x=df_mensual['Fecha'], y=df_mensual['Ahorro'],
                  name='Ahorro Real', mode='lines+markers',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_mensual['Fecha'], y=df_mensual['Ahorro_MA3'],
                  name='Media Móvil (3m)', mode='lines',
                  line=dict(color='orange', width=3, dash='dash')),
        row=1, col=1
    )
    
    # 2. Ratio Gastos/Ingresos
    df_mensual['Ratio_Gastos'] = (df_mensual['Débitos'] / df_mensual['Créditos'] * 100)
    
    fig.add_trace(
        go.Scatter(x=df_mensual['Fecha'], y=df_mensual['Ratio_Gastos'],
                  name='Ratio G/I', mode='lines+markers',
                  line=dict(color='purple', width=2),
                  fill='tozeroy'),
        row=1, col=2
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red",
                  annotation_text="Límite Saludable (80%)", row=1, col=2)
    
    # 3. Velocidad de Gasto (cambio mes a mes)
    df_mensual['Cambio_Gastos'] = df_mensual['Débitos'].pct_change() * 100
    colors_cambio = ['green' if x < 0 else 'red' for x in df_mensual['Cambio_Gastos'].fillna(0)]
    
    fig.add_trace(
        go.Bar(x=df_mensual['Mes_Nombre'], y=df_mensual['Cambio_Gastos'],
              marker_color=colors_cambio, name='Cambio % Gastos'),
        row=2, col=1
    )
    
    # 4. Proyección Lineal
    from scipy import stats
    df_mensual['Mes_Num'] = range(len(df_mensual))
    slope, intercept, r, p, se = stats.linregress(df_mensual['Mes_Num'], df_mensual['Saldo'])
    
    # Proyección 6 meses
    meses_futuros = range(len(df_mensual), len(df_mensual) + 6)
    proyeccion = [slope * x + intercept for x in meses_futuros]
    
    fig.add_trace(
        go.Scatter(x=df_mensual['Fecha'], y=df_mensual['Saldo'],
                  name='Saldo Real', mode='lines+markers',
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )
    
    fechas_futuras = pd.date_range(start=df_mensual['Fecha'].max(), periods=7, freq='MS')[1:]
    fig.add_trace(
        go.Scatter(x=fechas_futuras, y=proyeccion,
                  name='Proyección', mode='lines+markers',
                  line=dict(color='red', width=2, dash='dash')),
        row=2, col=2
    )
    
    # Layout
    fig.update_layout(
        title_text="📈 Dashboard de Tendencias Temporales",
        title_font_size=24,
        showlegend=True,
        height=900
    )
    
    fig.show()


def crear_dashboard_comparativo(df, df_mensual):
    """Dashboard comparativo y de análisis avanzado"""
    
    # Crear figura con diferentes tipos de gráficos
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Heatmap Mensual de Actividad', 'Distribución de Gastos',
                       'Waterfall de Flujo de Efectivo', 'Sunburst de Categorías'),
        specs=[[{'type': 'heatmap'}, {'type': 'box'}],
               [{'type': 'waterfall'}, {'type': 'sunburst'}]]
    )
    
    # 1. Heatmap de actividad por día
    df['Dia'] = df['Fecha'].dt.day
    df['Mes_Num'] = df['Fecha'].dt.month
    
    pivot_gastos = df.pivot_table(values='Débitos', index='Dia', 
                                   columns='Mes_Num', aggfunc='sum', fill_value=0)
    
    fig.add_trace(
        go.Heatmap(z=pivot_gastos.values, x=pivot_gastos.columns, y=pivot_gastos.index,
                  colorscale='Reds', text=pivot_gastos.values, texttemplate='%{text:.0f}',
                  textfont={"size": 8}),
        row=1, col=1
    )
    
    # 2. Box Plot de distribución de gastos por categoría
    top5_cat = df.groupby('Categoria')['Débitos'].sum().nlargest(5).index
    df_top5 = df[df['Categoria'].isin(top5_cat)]
    
    for cat in top5_cat:
        fig.add_trace(
            go.Box(y=df_top5[df_top5['Categoria'] == cat]['Débitos'],
                  name=cat, boxmean='sd'),
            row=1, col=2
        )
    
    # 3. Waterfall del flujo de efectivo
    saldo_inicial = df_mensual['Saldo'].iloc[0] - df_mensual['Ahorro'].iloc[0]
    
    fig.add_trace(
        go.Waterfall(
            name="Flujo", orientation="v",
            measure=["relative"] * len(df_mensual),
            x=df_mensual['Mes_Nombre'],
            y=df_mensual['Ahorro'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
        ),
        row=2, col=1
    )
    
    # 4. Sunburst de categorías y subcategorías
    # Preparar datos para sunburst
    gastos_cat = df.groupby('Categoria').agg({
        'Débitos': 'sum',
        'Transacción_Detalle': 'first'
    }).reset_index()
    
    # Crear datos jerárquicos
    labels = ['Total'] + gastos_cat['Categoria'].tolist()
    parents = [''] + ['Total'] * len(gastos_cat)
    values = [gastos_cat['Débitos'].sum()] + gastos_cat['Débitos'].tolist()
    
    fig.add_trace(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        ),
        row=2, col=2
    )
    
    # Layout
    fig.update_layout(
        title_text="📊 Dashboard Comparativo y Análisis Avanzado",
        title_font_size=24,
        showlegend=True,
        height=900
    )
    
    fig.show()
