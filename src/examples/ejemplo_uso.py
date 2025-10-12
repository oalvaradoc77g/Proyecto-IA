import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Cargar datos desde el archivo CSV (sin Excel)
    data = pd.read_csv(r'C:\Users\omaroalvaradoc\Documents\Personal\Proyectos\CURSO IA\src\data\Datos Movimientos Financieros.csv')

    # Convertir fechas con soporte para formato con año incluido
    month_dict = {
        'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
        'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
        'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
    }
    
    def convert_date(date_str):
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
                # Determinar año basado en el mes (para compatibilidad con datos antiguos)
                year = '2024' if month in ['OCT', 'NOV', 'DIC'] else '2025'
            
            return f"{year}-{month_dict.get(month, '01')}-{day.zfill(2)}"
        except:
            return "2025-01-01"

    data['Fecha'] = data['Fecha'].apply(convert_date)
    data['Fecha'] = pd.to_datetime(data['Fecha'])

    # Agrupar los datos por fecha y calcular ingresos y gastos totales
    data['Total_Gastos'] = data['Débitos']
    data['Total_Ingresos'] = data['Créditos']
    data_grouped = data.groupby('Fecha').agg({'Total_Gastos': 'sum', 'Total_Ingresos': 'sum'}).reset_index()

    # Crear gráficos de líneas
    plt.figure(figsize=(12, 6))
    plt.plot(data_grouped['Fecha'], data_grouped['Total_Gastos'], label='Gastos', color='red')
    plt.plot(data_grouped['Fecha'], data_grouped['Total_Ingresos'], label='Ingresos', color='green')
    plt.title('Análisis de Tendencias de Ingresos y Gastos')
    plt.xlabel('Fecha')
    plt.ylabel('Monto')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
