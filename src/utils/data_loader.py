import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.external_data_service import ExternalDataService

class DataLoader:
    def __init__(self):
        self.external_service = ExternalDataService()
        # Actualizar nombres de columnas para coincidir con lo esperado
        self.valores_actuales = {
            'tasa_uvr': self.external_service.obtener_uvr() or 395.002,  
            'tasa_dtf': self.external_service.obtener_dtf() or 7.12,
            'inflacion_ipc': self.external_service.obtener_ipc() or 150.99
        }
    
    def enriquecer_datos(self, df_base):
        """Enriquece DataFrame base con datos macroeconómicos"""
        try:
            # Asegurar índice temporal
            if not isinstance(df_base.index, pd.DatetimeIndex):
                if 'fecha' in df_base.columns:
                    df_base['fecha'] = pd.to_datetime(df_base['fecha'])
                    df_base.set_index('fecha', inplace=True)
                else:
                    print("⚠️ Creando índice temporal")
                    df_base.index = pd.date_range(start='2025-01-01', periods=len(df_base), freq='M')
            
            # Agregar columnas macro directamente
            df_enriquecido = df_base.copy()
            for nombre_columna, valor in self.valores_actuales.items():
                print(f"✅ Agregando {nombre_columna}: {valor}")
                df_enriquecido[nombre_columna] = valor
            
            return df_enriquecido
            
        except Exception as e:
            print(f"❌ Error enriqueciendo datos: {e}")
            return df_base

if __name__ == "__main__":
    # Código de prueba
    loader = DataLoader()
    df_ejemplo = pd.DataFrame({
        'valor_credito': [1000000, 1500000, 2000000],
        'fecha': ['2024-01-31', '2024-02-29', '2024-03-31']
    })
    
    print("\n=== PRUEBA DEL DATA LOADER ===")
    df_enriquecido = loader.enriquecer_datos(df_ejemplo)
    print("\nDataFrame enriquecido:")
    print(df_enriquecido)
