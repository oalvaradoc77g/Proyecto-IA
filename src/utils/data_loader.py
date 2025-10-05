import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
import numpy as np

class DataLoader:
    """Clase para cargar datos macroeconómicos del Banco de la República"""
    
    SERIES_CODES = {
        'UVR': '60',      # Código real UVR
        'DTF': '271',     # Código real DTF
        'IPC': '1',       # Código real IPC
    }
    
    def __init__(self):
        self.base_url = "https://suameca.banrep.gov.co/buscador-de-series/descargar"
    
    def obtener_serie_banrep(self, codigo_serie, fecha_inicio, fecha_fin, formato="csv"):
        """Descarga datos del Banco de la República con mejor manejo de errores"""
        try:
            params = {
                "serie": codigo_serie,
                "fechaInicial": fecha_inicio,
                "fechaFinal": fecha_fin,
                "formato": formato
            }
            
            resp = requests.get(self.base_url, params=params, timeout=20)
            resp.raise_for_status()
            
            # Detectar separador automáticamente
            sep = ';' if ';' in resp.text.splitlines()[0] else ','
            df = pd.read_csv(StringIO(resp.text), sep=sep, engine='python')
            
            # Normalizar nombres de columnas
            df.columns = [col.lower() for col in df.columns]
            
            # Identificar columnas de fecha y valor
            fecha_col = next(col for col in df.columns if 'fecha' in col or 'mes' in col)
            valor_col = next(col for col in df.columns if 'valor' in col or 'dato' in col)
            
            # Limpiar y formatear datos
            df = df[[fecha_col, valor_col]].rename(columns={
                fecha_col: 'fecha',
                valor_col: 'valor'
            })
            
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df['valor'] = pd.to_numeric(
                df['valor'].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
            
            # Asignar nombre según tipo de serie
            if codigo_serie == self.SERIES_CODES['UVR']:
                df = df.rename(columns={'valor': 'tasa_uvr'})
            elif codigo_serie == self.SERIES_CODES['DTF']:
                df = df.rename(columns={'valor': 'tasa_dtf'})
            elif codigo_serie == self.SERIES_CODES['IPC']:
                df = df.rename(columns={'valor': 'inflacion_ipc'})
            
            return df.dropna().sort_values('fecha').reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error descargando serie {codigo_serie}: {e}")
            return None
    
    def cargar_datos_macro(self, fecha_inicio, fecha_fin):
        """Carga todas las series macroeconómicas y las combina con mejor manejo de errores"""
        print("🔄 Cargando datos macroeconómicos...")
        
        dfs = {}
        # Cargar cada serie
        for nombre, codigo in self.SERIES_CODES.items():
            df = self.obtener_serie_banrep(codigo, fecha_inicio, fecha_fin)
            if df is not None:
                dfs[nombre] = df
                print(f"✅ {nombre}: {len(df)} registros cargados")
            else:
                print(f"⚠️ {nombre}: usando valores simulados")
                # Crear serie simulada como fallback
                fechas = pd.date_range(fecha_inicio, fecha_fin, freq='M')
                if nombre == 'UVR':
                    valores = np.random.normal(4.5, 0.2, len(fechas))
                elif nombre == 'DTF':
                    valores = np.random.normal(5.2, 0.3, len(fechas))
                else:  # IPC
                    valores = np.random.normal(3.8, 0.4, len(fechas))
                
                dfs[nombre] = pd.DataFrame({
                    'fecha': fechas,
                    f'tasa_{nombre.lower()}': valores
                })
        
        # Combinar todas las series
        df_macro = dfs['UVR']
        for nombre in ['DTF', 'IPC']:
            if nombre in dfs:
                df_macro = df_macro.merge(dfs[nombre], on='fecha', how='outer')
        
        # Limpiar y ordenar
        df_macro = (df_macro
                   .sort_values('fecha')
                   .fillna(method='ffill')
                   .fillna(method='bfill'))
        
        print("✅ Datos macroeconómicos combinados exitosamente")
        return df_macro
    
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
            
            # Obtener rango de fechas
            fecha_inicio = df_base.index.min().strftime('%Y-%m-%d')
            fecha_fin = df_base.index.max().strftime('%Y-%m-%d')
            
            # Cargar datos macro
            df_macro = self.cargar_datos_macro(fecha_inicio, fecha_fin)
            
            # Combinar datos
            df_enriquecido = df_base.merge(
                df_macro.set_index('fecha'),
                left_index=True,
                right_index=True,
                how='left'
            )
            
            return df_enriquecido
            
        except Exception as e:
            print(f"❌ Error enriqueciendo datos: {e}")
            return df_base
