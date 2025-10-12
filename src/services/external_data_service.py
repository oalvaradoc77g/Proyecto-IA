import requests
import pandas as pd
from datetime import datetime

class ExternalDataService:
    """Servicio para obtener datos de fuentes externas"""
    
    def __init__(self):
        self.banrep_base_url = "https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/estadisticaEconomicaRestService"
        
    def obtener_ipc(self):
        """Obtiene el valor actual del IPC desde BanRep"""
        try:
            url = f"{self.banrep_base_url}/consultaMenuXId?idMenu=100002"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extraer el valor del IPC del objeto SERIES
            if 'SERIES' in data and len(data['SERIES']) > 0:
                serie_ipc = data['SERIES'][0]  # Tomar primera serie
                if 'valor' in serie_ipc:
                    valor_ipc = float(serie_ipc['valor'])
                    fecha_valor = serie_ipc.get('fecha', 'fecha no disponible')
                    print(f"✅ IPC obtenido: {valor_ipc} (fecha: {fecha_valor})")
                    return valor_ipc
                    
            print("⚠️ No se encontró el valor del IPC en la estructura esperada")
            return None
                
        except Exception as e:
            print(f"❌ Error obteniendo IPC: {e}")
            return None
            
    def obtener_dtf(self):
        """Obtiene el valor actual del DTF desde BanRep"""
        try:
            url = f"{self.banrep_base_url}/consultaMenuXId?idMenu=220003"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extraer el valor del DTF del objeto SERIES
            if 'SERIES' in data and len(data['SERIES']) > 0:
                serie_dtf = data['SERIES'][0]  # Tomar primera serie
                if 'valor' in serie_dtf:
                    valor_dtf = float(serie_dtf['valor'])
                    fecha_valor = serie_dtf.get('fecha', 'fecha no disponible')
                    print(f"✅ DTF obtenido: {valor_dtf}% (fecha: {fecha_valor})")
                    return valor_dtf
                    
            print("⚠️ No se encontró el valor del DTF en la estructura esperada")
            return None
                
        except Exception as e:
            print(f"❌ Error obteniendo DTF: {e}")
            return None
    
    def obtener_uvr(self):
        """Obtiene el valor actual del UVR desde BanRep"""
        try:
            url = f"{self.banrep_base_url}/consultaMenuXId?idMenu=100005"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'SERIES' in data and len(data['SERIES']) > 0:
                serie_uvr = data['SERIES'][0]
                if 'valor' in serie_uvr:
                    valor_uvr = float(serie_uvr['valor'])
                    print(f"✅ UVR obtenido: {valor_uvr}")
                    return valor_uvr
            return None
        except Exception as e:
            print(f"❌ Error obteniendo UVR: {e}")
            return None
    
    def obtener_valores_actuales(self):
        """Obtiene todos los valores actuales disponibles"""
        return {
            'ipc': self.obtener_ipc() or 150.99,  # Valor por defecto si falla
            'dtf': self.obtener_dtf() or 7.12,    # Valor por defecto si falla
            'uvr': self.obtener_uvr() or 395.002  # Valor por defecto si falla
        }
