"""Adaptador para servicio externo de datos - Implementa puerto de salida"""

from typing import Optional

from ports.output_ports import IExternalDataService
from services.external_data_service import ExternalDataService


class ExternalDataAdapter(IExternalDataService):
    """
    Adaptador que conecta el dominio con el servicio externo de datos
    """

    def __init__(self):
        self.external_service = ExternalDataService()

    def obtener_tasa_uvr(self) -> Optional[float]:
        """Obtiene la tasa UVR actual"""
        return self.external_service.obtener_uvr()

    def obtener_tasa_dtf(self) -> Optional[float]:
        """Obtiene la tasa DTF actual"""
        return self.external_service.obtener_dtf()

    def obtener_inflacion_ipc(self) -> Optional[float]:
        """Obtiene el IPC actual"""
        return self.external_service.obtener_ipc()

    def obtener_valores_actuales(self) -> dict:
        """Obtiene todos los valores macroeconómicos actuales"""
        return self.external_service.obtener_valores_actuales()
