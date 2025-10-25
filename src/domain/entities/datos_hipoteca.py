"""Entidad de dominio: Datos de Hipoteca"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DatosHipoteca:
    """
    Representa los datos mensuales de una hipoteca
    """

    fecha: datetime
    capital: float
    gastos_fijos: float
    total_mensual: float
    tasa_uvr: Optional[float] = None
    tasa_dtf: Optional[float] = None
    inflacion_ipc: Optional[float] = None
    tipo_pago: str = "Ordinario"

    def __post_init__(self):
        """Validaciones después de inicialización"""
        if self.capital < 0:
            raise ValueError("El capital no puede ser negativo")

        if self.gastos_fijos < 0:
            raise ValueError("Los gastos fijos no pueden ser negativos")

        if self.total_mensual < 0:
            raise ValueError("El total mensual no puede ser negativo")

        if self.tipo_pago not in ["Ordinario", "Abono extra"]:
            raise ValueError("Tipo de pago debe ser 'Ordinario' o 'Abono extra'")

    @property
    def tiene_datos_macro(self) -> bool:
        """Indica si tiene datos macroeconómicos"""
        return (
            self.tasa_uvr is not None
            and self.tasa_dtf is not None
            and self.inflacion_ipc is not None
        )

    def to_dict(self) -> dict:
        """Convierte a diccionario"""
        return {
            "fecha": self.fecha.isoformat(),
            "capital": self.capital,
            "gastos_fijos": self.gastos_fijos,
            "total_mensual": self.total_mensual,
            "tasa_uvr": self.tasa_uvr,
            "tasa_dtf": self.tasa_dtf,
            "inflacion_ipc": self.inflacion_ipc,
            "tipo_pago": self.tipo_pago,
        }
