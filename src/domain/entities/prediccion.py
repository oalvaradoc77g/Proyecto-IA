"""Entidad de dominio: Predicción"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict


@dataclass
class Prediccion:
    """
    Representa una predicción de cuota hipotecaria
    """

    fecha: datetime
    valor_predicho: float
    componente_lineal: float
    componente_temporal: float
    intervalo_confianza_inferior: Optional[float] = None
    intervalo_confianza_superior: Optional[float] = None
    metricas: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validaciones después de inicialización"""
        if self.valor_predicho < 0:
            raise ValueError("El valor predicho no puede ser negativo")

        if self.intervalo_confianza_inferior and self.intervalo_confianza_superior:
            if self.intervalo_confianza_inferior > self.intervalo_confianza_superior:
                raise ValueError("El intervalo de confianza es inválido")

    @property
    def tiene_intervalo_confianza(self) -> bool:
        """Indica si la predicción tiene intervalo de confianza"""
        return (
            self.intervalo_confianza_inferior is not None
            and self.intervalo_confianza_superior is not None
        )

    @property
    def ancho_intervalo(self) -> Optional[float]:
        """Calcula el ancho del intervalo de confianza"""
        if self.tiene_intervalo_confianza:
            return self.intervalo_confianza_superior - self.intervalo_confianza_inferior
        return None

    def to_dict(self) -> Dict:
        """Convierte la predicción a diccionario"""
        return {
            "fecha": self.fecha.isoformat(),
            "valor_predicho": self.valor_predicho,
            "componente_lineal": self.componente_lineal,
            "componente_temporal": self.componente_temporal,
            "intervalo_confianza_inferior": self.intervalo_confianza_inferior,
            "intervalo_confianza_superior": self.intervalo_confianza_superior,
            "metricas": self.metricas,
        }
