"""Entidad de dominio: Modelo"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime


@dataclass
class Modelo:
    """
    Representa un modelo de Machine Learning entrenado
    """

    id: str
    tipo: str  # 'hibrido', 'lineal', 'temporal'
    fecha_entrenamiento: datetime
    metricas: Dict[str, float] = field(default_factory=dict)
    parametros: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    activo: bool = True

    def __post_init__(self):
        """Validaciones después de inicialización"""
        tipos_validos = ["hibrido", "lineal", "temporal"]
        if self.tipo not in tipos_validos:
            raise ValueError(f"Tipo de modelo debe ser uno de: {tipos_validos}")

    @property
    def calidad(self) -> str:
        """Determina la calidad del modelo basado en métricas"""
        if "r2" in self.metricas:
            r2 = self.metricas["r2"]
            if r2 >= 0.9:
                return "Excelente"
            elif r2 >= 0.7:
                return "Buena"
            elif r2 >= 0.5:
                return "Aceptable"
            else:
                return "Pobre"
        return "Sin evaluar"

    def actualizar_metricas(self, nuevas_metricas: Dict[str, float]):
        """Actualiza las métricas del modelo"""
        self.metricas.update(nuevas_metricas)

    def to_dict(self) -> dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "tipo": self.tipo,
            "fecha_entrenamiento": self.fecha_entrenamiento.isoformat(),
            "metricas": self.metricas,
            "parametros": self.parametros,
            "version": self.version,
            "activo": self.activo,
            "calidad": self.calidad,
        }
