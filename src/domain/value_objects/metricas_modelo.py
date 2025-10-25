"""Value Object: Métricas del Modelo"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricasModelo:
    """
    Value Object inmutable que representa las métricas de evaluación de un modelo
    """

    r2: float
    mse: float
    rmse: float
    mae: float
    mape: Optional[float] = None

    def __post_init__(self):
        """Validaciones"""
        if self.r2 < -1 or self.r2 > 1:
            raise ValueError("R² debe estar entre -1 y 1")

        if self.mse < 0:
            raise ValueError("MSE no puede ser negativo")

        if self.rmse < 0:
            raise ValueError("RMSE no puede ser negativo")

        if self.mae < 0:
            raise ValueError("MAE no puede ser negativo")

        if self.mape is not None and self.mape < 0:
            raise ValueError("MAPE no puede ser negativo")

    @property
    def es_buen_ajuste(self) -> bool:
        """Determina si el modelo tiene buen ajuste"""
        return self.r2 >= 0.7 and self.mape is not None and self.mape <= 15

    def to_dict(self) -> dict:
        """Convierte a diccionario"""
        return {
            "r2": self.r2,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
        }
