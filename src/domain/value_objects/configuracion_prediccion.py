"""Value Object: Configuración de Predicción"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfiguracionPrediccion:
    """
    Value Object inmutable que representa la configuración para realizar predicciones
    """

    numero_predicciones: int = 6
    incluir_intervalo_confianza: bool = True
    incluir_componentes: bool = False
    nivel_confianza: float = 0.95

    def __post_init__(self):
        """Validaciones"""
        if self.numero_predicciones <= 0:
            raise ValueError("El número de predicciones debe ser mayor a 0")

        if self.numero_predicciones > 24:
            raise ValueError("El número de predicciones no puede exceder 24 meses")

        if self.nivel_confianza <= 0 or self.nivel_confianza >= 1:
            raise ValueError("El nivel de confianza debe estar entre 0 y 1")

    def to_dict(self) -> dict:
        """Convierte a diccionario"""
        return {
            "numero_predicciones": self.numero_predicciones,
            "incluir_intervalo_confianza": self.incluir_intervalo_confianza,
            "incluir_componentes": self.incluir_componentes,
            "nivel_confianza": self.nivel_confianza,
        }
