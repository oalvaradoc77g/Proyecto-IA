"""Puertos de entrada - Interfaces de servicios de aplicación"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from domain.entities import Prediccion, DatosHipoteca, Modelo
from domain.value_objects import ConfiguracionPrediccion


class IPrediccionService(ABC):
    """
    Puerto de entrada para el servicio de predicciones
    """

    @abstractmethod
    def predecir_cuotas_futuras(
        self, configuracion: ConfiguracionPrediccion
    ) -> List[Prediccion]:
        """
        Predice cuotas futuras basándose en el modelo activo

        Args:
            configuracion: Configuración de la predicción

        Returns:
            Lista de predicciones
        """
        pass

    @abstractmethod
    def obtener_predicciones_historicas(
        self,
        fecha_inicio: Optional[datetime] = None,
        fecha_fin: Optional[datetime] = None,
    ) -> List[Prediccion]:
        """
        Obtiene predicciones históricas

        Args:
            fecha_inicio: Fecha de inicio del rango
            fecha_fin: Fecha de fin del rango

        Returns:
            Lista de predicciones históricas
        """
        pass


class IEntrenamientoService(ABC):
    """
    Puerto de entrada para el servicio de entrenamiento
    """

    @abstractmethod
    def entrenar_modelo(
        self, datos: List[DatosHipoteca], tipo_modelo: str = "hibrido"
    ) -> Modelo:
        """
        Entrena un nuevo modelo

        Args:
            datos: Lista de datos históricos
            tipo_modelo: Tipo de modelo a entrenar

        Returns:
            Modelo entrenado
        """
        pass

    @abstractmethod
    def evaluar_modelo(self, modelo_id: str) -> dict:
        """
        Evalúa un modelo existente

        Args:
            modelo_id: ID del modelo a evaluar

        Returns:
            Diccionario con métricas de evaluación
        """
        pass

    @abstractmethod
    def obtener_modelo_activo(self) -> Optional[Modelo]:
        """
        Obtiene el modelo actualmente activo

        Returns:
            Modelo activo o None
        """
        pass
