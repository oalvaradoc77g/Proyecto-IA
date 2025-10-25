"""Servicio de predicciones - Implementación del puerto de entrada"""

from typing import List, Optional
from datetime import datetime

from ports.input_ports import IPrediccionService
from ports.output_ports import IModeloRepository, IPrediccionRepository
from domain.entities import Prediccion
from domain.value_objects import ConfiguracionPrediccion
from application.use_cases import PredecirCuotasUseCase


class PrediccionService(IPrediccionService):
    """
    Implementación del servicio de predicciones
    """

    def __init__(
        self,
        modelo_repository: IModeloRepository,
        prediccion_repository: IPrediccionRepository,
    ):
        self.modelo_repository = modelo_repository
        self.prediccion_repository = prediccion_repository
        self.predecir_use_case = PredecirCuotasUseCase(
            modelo_repository, prediccion_repository
        )

    def predecir_cuotas_futuras(
        self, configuracion: ConfiguracionPrediccion
    ) -> List[Prediccion]:
        """
        Predice cuotas futuras
        """
        return self.predecir_use_case.execute(configuracion)

    def obtener_predicciones_historicas(
        self,
        fecha_inicio: Optional[datetime] = None,
        fecha_fin: Optional[datetime] = None,
    ) -> List[Prediccion]:
        """
        Obtiene predicciones históricas
        """
        if fecha_inicio and fecha_fin:
            return self.prediccion_repository.obtener_por_fecha(fecha_inicio, fecha_fin)
        else:
            return self.prediccion_repository.obtener_ultimas(10)
