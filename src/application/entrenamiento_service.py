"""Servicio de entrenamiento - Implementación del puerto de entrada"""

from typing import List, Optional

from ports.input_ports import IEntrenamientoService
from ports.output_ports import IModeloRepository, IDatosRepository
from domain.entities import DatosHipoteca, Modelo
from application.use_cases import EntrenarModeloUseCase


class EntrenamientoService(IEntrenamientoService):
    """
    Implementación del servicio de entrenamiento
    """

    def __init__(
        self, modelo_repository: IModeloRepository, datos_repository: IDatosRepository
    ):
        self.modelo_repository = modelo_repository
        self.datos_repository = datos_repository
        self.entrenar_use_case = EntrenarModeloUseCase(
            modelo_repository, datos_repository
        )

    def entrenar_modelo(
        self, datos: List[DatosHipoteca], tipo_modelo: str = "hibrido"
    ) -> Modelo:
        """
        Entrena un nuevo modelo
        """
        return self.entrenar_use_case.execute(datos, tipo_modelo)

    def evaluar_modelo(self, modelo_id: str) -> dict:
        """
        Evalúa un modelo existente
        """
        modelo = self.modelo_repository.obtener_por_id(modelo_id)
        if not modelo:
            raise ValueError(f"Modelo {modelo_id} no encontrado")

        return {
            "metricas": modelo.metricas,
            "calidad": modelo.calidad,
            "activo": modelo.activo,
        }

    def obtener_modelo_activo(self) -> Optional[Modelo]:
        """
        Obtiene el modelo actualmente activo
        """
        return self.modelo_repository.obtener_activo()
