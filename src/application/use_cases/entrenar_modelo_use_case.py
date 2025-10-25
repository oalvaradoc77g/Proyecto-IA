"""Caso de uso: Entrenar Modelo"""

from typing import List
from datetime import datetime
import uuid

from domain.entities import DatosHipoteca, Modelo
from domain.value_objects import MetricasModelo
from ports.output_ports import IModeloRepository, IDatosRepository


class EntrenarModeloUseCase:
    """
    Caso de uso para entrenar un nuevo modelo de predicción
    """

    def __init__(
        self, modelo_repository: IModeloRepository, datos_repository: IDatosRepository
    ):
        self.modelo_repository = modelo_repository
        self.datos_repository = datos_repository

    def execute(
        self, datos: List[DatosHipoteca], tipo_modelo: str = "hibrido"
    ) -> Modelo:
        """
        Ejecuta el caso de uso de entrenamiento

        Args:
            datos: Lista de datos históricos para entrenamiento
            tipo_modelo: Tipo de modelo a entrenar

        Returns:
            Modelo entrenado
        """
        # Validar datos
        if not datos or len(datos) < 8:
            raise ValueError("Se requieren al menos 8 registros para entrenar")

        # Crear modelo
        modelo_id = str(uuid.uuid4())
        modelo = Modelo(
            id=modelo_id,
            tipo=tipo_modelo,
            fecha_entrenamiento=datetime.now(),
            metricas={},
            parametros={"orden_arima_auto": True, "alpha_ridge": None},
            version="1.0.0",
            activo=True,
        )

        # Aquí se integraría con el ModeloHibrido real
        # Por ahora simulamos métricas
        modelo.metricas = {
            "r2": 0.85,
            "mse": 1000.0,
            "rmse": 31.62,
            "mae": 25.0,
            "mape": 5.2,
        }

        # Desactivar modelo anterior si existe
        modelo_anterior = self.modelo_repository.obtener_activo()
        if modelo_anterior:
            modelo_anterior.activo = False
            self.modelo_repository.actualizar(modelo_anterior)

        # Guardar nuevo modelo
        self.modelo_repository.guardar(modelo)

        return modelo
