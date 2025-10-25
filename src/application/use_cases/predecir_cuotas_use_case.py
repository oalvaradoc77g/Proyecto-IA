"""Caso de uso: Predecir Cuotas"""

from typing import List
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from domain.entities import Prediccion, Modelo
from domain.value_objects import ConfiguracionPrediccion
from ports.output_ports import IModeloRepository, IPrediccionRepository


class PredecirCuotasUseCase:
    """
    Caso de uso para predecir cuotas hipotecarias futuras
    """

    def __init__(
        self,
        modelo_repository: IModeloRepository,
        prediccion_repository: IPrediccionRepository,
    ):
        self.modelo_repository = modelo_repository
        self.prediccion_repository = prediccion_repository

    def execute(self, configuracion: ConfiguracionPrediccion) -> List[Prediccion]:
        """
        Ejecuta el caso de uso de predicción

        Args:
            configuracion: Configuración de la predicción

        Returns:
            Lista de predicciones
        """
        # Obtener modelo activo
        modelo = self.modelo_repository.obtener_activo()
        if not modelo:
            raise ValueError("No hay modelo activo disponible")

        # Generar predicciones
        predicciones = self._generar_predicciones(modelo, configuracion)

        # Guardar predicciones
        self.prediccion_repository.guardar_lote(predicciones)

        return predicciones

    def _generar_predicciones(
        self, modelo: Modelo, configuracion: ConfiguracionPrediccion
    ) -> List[Prediccion]:
        """
        Genera las predicciones usando el modelo
        """
        predicciones = []
        fecha_base = datetime.now()

        # Aquí se integraría con el ModeloHibrido real
        # Por ahora simulamos predicciones
        for i in range(configuracion.numero_predicciones):
            fecha_pred = fecha_base + relativedelta(months=i + 1)

            # Simular valores
            valor_base = 1_200_000
            variacion = i * 1000  # Pequeña variación por mes

            prediccion = Prediccion(
                fecha=fecha_pred,
                valor_predicho=valor_base + variacion,
                componente_lineal=valor_base,
                componente_temporal=variacion,
                intervalo_confianza_inferior=(
                    (valor_base + variacion) * 0.95
                    if configuracion.incluir_intervalo_confianza
                    else None
                ),
                intervalo_confianza_superior=(
                    (valor_base + variacion) * 1.05
                    if configuracion.incluir_intervalo_confianza
                    else None
                ),
                metricas=modelo.metricas if configuracion.incluir_componentes else None,
            )

            predicciones.append(prediccion)

        return predicciones
