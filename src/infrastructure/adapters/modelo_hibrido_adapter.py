"""Adaptador para el Modelo Híbrido existente"""

import pandas as pd
from typing import List, Optional

from core.modelo_hibrido import ModeloHibrido
from domain.entities import DatosHipoteca, Prediccion
from domain.value_objects import ConfiguracionPrediccion


class ModeloHibridoAdapter:
    """
    Adaptador que envuelve el ModeloHibrido existente para usarlo en la arquitectura hexagonal
    """

    def __init__(self):
        self.modelo = ModeloHibrido()
        self.entrenado = False

    def entrenar(self, datos: List[DatosHipoteca]) -> bool:
        """
        Entrena el modelo con datos del dominio

        Args:
            datos: Lista de entidades DatosHipoteca

        Returns:
            True si el entrenamiento fue exitoso
        """
        try:
            # Convertir entidades a DataFrame
            df = self._entidades_a_dataframe(datos)

            # Entrenar modelo
            exito = self.modelo.entrenar(df)
            self.entrenado = exito

            return exito

        except Exception as e:
            print(f"Error entrenando modelo: {e}")
            return False

    def predecir(self, configuracion: ConfiguracionPrediccion) -> List[Prediccion]:
        """
        Realiza predicciones usando el modelo entrenado

        Args:
            configuracion: Configuración de la predicción

        Returns:
            Lista de predicciones
        """
        if not self.entrenado:
            raise ValueError("El modelo no ha sido entrenado")

        try:
            # Obtener predicciones del modelo
            df_pred = self.modelo.predecir_futuro(
                n_predicciones=configuracion.numero_predicciones,
                retornar_componentes=True,
            )

            # Convertir DataFrame a entidades
            predicciones = self._dataframe_a_predicciones(df_pred, configuracion)

            return predicciones

        except Exception as e:
            print(f"Error en predicción: {e}")
            return []

    def _entidades_a_dataframe(self, datos: List[DatosHipoteca]) -> pd.DataFrame:
        """Convierte lista de entidades a DataFrame"""
        data_dict = {
            "fecha": [d.fecha for d in datos],
            "capital": [d.capital for d in datos],
            "gastos_fijos": [d.gastos_fijos for d in datos],
            "total_mensual": [d.total_mensual for d in datos],
            "tasa_uvr": [d.tasa_uvr or 395.002 for d in datos],
            "tasa_dtf": [d.tasa_dtf or 7.12 for d in datos],
            "inflacion_ipc": [d.inflacion_ipc or 150.99 for d in datos],
            "tipo_pago": [d.tipo_pago for d in datos],
        }

        df = pd.DataFrame(data_dict)
        df.set_index("fecha", inplace=True)

        return df

    def _dataframe_a_predicciones(
        self, df: pd.DataFrame, configuracion: ConfiguracionPrediccion
    ) -> List[Prediccion]:
        """Convierte DataFrame de predicciones a entidades"""
        predicciones = []

        for fecha, row in df.iterrows():
            prediccion = Prediccion(
                fecha=fecha,
                valor_predicho=row["prediccion_hibrida"],
                componente_lineal=row.get("componente_lineal", 0),
                componente_temporal=row.get("componente_arima", 0),
                intervalo_confianza_inferior=(
                    row.get("ic_inferior")
                    if configuracion.incluir_intervalo_confianza
                    else None
                ),
                intervalo_confianza_superior=(
                    row.get("ic_superior")
                    if configuracion.incluir_intervalo_confianza
                    else None
                ),
                metricas=(
                    self.modelo.metricas if configuracion.incluir_componentes else None
                ),
            )
            predicciones.append(prediccion)

        return predicciones

    def obtener_metricas(self) -> dict:
        """Obtiene las métricas del modelo entrenado"""
        if not self.entrenado:
            return {}

        return self.modelo.metricas
