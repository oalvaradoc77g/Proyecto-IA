"""Repositorio de predicciones - Implementación con persistencia en JSON"""

import json
import os
from typing import List
from datetime import datetime

from ports.output_ports import IPrediccionRepository
from domain.entities import Prediccion


class PrediccionRepository(IPrediccionRepository):
    """
    Implementación del repositorio de predicciones con persistencia en archivos JSON
    """

    def __init__(self, data_dir: str = "data/predictions"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def guardar(self, prediccion: Prediccion) -> bool:
        """Guarda una predicción"""
        return self.guardar_lote([prediccion])

    def guardar_lote(self, predicciones: List[Prediccion]) -> bool:
        """Guarda múltiples predicciones"""
        try:
            if not predicciones:
                return True

            # Crear directorio para el año/mes
            fecha = predicciones[0].fecha
            year_dir = os.path.join(self.data_dir, str(fecha.year))
            month_dir = os.path.join(year_dir, fecha.strftime("%B").lower())
            os.makedirs(month_dir, exist_ok=True)

            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(month_dir, f"predicciones_{timestamp}.json")

            # Convertir predicciones a dict
            data = {
                "fecha_generacion": datetime.now().isoformat(),
                "cantidad": len(predicciones),
                "predicciones": [p.to_dict() for p in predicciones],
            }

            # Guardar archivo
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)

            return True

        except Exception as e:
            print(f"Error guardando predicciones: {e}")
            return False

    def obtener_por_fecha(
        self, fecha_inicio: datetime, fecha_fin: datetime
    ) -> List[Prediccion]:
        """Obtiene predicciones por rango de fechas"""
        # Implementación simplificada
        # En producción, buscaría en los archivos correspondientes
        return []

    def obtener_ultimas(self, cantidad: int) -> List[Prediccion]:
        """Obtiene las últimas N predicciones"""
        # Implementación simplificada
        # En producción, buscaría los archivos más recientes
        return []
