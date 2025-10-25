"""Puertos de salida - Interfaces de repositorios y servicios externos"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
import pandas as pd

from domain.entities import Prediccion, DatosHipoteca, Modelo


class IModeloRepository(ABC):
    """
    Puerto de salida para persistencia de modelos
    """

    @abstractmethod
    def guardar(self, modelo: Modelo) -> bool:
        """Guarda un modelo"""
        pass

    @abstractmethod
    def obtener_por_id(self, modelo_id: str) -> Optional[Modelo]:
        """Obtiene un modelo por ID"""
        pass

    @abstractmethod
    def obtener_activo(self) -> Optional[Modelo]:
        """Obtiene el modelo activo"""
        pass

    @abstractmethod
    def listar_todos(self) -> List[Modelo]:
        """Lista todos los modelos"""
        pass

    @abstractmethod
    def actualizar(self, modelo: Modelo) -> bool:
        """Actualiza un modelo existente"""
        pass


class IPrediccionRepository(ABC):
    """
    Puerto de salida para persistencia de predicciones
    """

    @abstractmethod
    def guardar(self, prediccion: Prediccion) -> bool:
        """Guarda una predicción"""
        pass

    @abstractmethod
    def guardar_lote(self, predicciones: List[Prediccion]) -> bool:
        """Guarda múltiples predicciones"""
        pass

    @abstractmethod
    def obtener_por_fecha(
        self, fecha_inicio: datetime, fecha_fin: datetime
    ) -> List[Prediccion]:
        """Obtiene predicciones por rango de fechas"""
        pass

    @abstractmethod
    def obtener_ultimas(self, cantidad: int) -> List[Prediccion]:
        """Obtiene las últimas N predicciones"""
        pass


class IDatosRepository(ABC):
    """
    Puerto de salida para acceso a datos históricos
    """

    @abstractmethod
    def cargar_datos(self, ruta: str) -> pd.DataFrame:
        """Carga datos desde un archivo"""
        pass

    @abstractmethod
    def obtener_datos_historicos(
        self,
        fecha_inicio: Optional[datetime] = None,
        fecha_fin: Optional[datetime] = None,
    ) -> List[DatosHipoteca]:
        """Obtiene datos históricos"""
        pass

    @abstractmethod
    def guardar_datos(self, datos: List[DatosHipoteca]) -> bool:
        """Guarda datos históricos"""
        pass


class IExternalDataService(ABC):
    """
    Puerto de salida para servicios externos (APIs, etc.)
    """

    @abstractmethod
    def obtener_tasa_uvr(self) -> Optional[float]:
        """Obtiene la tasa UVR actual"""
        pass

    @abstractmethod
    def obtener_tasa_dtf(self) -> Optional[float]:
        """Obtiene la tasa DTF actual"""
        pass

    @abstractmethod
    def obtener_inflacion_ipc(self) -> Optional[float]:
        """Obtiene el IPC actual"""
        pass

    @abstractmethod
    def obtener_valores_actuales(self) -> dict:
        """Obtiene todos los valores macroeconómicos actuales"""
        pass
