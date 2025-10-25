"""Puertos (Interfaces) de la aplicación"""

from .input_ports import IPrediccionService, IEntrenamientoService
from .output_ports import (
    IModeloRepository,
    IPrediccionRepository,
    IDatosRepository,
    IExternalDataService,
)

__all__ = [
    "IPrediccionService",
    "IEntrenamientoService",
    "IModeloRepository",
    "IPrediccionRepository",
    "IDatosRepository",
    "IExternalDataService",
]
