"""
Módulo de análisis financiero
"""

from .tendencias import analizar_tendencias
from .categorias import analizar_categorias
from .proyecciones import proyectar_tendencias
from .estacionales import analizar_patrones_estacionales
from .ratios import calcular_ratios_financieros
from .sensibilidad import analizar_sensibilidad
from .dashboard import crear_dashboard_interactivo

__all__ = [
    'analizar_tendencias',
    'analizar_categorias',
    'proyectar_tendencias',
    'analizar_patrones_estacionales',
    'calcular_ratios_financieros',
    'analizar_sensibilidad',
    'crear_dashboard_interactivo'
]
