"""
PUNTO DE ENTRADA PRINCIPAL DE LA APLICACIÓN
Sistema de Predicción de Cuotas Hipotecarias con Arquitectura Hexagonal
"""

import sys
import os

# Agregar src al path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports de la aplicación
from application import PrediccionService, EntrenamientoService
from infrastructure.repositories import (
    ModeloRepository,
    PrediccionRepository,
    DatosRepository,
)
from infrastructure.adapters import ExternalDataAdapter
from presentation import CLIController


def crear_directorios():
    """Crea directorios necesarios para la aplicación"""
    directorios = ["data/models", "data/predictions", "data/raw"]

    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)


def configurar_dependencias():
    """
    Configura las dependencias de la aplicación (Inyección de Dependencias)

    Esta función implementa el patrón de Composición de la arquitectura hexagonal,
    conectando todas las capas a través de los puertos e interfaces.
    """
    print("🔧 Configurando dependencias...")

    # Repositorios (Infraestructura)
    modelo_repository = ModeloRepository(data_dir="data/models")
    prediccion_repository = PrediccionRepository(data_dir="data/predictions")
    datos_repository = DatosRepository()

    # Adaptadores (Infraestructura)
    external_data_service = ExternalDataAdapter()

    # Servicios de Aplicación
    prediccion_service = PrediccionService(
        modelo_repository=modelo_repository, prediccion_repository=prediccion_repository
    )

    entrenamiento_service = EntrenamientoService(
        modelo_repository=modelo_repository, datos_repository=datos_repository
    )

    # Controlador de Presentación
    cli_controller = CLIController(
        prediccion_service=prediccion_service,
        entrenamiento_service=entrenamiento_service,
        datos_repository=datos_repository,
    )

    print("✅ Dependencias configuradas correctamente\n")

    return cli_controller


def mostrar_bienvenida():
    """Muestra mensaje de bienvenida"""
    print("\n" + "=" * 70)
    print("🏠 SISTEMA DE PREDICCIÓN DE CUOTAS HIPOTECARIAS".center(70))
    print("=" * 70)
    print("\n📐 Arquitectura: Hexagonal (Ports & Adapters)")
    print("🧠 Modelo: Híbrido (Ridge + ARIMA)")
    print("📊 Datos: Series temporales con variables macroeconómicas")
    print("\n" + "=" * 70)


def mostrar_arquitectura():
    """Muestra la estructura de la arquitectura"""
    print("\n📐 ARQUITECTURA HEXAGONAL:")
    print(
        """
    ┌─────────────────────────────────────────────────────────┐
    │                   🎯 PRESENTATION                       │
    │                  (CLI Controller)                       │
    └─────────────────────────────────────────────────────────┘
                             ↕
    ┌─────────────────────────────────────────────────────────┐
    │                   🔧 APPLICATION                        │
    │         (Services & Use Cases - Puertos Entrada)       │
    │  • PrediccionService    • EntrenamientoService         │
    └─────────────────────────────────────────────────────────┘
                             ↕
    ┌─────────────────────────────────────────────────────────┐
    │                    💎 DOMAIN                           │
    │               (Entities & Value Objects)                │
    │  • Prediccion  • DatosHipoteca  • Modelo               │
    │  • MetricasModelo  • ConfiguracionPrediccion           │
    └─────────────────────────────────────────────────────────┘
                             ↕
    ┌─────────────────────────────────────────────────────────┐
    │                  🔌 PORTS (Interfaces)                 │
    │         Input Ports          Output Ports              │
    │    • IPrediccionService   • IModeloRepository          │
    │    • IEntrenamientoService • IPrediccionRepository     │
    │                            • IDatosRepository          │
    │                            • IExternalDataService      │
    └─────────────────────────────────────────────────────────┘
                             ↕
    ┌─────────────────────────────────────────────────────────┐
    │                 🏗️ INFRASTRUCTURE                       │
    │         (Repositories & Adapters - Implementaciones)    │
    │  • ModeloRepository     • ExternalDataAdapter          │
    │  • PrediccionRepository • ModeloHibridoAdapter         │
    │  • DatosRepository                                     │
    └─────────────────────────────────────────────────────────┘
    """
    )


def main():
    """
    Función principal de la aplicación

    Flujo:
    1. Crea directorios necesarios
    2. Muestra información de la arquitectura
    3. Configura todas las dependencias (DI)
    4. Inicia la interfaz CLI
    """
    try:
        # Crear estructura de directorios
        crear_directorios()

        # Mostrar bienvenida
        mostrar_bienvenida()

        # Mostrar arquitectura (opcional - comentar si no se desea)
        # mostrar_arquitectura()

        # Configurar dependencias (Dependency Injection)
        cli_controller = configurar_dependencias()

        # Ejecutar aplicación
        cli_controller.ejecutar()

    except KeyboardInterrupt:
        print("\n\n👋 Aplicación interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
