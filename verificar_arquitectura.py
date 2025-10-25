"""
Script de verificación de la arquitectura hexagonal
Ejecutar para verificar que todos los componentes están correctamente configurados
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def verificar_imports():
    """Verifica que todos los módulos se pueden importar"""
    print("🔍 Verificando imports de la arquitectura hexagonal...")

    intentos = []

    # Domain
    intentos.append(("Domain - Entities", lambda: __import__("domain.entities")))
    intentos.append(
        ("Domain - Value Objects", lambda: __import__("domain.value_objects"))
    )

    # Ports
    intentos.append(("Ports - Input", lambda: __import__("ports.input_ports")))
    intentos.append(("Ports - Output", lambda: __import__("ports.output_ports")))

    # Application
    intentos.append(("Application - Services", lambda: __import__("application")))
    intentos.append(
        ("Application - Use Cases", lambda: __import__("application.use_cases"))
    )

    # Infrastructure
    intentos.append(
        (
            "Infrastructure - Repositories",
            lambda: __import__("infrastructure.repositories"),
        )
    )
    intentos.append(
        ("Infrastructure - Adapters", lambda: __import__("infrastructure.adapters"))
    )

    # Presentation
    intentos.append(("Presentation - CLI", lambda: __import__("presentation")))

    errores = []
    for nombre, func in intentos:
        try:
            func()
            print(f"  ✅ {nombre}")
        except Exception as e:
            print(f"  ❌ {nombre}: {e}")
            errores.append((nombre, e))

    return len(errores) == 0, errores


def verificar_estructura():
    """Verifica que la estructura de directorios existe"""
    print("\n📁 Verificando estructura de directorios...")

    directorios_requeridos = [
        "src/domain/entities",
        "src/domain/value_objects",
        "src/ports",
        "src/application/use_cases",
        "src/infrastructure/repositories",
        "src/infrastructure/adapters",
        "src/presentation",
        "data/models",
        "data/predictions",
        "data/raw",
    ]

    errores = []
    for directorio in directorios_requeridos:
        if os.path.exists(directorio):
            print(f"  ✅ {directorio}")
        else:
            print(f"  ❌ {directorio} - NO EXISTE")
            errores.append(directorio)

    return len(errores) == 0, errores


def verificar_dependencias():
    """Verifica que las dependencias necesarias están instaladas"""
    print("\n📦 Verificando dependencias...")

    dependencias = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "statsmodels",
        "joblib",
        "requests",
        "dateutil",
    ]

    errores = []
    for dep in dependencias:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - NO INSTALADO")
            errores.append(dep)

    return len(errores) == 0, errores


def crear_datos_prueba():
    """Crea datos de prueba para verificar el flujo completo"""
    print("\n🧪 Creando datos de prueba...")

    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Crear datos simulados
        np.random.seed(42)
        n = 10

        fechas = pd.date_range(start="2025-01-01", periods=n, freq="M")
        datos = {
            "fecha": fechas,
            "capital": 1_000_000 + np.cumsum(np.random.normal(0, 10000, n)),
            "gastos_fijos": 50_000 + np.random.normal(0, 1000, n),
            "total_mensual": 1_200_000 + np.random.normal(0, 5000, n),
            "tasa_uvr": np.full(n, 395.002),
            "tasa_dtf": np.full(n, 7.12),
            "inflacion_ipc": np.full(n, 150.99),
            "tipo_pago": np.random.choice(["Ordinario", "Abono extra"], n),
        }

        df = pd.DataFrame(datos)

        # Guardar en data/raw
        os.makedirs("data/raw", exist_ok=True)
        ruta = "data/raw/datos_prueba.xlsx"
        df.to_excel(ruta, index=False)

        print(f"  ✅ Datos de prueba creados en {ruta}")
        return True, ruta

    except Exception as e:
        print(f"  ❌ Error creando datos: {e}")
        return False, None


def mostrar_resumen(resultados):
    """Muestra resumen de la verificación"""
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 70)

    total_checks = len(resultados)
    exitosos = sum(1 for r in resultados.values() if r[0])

    for nombre, (exito, detalles) in resultados.items():
        estado = "✅ EXITOSO" if exito else "❌ FALLIDO"
        print(f"\n{nombre}: {estado}")
        if not exito and detalles:
            print(f"  Errores: {len(detalles)}")
            for detalle in detalles[:3]:  # Mostrar máximo 3 errores
                print(f"    - {detalle}")

    print("\n" + "=" * 70)
    print(f"Resultado: {exitosos}/{total_checks} verificaciones exitosas")
    print("=" * 70)

    if exitosos == total_checks:
        print("\n🎉 ¡SISTEMA LISTO PARA USAR!")
        print("\nPróximos pasos:")
        print("  1. Ejecutar: python src/main.py")
        print("  2. Seleccionar opción 1 para entrenar modelo")
        print("  3. Usar ruta: data/raw/datos_prueba.xlsx")
        return True
    else:
        print("\n⚠️  SISTEMA NO ESTÁ COMPLETAMENTE CONFIGURADO")
        print("\nAcciones sugeridas:")
        if not resultados["Dependencias"][0]:
            print("  - Ejecutar: pip install -r requirements.txt")
        if not resultados["Estructura"][0]:
            print("  - Ejecutar: python src/main.py (creará directorios)")
        return False


def main():
    """Función principal de verificación"""
    print("=" * 70)
    print("🏗️  VERIFICACIÓN DE ARQUITECTURA HEXAGONAL")
    print("=" * 70)

    resultados = {}

    # Verificar estructura
    exito, errores = verificar_estructura()
    resultados["Estructura"] = (exito, errores)

    # Verificar dependencias
    exito, errores = verificar_dependencias()
    resultados["Dependencias"] = (exito, errores)

    # Verificar imports
    exito, errores = verificar_imports()
    resultados["Imports"] = (exito, errores)

    # Crear datos de prueba
    exito, ruta = crear_datos_prueba()
    resultados["Datos de Prueba"] = (exito, [ruta] if ruta else [])

    # Mostrar resumen
    sistema_ok = mostrar_resumen(resultados)

    return 0 if sistema_ok else 1


if __name__ == "__main__":
    sys.exit(main())
