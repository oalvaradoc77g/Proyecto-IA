"""Controlador CLI para interacción con el usuario"""

from typing import Optional
from datetime import datetime

from application import PrediccionService, EntrenamientoService
from domain.value_objects import ConfiguracionPrediccion
from infrastructure.repositories import DatosRepository


class CLIController:
    """
    Controlador de línea de comandos para la aplicación
    """

    def __init__(
        self,
        prediccion_service: PrediccionService,
        entrenamiento_service: EntrenamientoService,
        datos_repository: DatosRepository,
    ):
        self.prediccion_service = prediccion_service
        self.entrenamiento_service = entrenamiento_service
        self.datos_repository = datos_repository

    def mostrar_menu(self):
        """Muestra el menú principal"""
        print("\n" + "=" * 60)
        print("🏠 SISTEMA DE PREDICCIÓN DE CUOTAS HIPOTECARIAS")
        print("=" * 60)
        print("\n📋 MENÚ PRINCIPAL:")
        print("  1. Entrenar modelo")
        print("  2. Realizar predicciones")
        print("  3. Ver modelo activo")
        print("  4. Evaluar modelo")
        print("  5. Salir")
        print("-" * 60)

    def ejecutar(self):
        """Ejecuta el controlador CLI"""
        while True:
            self.mostrar_menu()
            opcion = input("\n👉 Seleccione una opción: ").strip()

            if opcion == "1":
                self.entrenar_modelo()
            elif opcion == "2":
                self.realizar_predicciones()
            elif opcion == "3":
                self.ver_modelo_activo()
            elif opcion == "4":
                self.evaluar_modelo()
            elif opcion == "5":
                print("\n👋 ¡Hasta luego!")
                break
            else:
                print("\n❌ Opción inválida. Intente nuevamente.")

    def entrenar_modelo(self):
        """Maneja el flujo de entrenamiento"""
        print("\n🔄 ENTRENAR MODELO")
        print("-" * 60)

        ruta = input("📁 Ingrese la ruta del archivo de datos: ").strip()

        if not ruta:
            print("❌ Ruta no válida")
            return

        try:
            # Cargar datos
            print("\n📊 Cargando datos...")
            df = self.datos_repository.cargar_datos(ruta)

            if df.empty:
                print("❌ No se pudieron cargar los datos")
                return

            print(f"✅ Datos cargados: {len(df)} registros")

            # Convertir a entidades
            datos = self.datos_repository.obtener_datos_historicos()

            # Entrenar
            print("\n🤖 Entrenando modelo...")
            modelo = self.entrenamiento_service.entrenar_modelo(datos)

            print("\n✅ MODELO ENTRENADO EXITOSAMENTE")
            print(f"   ID: {modelo.id}")
            print(f"   Tipo: {modelo.tipo}")
            print(f"   Calidad: {modelo.calidad}")
            print(f"   Métricas:")
            for metrica, valor in modelo.metricas.items():
                print(f"      {metrica}: {valor:.4f}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    def realizar_predicciones(self):
        """Maneja el flujo de predicciones"""
        print("\n🔮 REALIZAR PREDICCIONES")
        print("-" * 60)

        try:
            # Verificar modelo activo
            modelo = self.entrenamiento_service.obtener_modelo_activo()
            if not modelo:
                print("❌ No hay modelo activo. Entrene un modelo primero.")
                return

            print(f"✅ Modelo activo: {modelo.id} (Calidad: {modelo.calidad})")

            # Configurar predicción
            num_pred = input("\n📅 Número de meses a predecir (default: 6): ").strip()
            num_pred = int(num_pred) if num_pred else 6

            incluir_ic = (
                input("📊 ¿Incluir intervalos de confianza? (s/n, default: s): ")
                .strip()
                .lower()
            )
            incluir_ic = incluir_ic != "n"

            # Crear configuración
            config = ConfiguracionPrediccion(
                numero_predicciones=num_pred,
                incluir_intervalo_confianza=incluir_ic,
                incluir_componentes=True,
            )

            # Realizar predicción
            print("\n🔄 Generando predicciones...")
            predicciones = self.prediccion_service.predecir_cuotas_futuras(config)

            # Mostrar resultados
            print("\n📈 PREDICCIONES GENERADAS:")
            print("-" * 60)
            for pred in predicciones:
                print(f"\n📅 {pred.fecha.strftime('%B %Y')}")
                print(f"   Valor predicho: ${pred.valor_predicho:,.2f}")

                if pred.tiene_intervalo_confianza:
                    print(
                        f"   Rango: ${pred.intervalo_confianza_inferior:,.2f} - ${pred.intervalo_confianza_superior:,.2f}"
                    )

                print(f"   Componente lineal: ${pred.componente_lineal:,.2f}")
                print(f"   Componente temporal: ${pred.componente_temporal:,.2f}")

            print("\n✅ Predicciones guardadas exitosamente")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    def ver_modelo_activo(self):
        """Muestra información del modelo activo"""
        print("\n📊 MODELO ACTIVO")
        print("-" * 60)

        try:
            modelo = self.entrenamiento_service.obtener_modelo_activo()

            if not modelo:
                print("❌ No hay modelo activo")
                return

            print(f"\n✅ Modelo Activo:")
            print(f"   ID: {modelo.id}")
            print(f"   Tipo: {modelo.tipo}")
            print(
                f"   Fecha entrenamiento: {modelo.fecha_entrenamiento.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"   Versión: {modelo.version}")
            print(f"   Calidad: {modelo.calidad}")
            print(f"\n📊 Métricas:")
            for metrica, valor in modelo.metricas.items():
                print(f"      {metrica}: {valor:.4f}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    def evaluar_modelo(self):
        """Evalúa el modelo activo"""
        print("\n🔍 EVALUAR MODELO")
        print("-" * 60)

        try:
            modelo = self.entrenamiento_service.obtener_modelo_activo()

            if not modelo:
                print("❌ No hay modelo activo")
                return

            evaluacion = self.entrenamiento_service.evaluar_modelo(modelo.id)

            print(f"\n✅ Evaluación del Modelo {modelo.id}:")
            print(f"   Calidad: {evaluacion['calidad']}")
            print(f"   Estado: {'Activo' if evaluacion['activo'] else 'Inactivo'}")
            print(f"\n📊 Métricas:")
            for metrica, valor in evaluacion["metricas"].items():
                print(f"      {metrica}: {valor:.4f}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
