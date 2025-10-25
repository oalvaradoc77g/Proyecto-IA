# 🏗️ Arquitectura Hexagonal - Sistema de Predicción Hipotecaria

## 📋 Descripción

Este proyecto implementa un **sistema de predicción de cuotas hipotecarias** utilizando **Arquitectura Hexagonal** (también conocida como Ports & Adapters). Esta arquitectura permite:

- ✅ **Separación clara de responsabilidades**
- ✅ **Independencia del framework y tecnologías externas**
- ✅ **Facilidad para testing**
- ✅ **Mantenibilidad y escalabilidad**

## 🎯 Arquitectura

### Capas de la Aplicación

```
src/
├── domain/                 # 💎 DOMINIO (Núcleo del negocio)
│   ├── entities/          # Entidades de negocio
│   │   ├── prediccion.py
│   │   ├── datos_hipoteca.py
│   │   └── modelo.py
│   └── value_objects/     # Objetos de valor
│       ├── metricas_modelo.py
│       └── configuracion_prediccion.py
│
├── ports/                  # 🔌 PUERTOS (Interfaces)
│   ├── input_ports.py     # Servicios de entrada
│   └── output_ports.py    # Repositorios y servicios externos
│
├── application/            # 🔧 APLICACIÓN (Casos de uso)
│   ├── use_cases/         # Casos de uso específicos
│   │   ├── entrenar_modelo_use_case.py
│   │   └── predecir_cuotas_use_case.py
│   ├── prediccion_service.py
│   └── entrenamiento_service.py
│
├── infrastructure/         # 🏗️ INFRAESTRUCTURA (Implementaciones)
│   ├── repositories/      # Persistencia
│   │   ├── modelo_repository.py
│   │   ├── prediccion_repository.py
│   │   └── datos_repository.py
│   └── adapters/          # Adaptadores externos
│       ├── external_data_adapter.py
│       └── modelo_hibrido_adapter.py
│
├── presentation/           # 🎯 PRESENTACIÓN (UI/CLI)
│   └── cli_controller.py
│
└── main.py                # 🚀 PUNTO DE ENTRADA
```

## 🔄 Flujo de Datos

```
Usuario
   ↓
[CLI Controller] ← Capa de Presentación
   ↓
[Services] ← Puertos de Entrada (Interfaces)
   ↓
[Use Cases] ← Lógica de Aplicación
   ↓
[Domain Entities] ← Lógica de Negocio (Núcleo)
   ↓
[Repositories/Adapters] ← Puertos de Salida (Interfaces)
   ↓
[External Systems] ← Base de datos, APIs, etc.
```

## 🚀 Uso del Sistema

### Iniciar la Aplicación

```powershell
python src/main.py
```

### Menú Principal

```
1. Entrenar modelo
2. Realizar predicciones
3. Ver modelo activo
4. Evaluar modelo
5. Salir
```

### Ejemplo de Uso

#### 1. Entrenar Modelo

```powershell
Opción: 1
Ruta: C:\path\to\hipoteca_datos.xlsx
```

El sistema:

- Carga los datos históricos
- Entrena el modelo híbrido (Ridge + ARIMA)
- Valida las métricas
- Guarda el modelo en `data/models/`

#### 2. Realizar Predicciones

```powershell
Opción: 2
Número de meses: 6
Intervalos de confianza: s
```

El sistema:

- Usa el modelo activo
- Genera predicciones para N meses
- Muestra componentes (lineal + temporal)
- Guarda las predicciones en `data/predictions/`

## 📦 Componentes Principales

### Dominio (Domain)

**Entidades:**

- `Prediccion`: Representa una predicción de cuota
- `DatosHipoteca`: Datos mensuales de la hipoteca
- `Modelo`: Modelo de ML entrenado

**Value Objects:**

- `MetricasModelo`: Métricas de evaluación (R², MSE, etc.)
- `ConfiguracionPrediccion`: Configuración inmutable para predicciones

### Puertos (Ports)

**Input Ports (Servicios):**

- `IPrediccionService`: Interfaz para predicciones
- `IEntrenamientoService`: Interfaz para entrenamiento

**Output Ports (Repositorios):**

- `IModeloRepository`: Persistencia de modelos
- `IPrediccionRepository`: Persistencia de predicciones
- `IDatosRepository`: Acceso a datos históricos
- `IExternalDataService`: Servicios externos (APIs)

### Aplicación (Application)

**Use Cases:**

- `EntrenarModeloUseCase`: Entrenar nuevo modelo
- `PredecirCuotasUseCase`: Generar predicciones

**Services:**

- `PrediccionService`: Implementa `IPrediccionService`
- `EntrenamientoService`: Implementa `IEntrenamientoService`

### Infraestructura (Infrastructure)

**Repositories:**

- `ModeloRepository`: Guarda modelos en JSON
- `PrediccionRepository`: Guarda predicciones en JSON/Excel
- `DatosRepository`: Carga datos con pandas

**Adapters:**

- `ExternalDataAdapter`: Conecta con BanRep API
- `ModeloHibridoAdapter`: Envuelve el modelo ML existente

## 🎨 Principios Aplicados

### 1. Inversión de Dependencias (DIP)

Las capas externas dependen de las internas a través de interfaces:

```python
# ✅ CORRECTO
class PrediccionService(IPrediccionService):
    def __init__(self, modelo_repo: IModeloRepository):
        self.modelo_repo = modelo_repo

# ❌ INCORRECTO
class PrediccionService:
    def __init__(self):
        self.modelo_repo = ModeloRepository()  # Dependencia concreta
```

### 2. Separación de Responsabilidades (SRP)

Cada clase tiene una única responsabilidad:

- `Prediccion`: Representa una predicción
- `PrediccionRepository`: Persiste predicciones
- `PredecirCuotasUseCase`: Lógica de predicción

### 3. Abierto/Cerrado (OCP)

Fácil extensión sin modificar código existente:

```python
# Nueva implementación sin cambiar interfaces
class PrediccionRepositorySQL(IPrediccionRepository):
    # Implementación con SQL en lugar de JSON
    pass
```

## 🔧 Inyección de Dependencias

El archivo `main.py` configura todas las dependencias:

```python
def configurar_dependencias():
    # Repositorios
    modelo_repo = ModeloRepository()
    prediccion_repo = PrediccionRepository()

    # Servicios
    prediccion_service = PrediccionService(
        modelo_repository=modelo_repo,
        prediccion_repository=prediccion_repo
    )

    # Controlador
    cli = CLIController(prediccion_service)

    return cli
```

## 📊 Estructura de Datos

### Datos de Entrada

```json
{
  "fecha": "2025-01-31",
  "capital": 1200000,
  "gastos_fijos": 50000,
  "total_mensual": 1350000,
  "tasa_uvr": 395.002,
  "tasa_dtf": 7.12,
  "inflacion_ipc": 150.99,
  "tipo_pago": "Ordinario"
}
```

### Predicción Generada

```json
{
  "fecha": "2025-11-01",
  "valor_predicho": 1355000,
  "componente_lineal": 1350000,
  "componente_temporal": 5000,
  "intervalo_confianza_inferior": 1287250,
  "intervalo_confianza_superior": 1422750,
  "metricas": {
    "r2": 0.85,
    "mse": 1000.0,
    "mae": 25.0
  }
}
```

## 🧪 Testing

La arquitectura facilita el testing con mocks:

```python
# Mock del repositorio
class MockModeloRepository(IModeloRepository):
    def obtener_activo(self):
        return Modelo(id="test", tipo="hibrido", ...)

# Test del servicio
def test_predecir():
    mock_repo = MockModeloRepository()
    service = PrediccionService(mock_repo)

    predicciones = service.predecir_cuotas_futuras(config)
    assert len(predicciones) == 6
```

## 🔄 Extensibilidad

### Agregar Nueva Fuente de Datos

```python
# 1. Crear adaptador
class NewAPIAdapter(IExternalDataService):
    def obtener_tasa_uvr(self):
        # Implementación específica
        pass

# 2. Configurar en main.py
external_service = NewAPIAdapter()
```

### Agregar Nueva Presentación (API REST)

```python
# 1. Crear controlador
class APIController:
    def __init__(self, prediccion_service):
        self.service = prediccion_service

    @app.post("/predicciones")
    def crear_prediccion(self, config):
        return self.service.predecir_cuotas_futuras(config)

# 2. Sin cambios en dominio ni aplicación
```

## 📂 Archivos Generados

```
data/
├── models/
│   ├── index.json
│   └── {modelo-id}.json
└── predictions/
    └── 2025/
        └── october/
            └── predicciones_20251025_143022.json
```

## 🎓 Beneficios de esta Arquitectura

1. **Testeable**: Fácil crear tests unitarios con mocks
2. **Mantenible**: Cambios localizados en capas específicas
3. **Escalable**: Agregar funcionalidades sin romper código existente
4. **Independiente**: No acoplado a frameworks o bases de datos
5. **Clara**: Separación explícita de responsabilidades

## 🔍 Comparación con Código Legacy

### Antes (Monolítico)

```python
# Todo mezclado
class ModeloHipoteca:
    def cargar_datos(self, path):
        df = pd.read_excel(path)  # IO
        # Lógica de negocio
        # Persistencia
        # Todo junto
```

### Después (Hexagonal)

```python
# Separado por responsabilidades
# Dominio
class Prediccion: pass

# Puerto
class IPrediccionService(ABC): pass

# Aplicación
class PrediccionService(IPrediccionService): pass

# Infraestructura
class PrediccionRepository: pass
```

## 📝 Notas Adicionales

- El sistema mantiene compatibilidad con el código legacy (`core/`, `utils/`, `services/`)
- Los adaptadores envuelven el código existente sin modificarlo
- Migración gradual: puedes usar ambas arquitecturas en paralelo

---

**Autor**: Sistema de IA  
**Versión**: 1.0.0  
**Fecha**: Octubre 2025
