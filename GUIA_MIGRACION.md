# 🔄 Guía de Migración: Legacy → Hexagonal

## 📋 Cómo Migrar Funcionalidades del Código Legacy

Esta guía explica cómo trasladar funcionalidades del código legacy (`src/Proyectos/prediccion_hipoteca.py`) a la nueva arquitectura hexagonal.

## 🎯 Estrategia de Migración

### Opción 1: Migración Gradual (Recomendada)

✅ **Ventaja**: No rompe funcionalidad existente  
✅ **Estrategia**: Ambos sistemas coexisten  
✅ **Tiempo**: Migración incremental

### Opción 2: Migración Completa

⚠️ **Ventaja**: Sistema completamente nuevo  
⚠️ **Riesgo**: Requiere reescritura completa  
⚠️ **Tiempo**: Mayor esfuerzo inicial

## 📊 Ejemplo Práctico: Migrar Funcionalidad

### Código Legacy Original

```python
# src/Proyectos/prediccion_hipoteca.py
class ModeloHipoteca:
    def predecir(self, capital, gastos_fijos, seguros):
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None

        ejemplo = np.array([[capital, gastos_fijos]])
        pred = self.modelo.predict(ejemplo)[0]
        print(f"Total Mensual Predicho: {pred:,.2f}")
        return pred
```

### Paso 1: Crear Entidad de Dominio

```python
# src/domain/entities/solicitud_prediccion.py
from dataclasses import dataclass

@dataclass
class SolicitudPrediccion:
    """Representa una solicitud de predicción"""
    capital: float
    gastos_fijos: float

    def __post_init__(self):
        if self.capital < 0:
            raise ValueError("Capital no puede ser negativo")
        if self.gastos_fijos < 0:
            raise ValueError("Gastos fijos no pueden ser negativos")
```

### Paso 2: Crear Caso de Uso

```python
# src/application/use_cases/predecir_cuota_unica_use_case.py
from domain.entities import SolicitudPrediccion, Prediccion

class PredecirCuotaUnicaUseCase:
    def __init__(self, modelo_adapter):
        self.modelo_adapter = modelo_adapter

    def execute(self, solicitud: SolicitudPrediccion) -> Prediccion:
        # Validar
        if not self.modelo_adapter.esta_entrenado():
            raise ValueError("Modelo no entrenado")

        # Predecir
        resultado = self.modelo_adapter.predecir_cuota(
            capital=solicitud.capital,
            gastos_fijos=solicitud.gastos_fijos
        )

        # Crear entidad de dominio
        return Prediccion(
            fecha=datetime.now(),
            valor_predicho=resultado['valor'],
            componente_lineal=resultado['componente_lineal'],
            componente_temporal=resultado['componente_temporal']
        )
```

### Paso 3: Actualizar Adaptador

```python
# src/infrastructure/adapters/modelo_hibrido_adapter.py
class ModeloHibridoAdapter:
    # ... código existente ...

    def predecir_cuota(self, capital: float, gastos_fijos: float) -> dict:
        """Nueva función que envuelve el método legacy"""
        # Llamar al código legacy
        valor = self.modelo.predecir(capital, gastos_fijos, 0)

        # Retornar en formato estructurado
        return {
            'valor': valor,
            'componente_lineal': valor * 0.9,  # Aproximado
            'componente_temporal': valor * 0.1
        }

    def esta_entrenado(self) -> bool:
        """Verifica si el modelo está entrenado"""
        return self.entrenado
```

### Paso 4: Integrar en el Servicio

```python
# src/application/prediccion_service.py
class PrediccionService(IPrediccionService):
    # ... código existente ...

    def predecir_cuota_unica(
        self,
        capital: float,
        gastos_fijos: float
    ) -> Prediccion:
        """Nueva funcionalidad integrada"""
        solicitud = SolicitudPrediccion(
            capital=capital,
            gastos_fijos=gastos_fijos
        )

        use_case = PredecirCuotaUnicaUseCase(self.modelo_adapter)
        return use_case.execute(solicitud)
```

### Paso 5: Exponer en CLI

```python
# src/presentation/cli_controller.py
class CLIController:
    # ... código existente ...

    def predecir_cuota_unica(self):
        """Nueva opción en el menú"""
        print("\n💰 PREDECIR CUOTA ÚNICA")
        print("-" * 60)

        try:
            capital = float(input("Capital: "))
            gastos_fijos = float(input("Gastos fijos: "))

            prediccion = self.prediccion_service.predecir_cuota_unica(
                capital,
                gastos_fijos
            )

            print(f"\n✅ Cuota predicha: ${prediccion.valor_predicho:,.2f}")

        except Exception as e:
            print(f"❌ Error: {e}")
```

## 🔄 Patrón de Migración Completo

### Legacy

```python
# Código mezclado: IO + Lógica + Persistencia
def cargar_y_predecir(path):
    df = pd.read_excel(path)           # IO
    X = df[['capital', 'gastos']]      # Lógica
    modelo.fit(X, y)                   # ML
    pred = modelo.predict(X_new)       # Predicción
    joblib.dump(modelo, 'mod.pkl')    # Persistencia
    return pred
```

### Hexagonal

```python
# 1. DOMAIN: Entidades
class DatosHipoteca: pass
class Prediccion: pass

# 2. PORT: Interfaces
class IDatosRepository(ABC):
    def cargar(self, path): pass

# 3. INFRASTRUCTURE: Implementación
class DatosRepositoryExcel(IDatosRepository):
    def cargar(self, path):
        return pd.read_excel(path)

# 4. APPLICATION: Caso de uso
class EntrenarYPredecirUseCase:
    def __init__(self, datos_repo, modelo_repo):
        self.datos_repo = datos_repo
        self.modelo_repo = modelo_repo

    def execute(self, path, X_new):
        # Cargar datos
        datos = self.datos_repo.cargar(path)

        # Entrenar
        modelo = self.entrenar(datos)

        # Guardar
        self.modelo_repo.guardar(modelo)

        # Predecir
        return self.predecir(modelo, X_new)
```

## 📋 Checklist de Migración

### Para cada funcionalidad legacy:

- [ ] **1. Identificar responsabilidades**

  - ¿Qué hace la función?
  - ¿Cuáles son sus dependencias?
  - ¿Qué retorna?

- [ ] **2. Crear entidades de dominio**

  - Datos de entrada → Value Objects
  - Datos de salida → Entities
  - Lógica de negocio → Domain Services

- [ ] **3. Definir puertos (interfaces)**

  - ¿Qué operaciones necesita?
  - ¿Qué dependencias externas tiene?

- [ ] **4. Crear caso de uso**

  - Orquesta las entidades
  - Usa puertos para I/O
  - Sin dependencias de frameworks

- [ ] **5. Implementar adaptadores**

  - Conecta con código legacy
  - Implementa puertos de salida
  - Transforma datos

- [ ] **6. Integrar en servicio**

  - Implementa puerto de entrada
  - Coordina casos de uso

- [ ] **7. Exponer en presentación**
  - CLI, API, GUI, etc.
  - Usa servicios de aplicación

## 🎯 Ejemplos Específicos

### Migrar: Análisis de Multicolinealidad

**Legacy**:

```python
def analizar_multicolinealidad(self, df):
    X = df[self.columnas]
    corr_matrix = X.corr()
    print(corr_matrix.round(3))
    # ... más lógica ...
```

**Hexagonal**:

```python
# Domain Entity
@dataclass
class AnalisisMulticolinealidad:
    matriz_correlacion: pd.DataFrame
    vif_scores: Dict[str, float]
    tiene_multicolinealidad: bool

# Use Case
class AnalizarMulticolinealidadUseCase:
    def execute(self, datos: List[DatosHipoteca]) -> AnalisisMulticolinealidad:
        # Lógica de análisis
        pass

# Service
class AnalisisService:
    def analizar_multicolinealidad(self, datos):
        use_case = AnalizarMulticolinealidadUseCase()
        return use_case.execute(datos)
```

### Migrar: Validación de Predicción

**Legacy**:

```python
def validar_prediccion(self, prediccion, valor_real):
    error = abs((valor_real - prediccion) / valor_real)
    if error > 0.15:
        print("⚠️ Error significativo")
        return False
    return True
```

**Hexagonal**:

```python
# Domain Value Object
@dataclass(frozen=True)
class ResultadoValidacion:
    error_relativo: float
    es_valido: bool
    mensaje: str

    @staticmethod
    def crear(prediccion: float, valor_real: float) -> 'ResultadoValidacion':
        error = abs((valor_real - prediccion) / valor_real)
        es_valido = error <= 0.15
        mensaje = "Predicción válida" if es_valido else "Error significativo"
        return ResultadoValidacion(error, es_valido, mensaje)

# Use Case
class ValidarPrediccionUseCase:
    def execute(self, prediccion: Prediccion, valor_real: float) -> ResultadoValidacion:
        return ResultadoValidacion.crear(
            prediccion.valor_predicho,
            valor_real
        )
```

## 🛠️ Herramientas de Migración

### Script Helper para Migración

```python
# scripts/migration_helper.py
"""
Ayuda a identificar funcionalidades para migrar
"""

def analizar_clase_legacy(clase):
    """Analiza una clase legacy y sugiere arquitectura"""
    print(f"Analizando: {clase.__name__}")

    # Identificar métodos
    metodos = [m for m in dir(clase) if not m.startswith('_')]

    print("\n📋 Métodos encontrados:")
    for metodo in metodos:
        print(f"  - {metodo}")
        # Sugerir categoría (Domain, Use Case, etc.)
        categoria = clasificar_metodo(metodo)
        print(f"    → Sugerido para: {categoria}")

def clasificar_metodo(nombre):
    """Clasifica un método según su nombre"""
    if 'crear' in nombre or 'generar' in nombre:
        return "Domain Entity"
    elif 'validar' in nombre or 'verificar' in nombre:
        return "Domain Service"
    elif 'cargar' in nombre or 'guardar' in nombre:
        return "Repository (Infrastructure)"
    elif 'predecir' in nombre or 'entrenar' in nombre:
        return "Use Case (Application)"
    else:
        return "Revisar manualmente"

# Uso
from Proyectos.prediccion_hipoteca import ModeloHipoteca
analizar_clase_legacy(ModeloHipoteca)
```

## 📚 Recursos Adicionales

### Patrones de Migración

1. **Strangler Fig Pattern**

   - Migrar incrementalmente
   - Nueva funcionalidad en hexagonal
   - Legacy para funcionalidad antigua
   - Migrar gradualmente

2. **Adapter Pattern**

   - Envolver código legacy
   - Exponer interfaz limpia
   - Sin reescribir inmediatamente

3. **Anti-Corruption Layer**
   - Capa entre legacy y hexagonal
   - Traduce modelos
   - Protege dominio

### Orden Recomendado de Migración

1. ✅ **Primero**: Funcionalidades críticas

   - Predicción de cuotas
   - Entrenamiento de modelo
   - Evaluación

2. ✅ **Segundo**: Funcionalidades de soporte

   - Carga de datos
   - Validaciones
   - Análisis

3. ✅ **Tercero**: Funcionalidades auxiliares
   - Visualizaciones
   - Reportes
   - Utilidades

## ⚠️ Errores Comunes

### ❌ Error 1: Mezclar Capas

```python
# MAL: Domain depende de Infrastructure
class Prediccion:
    def guardar(self):
        # ❌ Acceso directo a DB
        db.save(self)
```

```python
# BIEN: Infrastructure depende de Domain
class PrediccionRepository:
    def guardar(self, prediccion: Prediccion):
        # ✅ Repository guarda entidad
        db.save(prediccion.to_dict())
```

### ❌ Error 2: Lógica en Adaptadores

```python
# MAL: Lógica de negocio en adaptador
class ModeloAdapter:
    def predecir(self, datos):
        # ❌ Validación de negocio aquí
        if datos.capital < 0:
            raise ValueError("...")
        # ❌ Cálculos de negocio aquí
        resultado = datos.capital * 1.05
```

```python
# BIEN: Lógica en Domain/Application
class Prediccion:
    def __post_init__(self):
        # ✅ Validación en entidad
        if self.capital < 0:
            raise ValueError("...")

class PredecirUseCase:
    def execute(self, solicitud):
        # ✅ Lógica en caso de uso
        resultado = self.calcular_cuota(solicitud)
```

## 🎉 Conclusión

La migración a arquitectura hexagonal:

- ✅ Puede ser **gradual** (no todo a la vez)
- ✅ **Preserva** funcionalidad existente
- ✅ **Mejora** mantenibilidad
- ✅ **Facilita** testing
- ✅ **Permite** evolución futura

---

**Próximo paso**: Identificar una funcionalidad del código legacy y migrarla siguiendo esta guía.
