# 📊 Análisis Financiero Personal con IA

Sistema completo de análisis de movimientos financieros con visualizaciones interactivas y proyecciones.

## 🚀 Instalación Rápida

### Opción 1: Instalación Automática

```bash
python install_dependencies.py
```

### Opción 2: Instalación Manual

```bash
pip install -r requirements.txt
```

### Opción 3: Instalación Individual

```bash
pip install pandas numpy matplotlib scikit-learn scipy openpyxl seaborn plotly kaleido
```

## 📁 Estructura del Proyecto

```
CURSO_IA/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── .gitignore                        # Archivos a ignorar en git
│
├── data/                             # Datos del proyecto
│   ├── raw/                          # Datos crudos sin procesar
│   │   └── Datos Movimientos Financieros.csv
│   └── predictions/                  # Predicciones generadas
│       └── 2025/
│
├── src/                              # Código fuente
│   ├── main.py                       # Script principal de análisis
│   │
│   ├── core/                         # Modelos de Machine Learning
│   │   ├── modelo_hibrido.py        # Modelo híbrido Ridge + ARIMA
│   │   └── modelo_series_temporales.py  # Modelos de series temporales
│   │
│   ├── services/                     # Servicios externos
│   │   └── external_data_service.py # Obtención de datos macro (IPC, DTF, UVR)
│   │
│   └── utils/                        # Utilidades
│       └── data_loader.py           # Carga y preparación de datos
│
├── notebooks/                        # Jupyter notebooks para experimentación
│   └── experiments/
│
└── ejercicios/                       # Ejercicios del curso (separados)
    └── dataset_bigdata.xlsx
```

## 💻 Uso

### Análisis de Movimientos Financieros

Ejecutar el análisis completo:

```powershell
python src/main.py
```

Este script genera:

- 📈 Gráficos de tendencias de ingresos vs gastos
- 🏷️ Categorización automática de transacciones
- 💡 Sugerencias de ahorro personalizadas
- 🐜 Análisis de "gastos hormiga"

### Predicción con Modelo Híbrido

```python
from src.core.modelo_hibrido import ModeloHibrido
from src.utils.data_loader import DataLoader

# Cargar y preparar datos
loader = DataLoader()
df = loader.enriquecer_datos(df_base)

# Entrenar modelo
modelo = ModeloHibrido(orden_arima_auto=True)
modelo.entrenar(df)

# Predecir 6 meses
predicciones = modelo.predecir_futuro(n_predicciones=6)
```

## 📊 Características

### Análisis de Tendencias

- Visualización temporal de ingresos y gastos
- Evolución del saldo bancario
- Identificación de patrones mensuales

### Categorización Inteligente

Clasifica automáticamente transacciones en:

- 🍽️ Alimentación
- 🚗 Transporte
- 🏠 Vivienda
- 💳 Servicios Financieros
- 💊 Salud
- 🎮 Entretenimiento
- 📚 Educación

### Modelos de Predicción

1. **Modelo Híbrido** (`modelo_hibrido.py`)

   - Combina regresión Ridge + ARIMA
   - Incorpora variables macroeconómicas (IPC, DTF, UVR)
   - Predicción con intervalos de confianza

2. **Series Temporales** (`modelo_series_temporales.py`)
   - ARIMA optimizado
   - Prophet (para series largas)
   - Validación automática de estacionariedad

## 🔧 Dependencias Principales

- **Análisis de datos**: pandas, numpy
- **Visualización**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Series Temporales**: statsmodels, prophet
- **Datos externos**: requests (API Banco de la República)

## 📈 Resultados

El proyecto genera:

- Reportes visuales en ventanas matplotlib
- Predicciones guardadas en `data/predictions/`
- Métricas de rendimiento de modelos
- Sugerencias de ahorro basadas en análisis

## 🤝 Contribuciones

Este es un proyecto personal de aprendizaje. Sugerencias y mejoras son bienvenidas.

## 📝 Licencia

Proyecto educativo - Uso libre para aprendizaje

## 👤 Autor

**Omar Alvarado**

- GitHub: [@oalvaradoc77g](https://github.com/oalvaradoc77g)
- Proyecto: Curso IA Financiero

---

**Rama actual**: `IA_Financiero_Debito`  
**Última actualización**: Octubre 2025
